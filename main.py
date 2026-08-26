from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import sys
import time
import tomllib
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import feedparser
import requests
from openai import OpenAI


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
DOCS_DIR = ROOT / "docs"
SEEN_PATH = OUTPUT_DIR / "seen_dois.json"
BACKUP_PATH = OUTPUT_DIR / "backup_papers.json"
REPORT_PATH = OUTPUT_DIR / "daily_read.md"
HTML_PATH = DOCS_DIR / "index.html"
USER_AGENT = "Magnetocardiography-101-paper-digest/1.0 (GitHub Actions)"


@dataclass
class Paper:
    title: str
    abstract: str
    authors: list[str]
    published: str
    source: str
    url: str
    doi: str = ""
    source_id: str = ""
    affiliations: list[str] | None = None
    score: int = 0
    grade: str = "B"
    matched_terms: list[str] | None = None
    digest: dict[str, str] | None = None

    @property
    def stable_key(self) -> str:
        if self.doi:
            return normalize_doi(self.doi)
        if self.source_id:
            return f"{self.source.lower()}:{self.source_id.lower()}"
        fingerprint = hashlib.sha256(self.title.lower().encode("utf-8")).hexdigest()[:20]
        return f"title:{fingerprint}"


def load_config() -> dict[str, Any]:
    with (ROOT / "config.toml").open("rb") as handle:
        return tomllib.load(handle)


def normalize_doi(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", value)
    return re.sub(r"^doi:\s*", "", value)


def request_get(url: str, **kwargs: Any) -> requests.Response:
    headers = kwargs.pop("headers", {})
    headers.setdefault("User-Agent", USER_AGENT)
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = requests.get(url, headers=headers, timeout=30, **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(2**attempt)
    raise RuntimeError(f"Request failed after retries: {url}: {last_error}")


def text_of(node: ET.Element | None) -> str:
    if node is None:
        return ""
    return " ".join("".join(node.itertext()).split())


def fetch_pubmed(config: dict[str, Any], start_date: date) -> list[Paper]:
    query = " ".join(config["search"]["pubmed_query"].split())
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": config["sources"]["pubmed_retmax"],
        "sort": "pub date",
        "mindate": start_date.isoformat(),
        "maxdate": date.today().isoformat(),
        "datetype": "pdat",
    }
    search = request_get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi", params=params
    ).json()
    ids = search.get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []

    xml_text = request_get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
        params={"db": "pubmed", "id": ",".join(ids), "retmode": "xml"},
    ).text
    root = ET.fromstring(xml_text)
    papers: list[Paper] = []
    for article in root.findall(".//PubmedArticle"):
        citation = article.find("MedlineCitation")
        pmid = text_of(citation.find("PMID") if citation is not None else None)
        journal_article = article.find(".//Article")
        if journal_article is None:
            continue
        title = text_of(journal_article.find("ArticleTitle"))
        abstract = " ".join(text_of(x) for x in journal_article.findall("Abstract/AbstractText"))
        authors = []
        affiliations = []
        for author in journal_article.findall("AuthorList/Author"):
            collective = text_of(author.find("CollectiveName"))
            personal = " ".join(
                part for part in [text_of(author.find("ForeName")), text_of(author.find("LastName"))] if part
            )
            if collective or personal:
                authors.append(collective or personal)
            for affiliation in author.findall("AffiliationInfo/Affiliation"):
                value = text_of(affiliation)
                if value and value not in affiliations:
                    affiliations.append(value)
        doi = ""
        for article_id in article.findall(".//ArticleId"):
            if article_id.attrib.get("IdType") == "doi":
                doi = text_of(article_id)
                break
        pub_date = journal_article.find("Journal/JournalIssue/PubDate")
        published = text_of(pub_date) or date.today().isoformat()
        if title:
            papers.append(
                Paper(
                    title=title,
                    abstract=abstract,
                    authors=authors,
                    published=published,
                    source="PubMed",
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    doi=doi,
                    source_id=pmid,
                    affiliations=affiliations,
                )
            )
    return papers


def fetch_arxiv(config: dict[str, Any], start_date: date) -> list[Paper]:
    query = " ".join(config["search"]["arxiv_query"].split())
    response = request_get(
        "https://export.arxiv.org/api/query",
        params={
            "search_query": query,
            "start": 0,
            "max_results": config["sources"]["arxiv_max_results"],
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        },
    )
    feed = feedparser.parse(response.text)
    papers: list[Paper] = []
    for entry in feed.entries:
        published = entry.get("published", "")
        try:
            published_date = datetime.fromisoformat(published.replace("Z", "+00:00")).date()
        except ValueError:
            published_date = date.today()
        if published_date < start_date:
            continue
        arxiv_id = entry.id.rsplit("/", 1)[-1]
        doi = entry.get("arxiv_doi", "")
        arxiv_affiliations = []
        for author in entry.get("authors", []):
            value = author.get("arxiv_affiliation", "") if hasattr(author, "get") else ""
            if value and value not in arxiv_affiliations:
                arxiv_affiliations.append(value)
        papers.append(
            Paper(
                title=" ".join(entry.title.split()),
                abstract=" ".join(entry.summary.split()),
                authors=[author.name for author in entry.get("authors", [])],
                published=published_date.isoformat(),
                source="arXiv",
                url=entry.link,
                doi=doi,
                source_id=arxiv_id,
                affiliations=arxiv_affiliations,
            )
        )
    return papers


def deduplicate(papers: list[Paper]) -> list[Paper]:
    chosen: dict[str, Paper] = {}
    for paper in papers:
        key = paper.stable_key
        existing = chosen.get(key)
        if existing is None or (paper.source == "PubMed" and existing.source != "PubMed"):
            chosen[key] = paper
    return list(chosen.values())


def score_paper(paper: Paper, config: dict[str, Any]) -> None:
    text = f"{paper.title} {paper.abstract}".lower()
    matched: list[str] = []
    score = 0
    for group in ("s_terms", "a_terms"):
        for term, weight in config["scoring"][group].items():
            if term.lower() in text:
                score += int(weight)
                matched.append(term)
    paper.score = score
    paper.matched_terms = matched
    has_s_anchor = any(term.lower() in text for term in config["scoring"]["s_anchor_terms"])
    if score >= int(config["scoring"]["s_threshold"]) and has_s_anchor:
        paper.grade = "S"
    elif score >= int(config["scoring"]["a_threshold"]):
        paper.grade = "A"
    else:
        paper.grade = "B"


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def generate_digests(papers: list[Paper], config: dict[str, Any], dry_run: bool) -> None:
    if not papers:
        return
    if dry_run:
        for paper in papers:
            paper.digest = {
                "takeaway": "Dry-run：已通过分级规则，未调用语言模型。",
                "methods": "请在正式运行后查看模型生成的方法概述。",
                "relevance": "与 OPM-MCG/MEG 文献情报主题相关。",
                "limitations": "正式运行后补充研究局限与可借鉴点。",
            }
        return

    api_key = os.getenv("OPENCODE_ZEN_API_KEY")
    if not api_key:
        raise RuntimeError("OPENCODE_ZEN_API_KEY is not configured")
    payload = [
        {
            "id": index,
            "title": paper.title,
            "abstract": paper.abstract[: int(config["project"]["abstract_max_chars"])],
            "grade": paper.grade,
            "score": paper.score,
        }
        for index, paper in enumerate(papers)
    ]
    prompt = (
        "你是OPM-MCG/MEG文献情报编辑。只依据题目和摘要生成中文速读，不得补充事实。"
        "返回JSON对象{items:[{id,takeaway,methods,relevance,limitations}]}。"
        "结论180字内，方法140字内，相关性140字内，局限或可借鉴点100字内。输入：\n"
        + json.dumps(payload, ensure_ascii=False)
    )
    client = OpenAI(api_key=api_key, base_url=config["project"]["base_url"])
    response = client.chat.completions.create(
        model=config["project"]["model"],
        messages=[
            {"role": "system", "content": "你是严谨的OPM-MCG/MEG文献情报编辑。"},
            {"role": "user", "content": prompt},
        ],
        max_tokens=int(config["project"]["max_output_tokens"]),
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE)
    decoded = json.loads(raw)
    items = decoded.get("items", decoded) if isinstance(decoded, dict) else decoded
    by_id = {int(item["id"]): item for item in items}
    for index, paper in enumerate(papers):
        item = by_id.get(index, {})
        paper.digest = {
            "takeaway": str(item.get("takeaway", "模型未返回核心结论。")),
            "methods": str(item.get("methods", "模型未返回方法概述。")),
            "relevance": str(item.get("relevance", "模型未返回相关性说明。")),
            "limitations": str(item.get("limitations", "模型未返回局限或可借鉴点。")),
        }


def author_line(paper: Paper) -> str:
    if not paper.authors:
        return "作者信息未提供"
    shown = ", ".join(paper.authors[:5])
    return shown + (" 等" if len(paper.authors) > 5 else "")


def affiliation_line(paper: Paper) -> str:
    values = paper.affiliations or []
    return "；".join(values[:3]) if values else "未提供"


def country_line(paper: Paper) -> str:
    text = " ".join(paper.affiliations or [])
    countries = [
        "China", "中国", "United States", "USA", "美国", "United Kingdom", "英国",
        "Canada", "加拿大", "Australia", "澳大利亚", "Germany", "德国", "France", "法国",
        "Japan", "日本", "South Korea", "韩国", "Singapore", "新加坡", "Switzerland", "瑞士",
    ]
    found = []
    for country in countries:
        if country.lower() in text.lower() and country not in found:
            found.append(country)
    return "、".join(found) if found else "未提供"


def render_markdown(selected: list[Paper], backups: list[Paper], run_day: str) -> str:
    lines = [
        "# OPM 文献每日速读",
        "",
        f"> 🕒今日速读，最多7篇，预估阅读8-10分钟；更多候选论文见仓库 backup_papers.json",
        "",
        f"更新日期：{run_day}",
        "",
    ]
    if not selected:
        lines.extend(["今日无高优先级OPM论文，可查阅备份浏览次要文献", ""])
    for index, paper in enumerate(selected, 1):
        digest = paper.digest or {}
        lines.extend(
            [
                f"## {index}. [{paper.title}]({paper.url})",
                "",
                f"**等级/评分：** {paper.grade} / {paper.score}　 **来源：** {paper.source}　 **日期：** {paper.published}",
                "",
                f"**作者：** {author_line(paper)}",
                "",
                f"**单位：** {affiliation_line(paper)}",
                "",
                f"**国家：** {country_line(paper)}",
                "",
                f"**一句话结论：** {digest.get('takeaway', '')}",
                "",
                f"**方法概述：** {digest.get('methods', '')}",
                "",
                f"**与你的方向的关系：** {digest.get('relevance', '')}",
                "",
                f"**局限/可借鉴点：** {digest.get('limitations', '')}",
                "",
                f"**摘要摘录：** {paper.abstract[:700]}{'……' if len(paper.abstract) > 700 else ''}",
                "",
                f"**命中词：** {', '.join(paper.matched_terms or []) or '无'}",
                "",
            ]
        )
    lines.extend(["---", "", f"本次备份候选：{len(backups)} 篇。", ""])
    return "\n".join(lines)


def render_html(selected: list[Paper], backups: list[Paper], run_day: str) -> str:
    cards = []
    for index, paper in enumerate(selected, 1):
        digest = paper.digest or {}
        tags = "".join(f"<span>{html.escape(term)}</span>" for term in (paper.matched_terms or [])[:6])
        cards.append(
            f"""
            <article class="paper grade-{paper.grade.lower()}">
              <div class="paper-head"><b>{index:02d}</b><em>{paper.grade} · {paper.score}</em></div>
              <h2><a href="{html.escape(paper.url)}">{html.escape(paper.title)}</a></h2>
              <p class="meta">作者：{html.escape(author_line(paper))}<br>单位：{html.escape(affiliation_line(paper))}<br>国家：{html.escape(country_line(paper))}<br>{html.escape(paper.source)} · {html.escape(paper.published)}</p>
              <p class="takeaway">{html.escape(digest.get('takeaway', ''))}</p>
              <dl><dt>方法</dt><dd>{html.escape(digest.get('methods', ''))}</dd><dt>相关性</dt><dd>{html.escape(digest.get('relevance', ''))}</dd><dt>局限/启示</dt><dd>{html.escape(digest.get('limitations', ''))}</dd><dt>摘要摘录</dt><dd>{html.escape(paper.abstract[:700])}{'……' if len(paper.abstract) > 700 else ''}</dd></dl>
              <div class="tags">{tags}</div>
            </article>"""
        )
    content = "\n".join(cards) if cards else '<section class="empty">今日无高优先级OPM论文，可查阅备份浏览次要文献</section>'
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OPM 文献每日速读</title><style>
:root{{--ink:#14201f;--muted:#65716f;--paper:#f7f8f4;--line:#d8ded8;--s:#c63d2f;--a:#187a70}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif}}
header{{padding:44px max(22px,calc((100% - 980px)/2));background:#122a28;color:white;border-bottom:5px solid #e0b84f}}
header h1{{margin:0;font-size:clamp(30px,5vw,54px);letter-spacing:0}} header p{{max-width:760px;margin:12px 0 0;color:#d8e6e3}}
main{{max-width:980px;margin:0 auto;padding:28px 22px 60px}} .notice{{padding:14px 0;border-bottom:1px solid var(--line);color:var(--muted)}}
.paper{{padding:26px 0;border-bottom:1px solid var(--line)}} .paper-head{{display:flex;justify-content:space-between;align-items:center}}
.paper-head b{{font-size:22px}} .paper-head em{{font-style:normal;font-weight:700}} .grade-s .paper-head em{{color:var(--s)}} .grade-a .paper-head em{{color:var(--a)}}
h2{{font-size:23px;line-height:1.35;letter-spacing:0;margin:10px 0}} a{{color:inherit;text-decoration-thickness:1px;text-underline-offset:4px}}
.meta{{color:var(--muted);font-size:14px}} .takeaway{{font-size:18px;font-weight:650;border-left:4px solid #e0b84f;padding-left:14px}}
dl{{display:grid;grid-template-columns:60px 1fr;gap:6px 12px}} dt{{font-weight:700}} dd{{margin:0}} .tags span{{display:inline-block;margin:5px 7px 0 0;padding:2px 7px;border:1px solid var(--line);font-size:12px}}
.empty{{padding:60px 0;font-size:20px;text-align:center}} footer{{color:var(--muted);font-size:13px;margin-top:30px}} @media(max-width:560px){{header{{padding-top:30px}}h2{{font-size:19px}}}}
</style></head><body><header><h1>OPM 文献每日速读</h1><p>聚焦 OPM-MCG/MEG、心磁与脑磁、自旋磁传感、脑网络、伪迹去除、多模态融合及临床转化。</p></header>
<main><div class="notice">🕒今日速读，最多7篇，预估阅读8-10分钟；更多候选论文见仓库 backup_papers.json<br>更新日期：{html.escape(run_day)}</div>
{content}<footer>本次备份候选：{len(backups)} 篇 · 自动情报仅用于文献初筛，请以论文原文为准。</footer></main></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Daily OPM paper digest")
    parser.add_argument("--dry-run", action="store_true", help="Skip OpenAI API calls")
    parser.add_argument("--fixture", type=Path, help="Use local fixture JSON instead of network sources")
    args = parser.parse_args()
    config = load_config()
    OUTPUT_DIR.mkdir(exist_ok=True)
    DOCS_DIR.mkdir(exist_ok=True)
    run_day = date.today().isoformat()
    start_date = date.today() - timedelta(days=int(config["project"]["lookback_days"]))

    if args.fixture:
        raw = json.loads(args.fixture.read_text(encoding="utf-8"))
        papers = [Paper(**item) for item in raw]
    else:
        papers = []
        errors = []
        if config["sources"].get("pubmed"):
            try:
                papers.extend(fetch_pubmed(config, start_date))
            except Exception as exc:
                errors.append(f"PubMed: {exc}")
        if config["sources"].get("arxiv"):
            try:
                papers.extend(fetch_arxiv(config, start_date))
            except Exception as exc:
                errors.append(f"arXiv: {exc}")
        if errors and not papers:
            raise RuntimeError("All literature sources failed: " + " | ".join(errors))
        for error in errors:
            print(f"WARNING {error}", file=sys.stderr)

    papers = deduplicate(papers)
    seen_data = read_json(SEEN_PATH, {"updated_at": "", "items": []})
    seen_items = set(seen_data.get("items", []))
    unseen = [paper for paper in papers if paper.stable_key not in seen_items]
    for paper in unseen:
        score_paper(paper, config)

    ranked = sorted(unseen, key=lambda item: (item.grade != "S", -item.score, item.title.lower()))
    eligible = [paper for paper in ranked if paper.grade in {"S", "A"}]
    selected = eligible[: int(config["project"]["max_report_papers"])]
    selected_keys = {paper.stable_key for paper in selected}
    backups = [paper for paper in ranked if paper.stable_key not in selected_keys]

    # Only selected S/A papers reach the LLM. B papers are scored and stored only.
    generate_digests(selected, config, args.dry_run)

    REPORT_PATH.write_text(render_markdown(selected, backups, run_day), encoding="utf-8")
    HTML_PATH.write_text(render_html(selected, backups, run_day), encoding="utf-8")
    BACKUP_PATH.write_text(
        json.dumps([asdict(paper) for paper in backups], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    seen_items.update(paper.stable_key for paper in unseen)
    SEEN_PATH.write_text(
        json.dumps(
            {"updated_at": datetime.now(timezone.utc).isoformat(), "items": sorted(seen_items)},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "fetched": len(papers),
                "unseen": len(unseen),
                "selected": len(selected),
                "backup": len(backups),
                "dry_run": args.dry_run,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
