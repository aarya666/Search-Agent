# multi_engine_search.py
"""
Multi-engine async job search module.

Exports:
 - run_job_search_async(...) -> dict with {"queries": [...], "results": [{query, hits: [...]}, ...]}
 - search_with_filters_only_async(...)
 - call_llm_generate_queries(...)  # simple passthrough / heuristic when no LLM configured

Make sure environment variables are set for whichever engines you want to use:
 - GOOGLE_API_KEY, GOOGLE_CX (Google Custom Search JSON API)
 - ADZUNA_APP_ID, ADZUNA_APP_KEY
 - JOOBLE_API_KEY
 - FINDWORK_API_KEY
"""

import os
import re
import json
import asyncio
import aiohttp
import hashlib
import tempfile
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import functools
import inspect
import random

load_dotenv()

# === Configuration / credentials from env ===
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CX = os.getenv("GOOGLE_CX")

ADZUNA_APP_ID = os.getenv("ADZUNA_APP_ID")
ADZUNA_APP_KEY = os.getenv("ADZUNA_APP_KEY")

JOOBLE_API_KEY = os.getenv("JOOBLE_API_KEY")  # expected to be the API key string
FINDWORK_API_KEY = os.getenv("FINDWORK_API_KEY")

# === Concurrency / timeouts ===
GLOBAL_SEMAPHORE = asyncio.Semaphore(8)  # global concurrency across all engines
ENGINE_TIMEOUT = 12  # seconds per request

# === Utilities ===
def dedupe_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for h in hits:
        key = (h.get("link") or h.get("url") or h.get("displayLink") or h.get("title", "")).strip()
        k = hashlib.sha1(key.encode("utf-8")).hexdigest()
        if k in seen:
            continue
        seen.add(k)
        out.append(h)
    return out

def build_india_city_list(extra_str: str = "") -> List[str]:
    default = ["Mumbai","Pune","Delhi NCR","Delhi","Bengaluru","Bangalore","Hyderabad","Chennai","Gurugram","Noida"]
    extra = [c.strip() for c in (extra_str or "").split(",") if c.strip()]
    out = []
    for c in default + extra:
        if c and c not in out:
            out.append(c)
    return out

# === Heuristics / sanitization / query generation / job filters ===
_ROLE_KEYWORDS = [
    "software engineer", "frontend", "frontend engineer", "backend", "backend engineer",
    "full stack", "fullstack", "full stack engineer", "data scientist", "machine learning",
    "machine learning engineer", "ml engineer", "devops", "qa", "test engineer", "android",
    "ios", "mobile developer", "product manager", "pm", "sde", "sde-i", "sde-ii",
    "intern", "internship", "research", "analyst"
]

_JOB_SIGNALS = [
    "apply", "apply now", "hiring", "vacancy", "vacancies", "job", "jobs", "position",
    "career", "careers", "openings", "opening", "recruit", "recruitment", "apply-online"
]

_EMAIL_RE = re.compile(r"\b\S+@\S+\.\S+\b")
_PHONE_RE = re.compile(r"(?:\+?\d[\d\-\s]{6,}\d)")

def _sanitize_resume_text(resume_text: str) -> str:
    if not resume_text:
        return ""
    text = resume_text
    # use correct internal regex names (_EMAIL_RE, _PHONE_RE)
    text = _EMAIL_RE.sub(" ", text)
    text = _PHONE_RE.sub(" ", text)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    clean = []
    for l in lines:
        # drop lines that look like simple names (1-2 tokens alphabetic) to avoid name leaking
        if len(l.split()) <= 2 and re.match(r"^[A-Za-z\.\-']+$", l):
            continue
        # skip extremely short single-token lines
        if len(l) < 8 and len(l.split()) == 1:
            continue
        clean.append(l)
    return "\n".join(clean)

def generate_queries_from_resume_text(resume_text: str, max_queries: int = 4, cities_hint: Optional[List[str]] = None) -> List[str]:
    """
    Create job-oriented queries from resume text.
    Returns a list of query strings (biased toward job-posting pages).
    """
    cities = cities_hint or []
    sanitized = _sanitize_resume_text(resume_text or "")
    lines = [l.lower() for l in sanitized.splitlines() if l.strip()]

    role_seeds = []
    skills_seed = None
    seniority_seed = None

    for l in lines:
        if "skills" in l or "technical skills" in l or "technologies" in l:
            skills_seed = re.sub(r"skills?:*", "", l)
            continue
        m = re.search(r"(\d{1,2}\+?)\s*(years|yrs|yr)", l)
        if m and not seniority_seed:
            seniority_seed = m.group(1) + " years"
        for rk in _ROLE_KEYWORDS:
            if rk in l and rk not in role_seeds:
                role_seeds.append(rk)

    if not role_seeds:
        for l in lines[:6]:
            for rk in _ROLE_KEYWORDS:
                if rk in l and rk not in role_seeds:
                    role_seeds.append(rk)
    if not role_seeds:
        first = (sanitized.splitlines()[0] if sanitized.splitlines() else "").strip()
        if first:
            t = re.sub(r"[^A-Za-z0-9\s\-\_]", "", first)[:60]
            if t:
                role_seeds.append(t.lower())
    if not role_seeds:
        role_seeds = ["software engineer"]

    # site hint: steer queries toward job posting pages (optional; remove to search fully)
    site_hint = " (site:linkedin.com/jobs OR site:indeed.com OR site:naukri.com OR site:adzuna.co.in OR site:jooble.org OR site:glassdoor.com)"
    def build_advanced_query(role, skills, seniority, cities):
        skill_part = ""
        if skills:
            skill_tokens = skills[:4]
            skill_part = " AND (" + " OR ".join(skill_tokens) + ")"

            city_part = ""
        if cities:
            city_part = " AND (" + " OR ".join(cities[:4]) + ")"

        seniority_part = f" AND {seniority}" if seniority else ""
        return f'"{role}"{skill_part}{seniority_part}{city_part} jobs'
    queries = []

    for i, role in enumerate(role_seeds[:max_queries]):
        toks = []
        if skills_seed:
            toks = re.findall(r"[a-zA-Z\+\#]{2,}", skills_seed)[:3]

            # always create q
        q = build_advanced_query(role, toks, seniority_seed, [])

        # rotate cities properly
        if cities:
            city = cities[i % len(cities)]
            q = f"{q} {city}"

        q = q + site_hint

        queries.append(q)

        if len(queries) >= max_queries:
            break

    if not queries:
        queries = ["software engineer jobs India" + site_hint]
    return queries[:max_queries]

def _is_likely_job_hit(hit: Dict[str, Any]) -> bool:
    s = " ".join([
        str(hit.get("title", "")).lower(),
        str(hit.get("snippet", "") or "").lower(),
        str(hit.get("link", "") or "").lower(),
        str(hit.get("displayLink", "") or "").lower(),
    ])
    if any(sig in s for sig in _JOB_SIGNALS):
        return True
    if re.search(r"/careers?/|/jobs?/|/openings?/|/vacanc", s):
        return True
    known_job_domains = ["indeed.com", "linkedin.com", "naukri.com", "adzuna", "jooble", "glassdoor.com"]
    if any(d in s for d in known_job_domains) and any(k in s for k in ["apply", "job", "opening", "career", "vacancy"]):
        return True
    return False

def filter_hits_to_jobs(hits: List[Dict[str, Any]], cities: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    cities = cities or []
    if not hits:
        return hits
    filtered = [h for h in hits if _is_likely_job_hit(h)]
    if filtered:
        return filtered
    city_tokens = [c.lower() for c in (cities or [])]
    filtered2 = []
    for h in hits:
        s = " ".join([str(h.get(k, "")) for k in ("title", "snippet", "displayLink", "link")]).lower()
        if "india" in s or ".in" in s or any(ct in s for ct in city_tokens):
            filtered2.append(h)
    return filtered2 or hits

# === Engine wrappers (async) ===
async def google_cse_search(session: aiohttp.ClientSession, query: str, num: int = 5) -> List[Dict[str, Any]]:
    if not GOOGLE_API_KEY or not GOOGLE_CX:
        raise RuntimeError("Google CSE credentials missing.")
    url = "https://www.googleapis.com/customsearch/v1"
    params = {"key": GOOGLE_API_KEY, "cx": GOOGLE_CX, "q": query, "num": num}
    async with GLOBAL_SEMAPHORE:
        async with session.get(url, params=params, timeout=ENGINE_TIMEOUT) as r:
            if r.status != 200:
                txt = await r.text()
                raise RuntimeError(f"Google CSE error {r.status}: {txt[:200]}")
            j = await r.json()
            items = []
            for it in j.get("items", []):
                items.append({
                    "title": it.get("title"),
                    "link": it.get("link"),
                    "snippet": it.get("snippet"),
                    "displayLink": it.get("displayLink"),
                })
            return items

async def adzuna_search(session: aiohttp.ClientSession, query: str, num: int = 5, country: str = "in") -> List[Dict[str, Any]]:
    if not ADZUNA_APP_ID or not ADZUNA_APP_KEY:
        raise RuntimeError("Adzuna credentials missing.")
    url = f"https://api.adzuna.com/v1/api/jobs/{country}/search/1"
    params = {"app_id": ADZUNA_APP_ID, "app_key": ADZUNA_APP_KEY, "results_per_page": num, "what": query}
    async with GLOBAL_SEMAPHORE:
        async with session.get(url, params=params, timeout=ENGINE_TIMEOUT) as r:
            if r.status != 200:
                txt = await r.text()
                raise RuntimeError(f"Adzuna error {r.status}: {txt[:200]}")
            j = await r.json()
            items = []
            for it in j.get("results", []):
                items.append({
                    "title": it.get("title"),
                    "link": it.get("redirect_url") or it.get("location", ""),
                    "snippet": it.get("description", ""),
                    "displayLink": (it.get("company", {}) or {}).get("display_name", "")
                })
            return items

async def jooble_search(session: aiohttp.ClientSession, query: str, num: int = 5, location: str = "India") -> List[Dict[str, Any]]:
    if not JOOBLE_API_KEY:
        raise RuntimeError("Jooble key missing.")
    url = f"https://jooble.org/api/{JOOBLE_API_KEY}"
    body = {"keywords": query, "location": location, "page": 1}
    async with GLOBAL_SEMAPHORE:
        async with session.post(url, json=body, timeout=ENGINE_TIMEOUT) as r:
            if r.status != 200:
                txt = await r.text()
                raise RuntimeError(f"Jooble error {r.status}: {txt[:200]}")
            j = await r.json()
            items = []
            for it in j.get("jobs", [])[:num]:
                items.append({
                    "title": it.get("title"),
                    "link": it.get("link"),
                    "snippet": it.get("description", ""),
                    "displayLink": it.get("company", "")
                })
            return items

async def findwork_search(session: aiohttp.ClientSession, query: str, num: int = 5) -> List[Dict[str, Any]]:
    if not FINDWORK_API_KEY:
        raise RuntimeError("Findwork key missing.")
    url = "https://findwork.dev/api/v1/jobs/"
    params = {"search": query, "limit": num}
    headers = {"Authorization": f"Token {FINDWORK_API_KEY}"}
    async with GLOBAL_SEMAPHORE:
        async with session.get(url, params=params, headers=headers, timeout=ENGINE_TIMEOUT) as r:
            if r.status != 200:
                txt = await r.text()
                raise RuntimeError(f"Findwork error {r.status}: {txt[:200]}")
            j = await r.json()
            items = []
            for it in j.get("results", []):
                items.append({
                    "title": it.get("title"),
                    "link": it.get("url"),
                    "snippet": it.get("description", ""),
                    "displayLink": it.get("company", "")
                })
            return items

# safe engine caller returns list or [{"error":...}]
async def call_engine_safe(engine_fn, session, query, num=5, **kwargs):
    try:
        return await engine_fn(session, query, num=num, **kwargs)
    except Exception as e:
        return [{"error": str(e)}]

# === Orchestration: run_job_search_async ===
async def run_job_search_async(
    filename: str,
    max_results: int = 10,
    remote_modes: str = "",
    cities: str = "",
    roles: str = "",
    job_type: str = "",
    mode: str = "india"   # ✅ NEW
) -> Dict[str, Any]:
    """
    Main async entrypoint used by the FastAPI server.

    - filename: uploaded file basename in the system tmpdir
    - returns: {"queries": [...], "results": [{ "query": q, "hits": [...] }, ...]}
    """
    tmpdir = tempfile.gettempdir()
    path = os.path.join(tmpdir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError("Uploaded file not found on server.")

    # Extract resume text (try to import fitz here to keep module optional)
    resume_text = ""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(path)
        resume_text = "\n".join([p.get_text("text") for p in doc])
        doc.close()
    except Exception:
        resume_text = ""

    if mode == "global":
        all_cities = []  # 🌍 no location bias
    else:
        all_cities = build_india_city_list(cities)
    random.shuffle(all_cities)
    # generate base queries
    base_queries = generate_queries_from_resume_text(resume_text, max_queries=4, cities_hint=all_cities)

    # augment queries with roles / job_type / remote preferences
    augmented_queries = []
    remote_clause = " ".join([m.strip() for m in (remote_modes or "").split(",") if m.strip()])
    for q in base_queries:
        qq = q
        if roles:
            qq = f"{qq} {roles}"
        if job_type:
            qq = f"{qq} {job_type}"
        if remote_clause:
            qq = f"{qq} {remote_clause}"
        augmented_queries.append(qq)
    
    def balance_by_city(hits: List[Dict[str, Any]], cities: List[str]) -> List[Dict[str, Any]]:
        if not hits or not cities:
            return hits

        buckets = {c.lower(): [] for c in cities}

        for h in hits:
            text = (
                str(h.get("title", "")) +
                str(h.get("snippet", "")) +
                str(h.get("link", ""))
            ).lower()

            for c in cities:
                if c.lower() in text:
                    buckets[c.lower()].append(h)
                    break

        balanced = []
        max_per_city = 2

        for c in cities:
            balanced.extend(buckets[c.lower()][:max_per_city])

        if not balanced:
            return hits

        return balanced

    aggregated = []
    all_hits = []  # ✅ collect everything globally

    # perform engine calls per query
    async with aiohttp.ClientSession() as session:
        for base_q in augmented_queries:
            tasks = []

            # prefer Google CSE if available
            if GOOGLE_API_KEY and GOOGLE_CX:
                tasks.append(call_engine_safe(google_cse_search, session, base_q, num=min(5, max_results)))
            if ADZUNA_APP_ID and ADZUNA_APP_KEY:
                tasks.append(call_engine_safe(adzuna_search, session, base_q, num=min(5, max_results)))
            if JOOBLE_API_KEY:
                tasks.append(call_engine_safe(jooble_search, session, base_q, num=min(5, max_results)))
            if FINDWORK_API_KEY:
                tasks.append(call_engine_safe(findwork_search, session, base_q, num=min(5, max_results)))

            if not tasks:
                aggregated.append({
                    "query": base_q,
                    "hits": [{"error": "No search engines configured (set GOOGLE_*, ADZUNA_*, JOOBLE_*, or FINDWORK_*)."}]
                })
                continue

            results_per_engine = await asyncio.gather(*tasks, return_exceptions=False)

            hits: List[Dict[str, Any]] = []
            for eng_res in results_per_engine:
                if isinstance(eng_res, list):
                    for it in eng_res:
                        if isinstance(it, dict):
                            hits.append(it)

            # dedupe + filter
            hits = dedupe_hits(hits)
            hits = filter_hits_to_jobs(hits, cities=all_cities)

            all_hits.extend(hits)  # ✅ collect globally

            aggregated.append({
                "query": base_q,
                "hits": hits  # temporary (will rebalance later)
            })

    # ✅ GLOBAL BALANCING
    all_hits = dedupe_hits(all_hits)
    if mode == "india":
        all_hits = balance_by_city(all_hits, all_cities)

    # limit total results
    all_hits = all_hits[: len(aggregated) * max_results]

    # ✅ redistribute back to queries
    idx = 0
    for block in aggregated:
        block["hits"] = all_hits[idx: idx + max_results]
        idx += max_results

    return {
        "queries": augmented_queries,
        "results": aggregated
    }

# === Filter-only search (no resume) ===
async def search_with_filters_only_async(max_results: int = 10, remote_modes: str = "", cities: str = "", roles: str = "", job_type: str = "") -> Dict[str, Any]:
    all_cities = build_india_city_list(cities)
    role_list = [r.strip() for r in (roles or "").split(",") if r.strip()] or ["Software Engineer"]
    queries = []
    for r in role_list[:4]:
        q = f"{r} jobs India"
        if job_type:
            q += f" {job_type}"
        if remote_modes:
            q += f" {remote_modes}"
        if all_cities:
            q += f" ({' OR '.join(all_cities)})"
        queries.append(q)

    aggregated = []
    async with aiohttp.ClientSession() as session:
        for q in queries:
            tasks = []
            if GOOGLE_API_KEY and GOOGLE_CX:
                tasks.append(call_engine_safe(google_cse_search, session, q, num=min(5, max_results)))
            if ADZUNA_APP_ID and ADZUNA_APP_KEY:
                tasks.append(call_engine_safe(adzuna_search, session, q, num=min(5, max_results)))
            if JOOBLE_API_KEY:
                tasks.append(call_engine_safe(jooble_search, session, q, num=min(5, max_results)))
            if FINDWORK_API_KEY:
                tasks.append(call_engine_safe(findwork_search, session, q, num=min(5, max_results)))

            if not tasks:
                aggregated.append({"query": q, "hits": [{"error": "No search engines configured."}]})
                continue

            results_per_engine = await asyncio.gather(*tasks, return_exceptions=False)
            hits: List[Dict[str, Any]] = []
            for eng_res in results_per_engine:
                if isinstance(eng_res, list):
                    for it in eng_res:
                        if isinstance(it, dict):
                            hits.append(it)
            hits = dedupe_hits(hits)
            hits = filter_hits_to_jobs(hits, cities=all_cities)
            hits = hits[:max_results]
            aggregated.append({"query": q, "hits": hits})

    return {"queries": queries, "results": aggregated}

# === LLM helper placeholder (for main.py summarize fallback) ===
def call_llm_generate_queries(resume_text: str, max_queries: int = 4, filters: Dict[str, Any] = None) -> List[str]:
    """
    Placeholder: if you have a GROQ or other LLM integration, implement here.
    By default this returns the heuristic queries used above.
    """
    cities_hint = None
    if filters and filters.get("cities"):
        cities_hint = [c.strip() for c in str(filters.get("cities")).split(",") if c.strip()]
    return generate_queries_from_resume_text(resume_text, max_queries=max_queries, cities_hint=cities_hint)

# Export the public API names expected by main.py
__all__ = [
    "run_job_search_async",
    "search_with_filters_only_async",
    "call_llm_generate_queries",
    "generate_queries_from_resume_text",
]
