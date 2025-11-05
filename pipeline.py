import os, json, re, time, glob, requests
from typing import List, Optional, Dict
from datetime import datetime, timezone
from urllib.parse import urlparse
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
from pydantic import BaseModel, Field
from tavily import TavilyClient
from scrapegraph_py import Client

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
SCRAPEGRAPH_API_KEY = os.getenv("scrap_key", "")
AGENTOPS_API_KEY = os.getenv("agentops_Key", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("Route_key_2", "")


basic_llm = LLM(
    model= "openrouter/google/gemini-2.5-flash",
    temperature=0.1,
    api_key=os.getenv("Route_key_2"),
    base_url="https://openrouter.ai/api/v1",
)

INCOMING_DIR = "./incoming"
OUTPUT_DIR = "./daily-job-recommendations"
os.makedirs(INCOMING_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

if not TAVILY_API_KEY:
    raise RuntimeError("TAVILY_API_KEY missing.")
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
if not SCRAPEGRAPH_API_KEY:
    raise RuntimeError("SCRAPEGRAPH_API_KEY missing.")
scrap_client = Client(api_key=SCRAPEGRAPH_API_KEY)

ALLOWED_WEBSITES = ["www.linkedin.com/jobs","www.bayt.com","www.akhtaboot.com","www.indeed.com","www.glassdoor.com"]
GCC_KEYWORDS = ["jordan","amman","saudi","riyadh","ksa","uae","dubai","abu dhabi","qatar","doha","oman","bahrain","kuwait"]
about_program = "Entry-level job recommender. Builds searches strictly from student's INTERESTS and LOCATIONS (skills optional). Targets internships, junior, entry level, graduate and trainee roles."
program_context = StringKnowledgeSource(content=about_program, name="Program Context")

class SuggestedSearchQueries(BaseModel):
    queries: List[str] = Field(..., min_items=1, max_items=50)

class JobURL(BaseModel):
    url: str
    title: str
    snippet: str
    source_domain: str
    score: float

class AllJobURLs(BaseModel):
    urls: List[JobURL] = Field(..., min_items=1)

class SingleJobPosting(BaseModel):
    page_url: str
    title: Optional[str] = None
    company: Optional[str] = None
    location: Optional[str] = None
    is_remote: Optional[bool] = None
    seniority: Optional[str] = None
    posted_at: Optional[str] = None
    job_type: Optional[str] = None
    salary: Optional[str] = None
    description: Optional[str] = None
    requirements: List[str] = []
    benefits: List[str] = []
    apply_url: Optional[str] = None
    source_domain: Optional[str] = None
    language: Optional[str] = None
    scrape_status: Optional[str] = None
    student_id: Optional[str] = None

class AllExtractedJobs(BaseModel):
    jobs: List[SingleJobPosting] = Field(..., min_items=1)

def _parse_date_try(s):
    if not s: return None
    fmts = ("%Y-%m-%d","%Y/%m/%d","%d-%m-%Y","%Y-%m-%dT%H:%M:%S%z","%Y-%m-%dT%H:%M:%S","%Y-%m-%dZ")
    for f in fmts:
        try:
            dt = datetime.strptime(s, f)
            if "%z" not in f and "T" not in f:
                return dt.replace(tzinfo=timezone.utc)
            return dt
        except:
            pass
    try:
        dt = datetime.fromisoformat(s.replace("Z",""))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except:
        return None

def _is_gcc_or_jordan(loc):
    if not loc: return False
    l = loc.lower()
    return any(k in l for k in GCC_KEYWORDS)

def _is_entry_level(s):
    if not s: return False
    s = s.lower()
    return any(t in s for t in ["intern","junior","entry","fresh","graduate","trainee"])

def _skills_match(skills: List[str], reqs: List[str], desc: str = "") -> tuple:
    if not skills: return 0.0, []
    S = {s.lower().strip() for s in skills}
    text = (" ".join(reqs) + " " + (desc or "")).lower()
    matched = [s for s in S if re.search(r"\b"+re.escape(s)+r"\b", text)]
    score = min(1.0, len(matched)/max(1,len(S)))
    return score, matched

def _is_valid_url(url: str, allowed_domains: list) -> bool:
    if not url or not url.startswith("http"): return False
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        path = parsed.path.lower()
        for allowed in allowed_domains:
            if allowed.lower() in domain or domain in allowed.lower():
                if "/jobs/" in path or "/job/" in path or "careers" in path:
                    if all(b not in path for b in ["/search","/company/","/companies","/browse","/alerts","/career-advice"]):
                        return True
        return False
    except:
        return False

def _is_job_posting_url(url: str) -> bool:
    if not url: return False
    url_lower = url.lower()
    if any(ind in url_lower for ind in ["/jobs/view/","/job/","/jobs-","/vacancy/","/career/","/apply/"]): return True
    if re.search(r"/jobs?/\d+", url_lower): return True
    if any(b in url_lower for b in ["/search","/company/","/companies","/browse","/alerts","/career-advice"]): return False
    return False

def split_csv(text: str):
    if not text: return []
    text = text.replace(" and ", ",")
    return [p.strip() for p in text.split(",") if p.strip()]

def rank_jobs_advanced(jobs: list, student_profile: dict, top_n: int = 10) -> Dict:
    sp = dict(student_profile or {})
    sid = sp.get("student_id", "UNKNOWN")
    langs = sp.get("languages", ["English"])
    remote_ok = sp.get("remote_ok", True)
    skills = sp.get("skills", [])
    interests = sp.get("interests", [])
    min_score = sp.get("min_match_score", 0.3)
    ranked = []
    for j in (jobs or []):
        reqs = j.get("requirements") or []
        desc = j.get("description") or ""
        title = (j.get("title") or "").lower()
        if not title.strip() or j.get("scrape_status") == "failed": continue
        if not j.get("apply_url") or not _is_valid_url(j.get("apply_url"), ALLOWED_WEBSITES): continue
        s, matched_skills = _skills_match(skills, reqs, desc)
        interest_match = 0.0
        matched_interests = []
        combined_text = (title + " " + " ".join(reqs) + " " + desc).lower()
        for it in interests:
            it = str(it).lower().strip()
            if it and re.search(r"\b"+re.escape(it)+r"\b", combined_text):
                interest_match += 0.34
                matched_interests.append(it)
        interest_match = min(1.0, interest_match)
        senior = 1.0 if _is_entry_level(j.get("seniority")) else 0.3
        loc_score = 0.0
        if j.get("is_remote") and remote_ok: loc_score = 1.0
        elif _is_gcc_or_jordan(j.get("location")): loc_score = 0.9
        days_old, fresh = 999, 0.3
        if j.get("posted_at"):
            dt = _parse_date_try(j.get("posted_at"))
            if dt:
                days_old = (datetime.now(timezone.utc)-dt.replace(tzinfo=timezone.utc)).days
                fresh = 1.0 if days_old <= 3 else 0.9 if days_old <= 7 else 0.8 if days_old <= 14 else 0.6 if days_old <= 30 else 0.3
        lang_val = j.get("language")
        lang = 1.0 if not lang_val or any(str(lang_val).lower().startswith(L.lower()) for L in langs) else 0.6
        source = (j.get("source_domain") or "").lower()
        priority_boost = 0.1 if "linkedin" in source else 0.05 if ("bayt" in source or "akhtaboot" in source) else 0.0
        score = (0.15*s + 0.34*interest_match + 0.18*senior + 0.14*loc_score + 0.12*fresh + 0.05*lang + 0.02*priority_boost)
        if score < min_score: continue
        reasons = []
        if matched_interests: reasons.append(f"Matches interests: {', '.join(matched_interests)}")
        if matched_skills: reasons.append(f"Matched skills: {', '.join(matched_skills[:5])}")
        if senior >= 0.9: reasons.append(f"Level fit: {j.get('seniority')}")
        if loc_score >= 0.9: reasons.append(f"Location fit: {j.get('location') or 'Remote'}")
        if days_old <= 7: reasons.append(f"Fresh posting: {days_old} day(s) ago")
        ranked.append({"job": j,"score": round(score, 4),"matched_skills": matched_skills[:5],"matched_interests": matched_interests,"days_old": days_old,"reasons": reasons})
    ranked.sort(key=lambda x: (-x["score"], x["days_old"]))
    return {"student_id": sid,"generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),"total_jobs_analyzed": len(jobs or []),"valid_jobs_found": len(ranked),"top_jobs": ranked[:top_n]}

_current_student_profile: dict = {}

@tool
def generate_search_queries(k: int = 12) -> Dict:
    """
    Build targeted job search queries strictly from the student's INTERESTS and preferred LOCATIONS, optionally appending SKILLS with low weight. Seniority terms are limited to early-career roles. Returns up to k unique queries preserving original order.

    Args:
        k (int): Maximum number of queries to return.

    Returns:
        Dict: {"queries": List[str]} where each query is "<interest or skill> <seniority> <location>" in lowercase.
    """
    profile = _current_student_profile or {}
    interests = [s for s in (profile.get("interests") or []) if isinstance(s, str) and s.strip()]
    locations = [l for l in (profile.get("preferred_locations") or []) if isinstance(l, str) and l.strip()]
    skills = [s for s in (profile.get("skills") or []) if isinstance(s, str) and s.strip()]
    if not interests or not locations: return {"queries": []}
    seniorities = ["intern", "internship", "junior", "entry level", "graduate", "trainee"]
    queries: List[str] = []
    loc_idx = 0
    for it in interests:
        it_clean = it.strip()
        for s_word in seniorities:
            loc = locations[loc_idx % len(locations)]
            loc_idx += 1
            q = f"{it_clean} {s_word} {loc}".lower().strip()
            queries.append(q)
    for sk in skills[:3]:
        sk_clean = sk.strip()
        for s_word in seniorities[:2]:
            loc = locations[loc_idx % len(locations)]
            loc_idx += 1
            q = f"{sk_clean} {s_word} {loc}".lower().strip()
            queries.append(q)
    seen, deduped = set(), []
    for q in queries:
        if q not in seen:
            seen.add(q)
            deduped.append(q)
    return {"queries": deduped[:k]}

@tool
def tavily_search_jobs(queries: list, max_results_per_query: int = 5) -> Dict:
    """
    Query Tavily for real job posting URLs on trusted boards and return a deduplicated, relevance-sorted list. Strictly filters to valid job pages and allowed domains to minimize noise.

    Args:
        queries (list): Search queries produced by generate_search_queries.
        max_results_per_query (int): Upper bound of results per query.

    Returns:
        Dict: {"urls": [ { "url": str, "title": str, "snippet": str, "source_domain": str, "score": float }, ... ]} with up to 40 high-confidence URLs across all queries.
    """
    all_urls = []
    for query in (queries or []):
        if not query: continue
        try:
            search_query = (f"{query} internship OR junior OR entry-level site:linkedin.com OR site:bayt.com OR site:akhtaboot.com OR site:indeed.com OR site:glassdoor.com")
            response = tavily_client.search(query=search_query, search_depth="advanced", max_results=max_results_per_query, include_domains=["linkedin.com","bayt.com","akhtaboot.com","indeed.com","glassdoor.com"], exclude_domains=[])
            for result in response.get('results', []):
                url = result.get('url', '') or ''
                if not _is_valid_url(url, ALLOWED_WEBSITES): continue
                if not _is_job_posting_url(url): continue
                job_url = {"url": url,"title": result.get('title', '') or '','snippet': (result.get('content', '') or '')[:200],"source_domain": urlparse(url).netloc,"score": float(result.get('score', 0.0) or 0.0)}
                all_urls.append(job_url)
            time.sleep(0.4)
        except Exception as e:
            print(f"[tavily] Error searching '{query}': {e}")
            continue
    seen, unique_urls = set(), []
    for item in all_urls:
        if item['url'] not in seen:
            seen.add(item['url'])
            unique_urls.append(item)
    unique_urls.sort(key=lambda x: x['score'], reverse=True)
    return {"urls": unique_urls[:40]}

@tool
def scrape_job_page(url: str) -> Dict:
    """
    Scrape a single job posting page and extract strictly on-page values into a SingleJobPosting-shaped dict. Detect language, infer entry-level seniority buckets, and ensure an actionable apply_url. Does not fabricate fields.

    Args:
        url (str): Target job posting URL on linkedin/bayt/akhtaboot/indeed/glassdoor.

    Returns:
        Dict: Fields aligned to SingleJobPosting with "scrape_status" set to "success" on valid extraction, or a failure code otherwise.
    """
    if not _is_valid_url(url, ALLOWED_WEBSITES):
        return {"page_url": url, "scrape_status": "invalid_domain"}
    if not _is_job_posting_url(url):
        return {"page_url": url, "scrape_status": "not_job_posting"}
    try:
        schema_json = SingleJobPosting.schema_json()
        prompt = f"""Extract job details from this page into JSON that matches this schema exactly:
{schema_json}
RULES:
- Extract EXACT on-page values only.
- Seniority ∈ {{internship, junior, entry-level, mid, senior, trainee}}.
- is_remote=true ONLY if the word 'remote' explicitly exists.
- If an apply link exists, return it; otherwise fallback to the page URL.
- Detect language (English/Arabic).
- Set scrape_status='success' when core fields are present."""
        result = scrap_client.smartscraper(website_url=url, user_prompt=prompt)
        if isinstance(result, str): result = json.loads(result)
        if isinstance(result, dict):
            result["page_url"] = url
            result["source_domain"] = urlparse(url).netloc
            if not result.get("apply_url"): result["apply_url"] = url
            if not result.get("scrape_status"): result["scrape_status"] = "success"
            result["student_id"] = _current_student_profile.get("student_id")
            return result
        return {"page_url": url, "scrape_status": "failed"}
    except Exception as e:
        return {"page_url": url, "scrape_status": "error", "error": str(e)}

def _load_latest_payload() -> dict:
    files = sorted(glob.glob(os.path.join(INCOMING_DIR, "*.json")))
    if not files:
        raise FileNotFoundError(f"ما لقيت ملفات JSON داخل {INCOMING_DIR}")
    payload_path = files[-1]
    print("Loaded payload:", payload_path)
    with open(payload_path, "r", encoding="utf-8") as f:
        return json.load(f)

def _build_student_profile(payload: dict) -> dict:
    interests = split_csv(payload.get("interests", ""))
    skills = split_csv(payload.get("job_skills", ""))
    return {"student_id": f"{payload.get('id', '01')}","name": payload.get("name", "Student"),"email": payload.get("email", "student@example.com"),"skills": skills or [],"interests": interests or [],"preferred_locations": payload.get("locations", ["Amman","Jordan","Riyadh","Dubai","Doha","Remote"]),"remote_ok": True,"languages": ["English","Arabic"],"min_match_score": 0.30}

def _send_results_webhook(file_path: str) -> None:
    url ='https://syncrew.app.n8n.cloud/webhook/27c1d9bf-73ef-4e6e-8623-4354292bb7e2'
    if not url:
        print("⚠️ No RESULTS_WEBHOOK_URL set — skipping webhook send.")
        return
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        resp = requests.post(url, json=payload, timeout=30)
        print(f"📤 Webhook POST → {url} | status={resp.status_code}")
        if resp.status_code >= 400:
            print(f"❌ Webhook error body: {resp.text[:500]}")
    except Exception as e:
        print(f"❌ فشل إرسال الويبهوك: {e}")

def run_pipeline() -> bool:
    global _current_student_profile
    payload = _load_latest_payload()
    _current_student_profile = _build_student_profile(payload)
    if not _current_student_profile.get("interests") or not _current_student_profile.get("preferred_locations"):
        raise ValueError("interests و preferred_locations مطلوبان لتوليد الاستعلامات بدون أمثلة جاهزة.")
    print("STUDENT_PROFILE =", _current_student_profile)
    query_generator_agent = Agent(role='Job Search Query Generator', goal='Generate search queries strictly from interests and locations (skills optional).', backstory='Builds only combinations of {interest} × {seniority} × {location}.', llm=basic_llm, tools=[generate_search_queries], verbose=True)
    tavily_search_agent = Agent(role='Tavily Job Search Specialist', goal='Find real job posting URLs from major boards.', backstory='Precise filtering for early-career roles.', llm=basic_llm, tools=[tavily_search_jobs], verbose=True)
    scraper_agent = Agent(role='Job Data Scraper', goal='Extract accurate job details present on the page.', backstory='Returns only what is written.', llm=basic_llm, tools=[scrape_job_page], verbose=True)
    query_task = Task(description="Generate 12 queries from INTERESTS×LOCATIONS (skills optional). Call generate_search_queries(k=12).", expected_output="JSON with 'queries'.", output_json=SuggestedSearchQueries, output_file=os.path.join(OUTPUT_DIR, "01_search_queries.json"), agent=query_generator_agent, verbose=True)
    search_task = Task(description="Use Tavily to fetch real job postings for the queries. Save verified URLs.", expected_output="JSON with 'urls'.", output_json=AllJobURLs, output_file=os.path.join(OUTPUT_DIR, "02_tavily_job_urls.json"), agent=tavily_search_agent, verbose=True)
    scrape_task = Task(description="Scrape each URL; return exact on-page values and attach student_id.", expected_output="JSON with 'jobs'.", output_json=AllExtractedJobs, output_file=os.path.join(OUTPUT_DIR, "03_scraped_jobs.json"), agent=scraper_agent, verbose=True)
    crew = Crew(agents=[query_generator_agent, tavily_search_agent, scraper_agent], tasks=[query_task, search_task, scrape_task], process=Process.sequential, knowledge_sources=[program_context], verbose=True)
    crew.kickoff(inputs={})
    base_path = os.path.join(OUTPUT_DIR, "03_scraped_jobs.json")
    if not os.path.exists(base_path):
        raise FileNotFoundError("Scraped jobs file not found at " + base_path)
    with open(base_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    student_id = _current_student_profile.get("student_id", "01")
    for job in data.get("jobs", []):
        job["student_id"] = student_id
    new_path = os.path.join(OUTPUT_DIR, "03_scraped_jobs_with_id.json")
    with open(new_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ تم إنشاء الملف: {new_path}")
    _send_results_webhook(new_path)
    return True

if __name__ == "__main__":
    ok = run_pipeline()
    print("Pipeline OK:", ok)
