import ast
import copy
import json
import os
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, session
from langchain_openai import ChatOpenAI
from werkzeug.utils import secure_filename

load_dotenv()
BASE = Path(__file__).parent
REFERENCE_PATH = Path(os.getenv("REFERENCE_PAYLOAD") or BASE / "reference_payload.json")
MAX_ATTEMPTS = 3
CMS_BASE = "https://cms.techcaddmohali.com"
HEADERS_FILE = Path(os.getenv("CMS_HEADERS_FILE") or BASE / "cms_headers.txt")




def cms_headers(referer, *, json_request=True, full_browser=False):
    """Build CMS request headers from one shared configuration."""
    headers = {
            'accept': '*/*',
            'accept-language': 'en-US,en;q=0.9',
            'origin': 'https://cms.techcaddmohali.com',
            'priority': 'u=1, i',
            'referer': 'https://cms.techcaddmohali.com/',
            'sec-ch-ua': '"Chromium";v="158", "Not.A/Brand";v="24", "Google Chrome";v="158"',
         
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/158.0.0.0 Safari/537.36',
            'Cookie': '_ga=GA1.1.1993469050.1795078505; _clck=1pfe81f%5E2%5Egbg%5E0%5E2514; _ga_9Z3DEFNXKL=GS2.1.s1795595839$o12$g1$t1795595901$j60$l0$h0; techcadd_session=s%3Af17c638e-c543-476b-a034-31d7196f0171.7aoZMLQlidp2A1g%2F3I5Z4V16NrUzs6QJE%2BDZoL5SdeE',
            }
    
    return headers


app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or os.urandom(32)
# --------------------------------------------------------------------------
# LLM
# --------------------------------------------------------------------------

def make_llm(json_mode: bool):
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    return ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("OPENAI_MODEL"),
        timeout=180,
        max_retries=2,
        model_kwargs=kwargs,
    )

def strip_fences(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json|html)?\s*", "", text)
    return re.sub(r"\s*```$", "", text).strip()
# --------------------------------------------------------------------------
# Validation helpers (all tolerant of missing / wrong-typed data)
# --------------------------------------------------------------------------

def rng(value, lo, hi, label, issues):
    n = len(value) if isinstance(value, str) else -1
    if not lo <= n <= hi:
        issues.append(f"{label}: {n} characters, must be {lo}-{hi}")

def check_items(d, key, n, fields, issues, list_fields=None):
    items = d.get(key)
    if not isinstance(items, list) or len(items) != n:
        issues.append(f"{key}: must contain exactly {n} objects (got "
                      f"{len(items) if isinstance(items, list) else 'invalid'})")
        return
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            issues.append(f"{key}[{i}] is not an object")
            continue
        for f, (lo, hi) in fields.items():
            rng(item.get(f), lo, hi, f"{key}[{i}].{f}", issues)
        for f, size in (list_fields or {}).items():
            v = item.get(f)
            if not isinstance(v, list) or len(v) != size:
                issues.append(f"{key}[{i}].{f}: must contain exactly {size} items")
META_LIMITS = {
    "tagline": (100, 145), "eyebrow": (15, 35), "badge": (10, 25),
    "intro": (175, 220), "ctaPrimaryText": (20, 35), "ctaSecondaryText": (20, 35),
    "syllabusIntro": (220, 270), "syllabusNote": (135, 175),
    "audienceIntro": (145, 190), "whyIntro": (175, 215),
    "comparisonIntro": (155, 195), "comparisonOthers": (20, 40),
    "comparisonNote": (120, 155), "demand": (180, 225),
    "shortDescription": (130, 155), "eligibility": (160, 200),
    "certification": (130, 170),
}
SEO_LIMITS = {
    "metaTitle": (45, 60), "metaDescription": (140, 160), "ogTitle": (40, 60),
    "ogDescription": (110, 155), "twitterTitle": (35, 60), "twitterDescription": (100, 150),
}

def validate_meta(d):
    issues = []
    for k, (lo, hi) in META_LIMITS.items():
        rng(d.get(k), lo, hi, k, issues)
    if not isinstance(d.get("duration"), str) or not d["duration"].strip():
        issues.append("duration: required non-empty string")
    seo = d.get("seo") if isinstance(d.get("seo"), dict) else {}
    for k, (lo, hi) in SEO_LIMITS.items():
        rng(seo.get(k), lo, hi, f"seo.{k}", issues)
    kw = seo.get("keywords")
    if not isinstance(kw, list) or len(kw) != 13 or not all(isinstance(x, str) for x in kw):
        issues.append("seo.keywords: must be exactly 13 strings")
    return issues

def validate_arrays(d):
    issues = []
    check_items(d, "audience", 5, {"title": (20, 40), "body": (110, 140)}, issues)
    check_items(d, "benefits", 6, {"title": (25, 40), "body": (100, 140)}, issues)
    check_items(d, "careerRoles", 5, {"role": (20, 40), "body": (80, 120)}, issues)
    check_items(d, "projects", 6, {"title": (20, 45), "body": (220, 310)}, issues, {"tags": 4})
    check_items(d, "workflow", 5, {"title": (20, 35), "body": (85, 130)}, issues)
    check_items(d, "whyPoints", 6, {"title": (20, 40), "body": (75, 120)}, issues)
    check_items(d, "comparisonRows", 7,
                {"feature": (8, 25), "ours": (35, 55), "theirs": (35, 55)}, issues)
    check_items(d, "toolItems", 21, {"name": (1, 60), "category": (1, 60)}, issues)
    hl = d.get("highlights")
    if not isinstance(hl, list) or len(hl) != 6:
        issues.append("highlights: must be exactly 6 strings")
    else:
        for i, h in enumerate(hl):
            rng(h, 40, 80, f"highlights[{i}]", issues)
    return issues

def validate_syllabus(d):
    issues = []
    check_items(d, "syllabus", 10,
                {"title": (35, 65), "body": (140, 175), "project": (60, 100)},
                issues, {"topics": 4, "outcomes": 2, "tools": 4})
    return issues

def html_validator(lo, hi, h2_count):
    def _v(html):
        issues = []
        rng(html, lo, hi, "html", issues)
        if len(re.findall(r"<h2>", html)) != h2_count:
            issues.append(f"must contain exactly {h2_count} <h2> sections")
        if len(re.findall(r"<p>", html)) != h2_count:
            issues.append(f"must contain exactly {h2_count} <p> paragraphs (one per h2)")
        if re.search(r"</?(?!h2\b|p\b|strong\b)[a-zA-Z]+", html):
            issues.append("only h2, p and strong tags are allowed")
        if "**" in html or "```" in html or "\n#" in html:
            issues.append("no Markdown allowed")
        return issues
    return _v
# --------------------------------------------------------------------------
# Generic "ask -> validate -> retry with feedback" loop
# --------------------------------------------------------------------------

def run_with_retries(llm, prompt, parse, validate, length_hint=None):
    """Returns (result, remaining_issues)."""
    best, best_issues, feedback = None, ["no output"], ""
    for _ in range(MAX_ATTEMPTS):
        try:
            text = llm.invoke(prompt + feedback).content
            result = parse(text)
        except Exception as e:  # bad JSON, API error, etc.
            feedback = f"\n\nYour previous attempt failed: {e}. Return the output in the exact requested format."
            best_issues = best_issues if best else [f"attempt failed: {e}"]
            continue
        issues = validate(result)
        if best is None or len(issues) < len(best_issues):
            best, best_issues = result, issues
        if not issues:
            return result, []
        feedback = ("\n\nYOUR PREVIOUS ANSWER HAD THESE PROBLEMS. Fix every one and "
                    "return the complete corrected output:\n- " + "\n- ".join(issues[:25]))
        if length_hint and isinstance(result, str):
            n = len(result)
            lo, hi = length_hint
            mid = (lo + hi) // 2
            if n < lo:
                feedback += f"\nThe text was {n} characters. Add roughly {mid - n} more characters of useful content."
            elif n > hi:
                feedback += f"\nThe text was {n} characters. Remove roughly {n - mid} characters."
    return best, best_issues
# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

def context(title, location, brand, extra):
    return f"""You are a senior SEO strategist, semantic/local SEO expert, AEO/GEO expert,
NLP content writer and CMS content specialist.
COURSE INPUT
Title: {title}
Location: {location}
Brand: {brand}  (always spell exactly "techcadd")
Additional instructions / keywords / technologies: {extra}
GLOBAL CONTENT RULES
- Natural, humanized, informative, conversion-focused writing; no exaggerated marketing language.
- No job guarantees, salaries, placement statistics, reviews, fake URLs or invented credentials.
- Use location-specific search intent naturally; avoid keyword stuffing and repetition.
- All character limits include spaces and punctuation. COUNT CAREFULLY and stay inside every range.
"""

def trim_example(obj):
    """Shrink reference data to a small shape example."""
    if isinstance(obj, list):
        return [trim_example(obj[0])] if obj else []
    if isinstance(obj, dict):
        return {k: trim_example(v) for k, v in obj.items()}
    return obj

def prompt_meta(ctx):
    return ctx + f"""
TASK: Return ONE JSON object with exactly these keys and character limits:
{json.dumps({k: f"{lo}-{hi} chars" for k, (lo, hi) in META_LIMITS.items()}, indent=2)}
"duration": string such as "6 Months" taken from the title/instructions,
"seo": {{
{json.dumps({k: f"{lo}-{hi} chars" for k, (lo, hi) in SEO_LIMITS.items()}, indent=2)},
  "keywords": exactly 13 strings mixing primary, secondary, long-tail, local, commercial,
              semantic, technical and student-intent keywords
}}
Field meaning: eyebrow/badge are short UI labels; ctaPrimaryText = enquiry button text;
ctaSecondaryText = curriculum/contact button text; comparisonOthers = name of a generic alternative
format (e.g. "A short introductory workshop"); comparisonNote = disclaimer that comparison rows are
criteria, not claims about a named competitor; syllabusNote = disclaimer to verify batch details.
Return only the JSON object."""

def prompt_arrays(ctx, ref):
    shape = {k: trim_example(ref.get(k)) for k in
             ["audience", "benefits", "careerRoles", "projects", "workflow",
              "whyPoints", "comparisonRows", "toolItems", "highlights"]}
    return ctx + f"""
TASK: Return ONE JSON object with these keys (shape example below is only for FORMAT, never copy its content).
Omit "id" fields (they are added by code).
RULES
- audience: exactly 5 objects {{title 20-40 chars, body 110-140 chars}} (who can join)
- benefits: exactly 6 {{title 25-40, body 100-140}}
- careerRoles: exactly 5 {{role 20-40, body 80-120}}  (realistic job roles for this course)
- projects: exactly 6 {{title 20-45, body 220-310, tags: exactly 4 strings}} practical, course-specific
- workflow: exactly 5 {{title 20-35, body 85-130}} showing a progressive learning process
- whyPoints: exactly 6 {{title 20-40, body 75-120}}
- comparisonRows: exactly 7 {{feature 8-25, ours 35-55, theirs 35-55}}  fair, no named-competitor claims
- toolItems: exactly 21 {{name, category, url}} relevant tools; official URL only if you are
  confident, otherwise ""
- highlights: exactly 6 strings, each 40-80 chars, about technical skills and practical outcomes
FORMAT EXAMPLE
{json.dumps(shape, ensure_ascii=False, indent=1)}
Return only the JSON object."""

def prompt_syllabus(ctx, ref):
    shape = trim_example({"syllabus": ref.get("syllabus", [])})
    return ctx + f"""
TASK: Return ONE JSON object {{"syllabus": [...]}} with exactly 10 detailed modules.
- If the course is 6 months, cover 24 weeks (use titles like "Weeks 1-2 | Topic"); otherwise adapt
  the schedule to the actual duration in the title.
- Beginner-to-advanced sequence; no repeated topics across modules; cover EVERY technology listed
  in the additional instructions.
- Each module: title (35-65 chars), topics (exactly 4 strings), body (140-175 chars),
  outcomes (exactly 2 strings), tools (exactly 4 strings), project (60-100 chars).
- Omit "id" (added by code).
FORMAT EXAMPLE (shape only)
{json.dumps(shape, ensure_ascii=False, indent=1)}
Return only the JSON object."""

def prompt_html(ctx, kind, sections, lo, hi, extra_rules):
    return ctx + f"""
TASK: Write the "{kind}" HTML field.
- Total length (including tags) MUST be {lo}-{hi} characters; aim for about {(lo + hi) // 2}.
- Exactly {sections} sections, each exactly:
  <h2><strong>Heading</strong></h2><p>One detailed paragraph with <strong>important keywords</strong>.</p>
- Only h2, p, strong tags. Valid HTML. No Markdown, no newlines needed between blocks.
{extra_rules}
Return ONLY the raw HTML, no code fences, no commentary."""
DESC_RULES = """- Include: technical explanations, tool/technology explanations, practical industry use cases,
  career opportunities (no guarantees), eligibility, practical projects, location-based search intent,
  direct answers to common student questions, and a natural enquiry call to action at the end."""
OVERVIEW_RULES = """- It must COMPLEMENT the description, not duplicate it. Use different headings and angles.
  Avoid these headings already used in the description: {used}"""
# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------

def slugify(title):
    s = re.sub(r"[^a-z0-9]+", "-", title.lower())
    return s.strip("-")

def with_id(item, order):
    out = {"id": str(uuid.uuid4())}
    for k in order:
        out[k] = item.get(k, "" if k not in ("topics", "outcomes", "tools", "tags") else [])
    return out

def assemble(ref, title, location, meta, arrays, syllabus, description, overview):
    g = {}
    g.update({k: v for k, v in meta.items() if k != "seo"})
    g.update(arrays)
    g.update(syllabus)
    g["audience"] = [with_id(i, ["title", "body"]) for i in g.get("audience", [])]
    g["benefits"] = [{**with_id(i, ["title", "body"]), "placement": "what-you-get"}
                     for i in g.get("benefits", [])]
    benefits_order = ["id", "placement", "title", "body"]
    g["benefits"] = [{k: b[k] for k in benefits_order} for b in g["benefits"]]
    g["careerRoles"] = [{**with_id(i, ["role", "body"]), "salaryStart": "", "salarySenior": "",
                         "market": location} for i in g.get("careerRoles", [])]
    g["projects"] = [{**with_id(i, ["title", "body", "tags"]), "demoUrl": "", "videoUrl": ""}
                     for i in g.get("projects", [])]
    g["workflow"] = [with_id(i, ["title", "body"]) for i in g.get("workflow", [])]
    g["whyPoints"] = [with_id(i, ["title", "body"]) for i in g.get("whyPoints", [])]
    g["comparisonRows"] = [with_id(i, ["feature", "ours", "theirs"]) for i in g.get("comparisonRows", [])]
    g["toolItems"] = [with_id(i, ["name", "category", "url"]) for i in g.get("toolItems", [])]
    g["syllabus"] = [with_id(i, ["title", "topics", "body", "outcomes", "tools", "project"])
                     for i in g.get("syllabus", [])]
    g["careers"] = [r["role"] for r in g["careerRoles"]]
    g["tools"] = [t["name"] for t in g["toolItems"]]
    g["description"], g["overview"] = description, overview
    g["title"], g["h1"], g["slug"] = title, title, slugify(title)
    g["ctaPrimary"] = {**ref.get("ctaPrimary", {}), "text": meta.get("ctaPrimaryText", "")}
    g["ctaSecondary"] = {**ref.get("ctaSecondary", {}), "text": meta.get("ctaSecondaryText", "")}
    seo = {**ref.get("seo", {}), **meta.get("seo", {})}
    seo.update(robotsIndex=True, inSitemap=True, faqSchema=False)
    g["seo"] = seo
    # Fixed fields always come from the reference (or required constants)
    fixed_keep = ["categoryId", "level", "mode"]
    empty_lists = ["facts", "plans", "faqIds", "reviewIds", "relatedIds",
                   "hiddenSections", "sectionOrder", "sections"]
    empty_strs = ["salary", "whyVideoUrl", "syllabusVideoUrl", "learningVideoUrl",
                  "highlightsVideoUrl", "caseVideoUrl", "certVideoUrl", "careerVideoUrl",
                  "reviewsVideoUrl", "videoUrl", "videoTitle"]
    for k in fixed_keep:
        g[k] = ref.get(k, "")
    g["segment"], g["status"], g["featured"] = "courses", "published", False
    for k in empty_lists:
        g[k] = []
    for k in empty_strs:
        g[k] = ""
    # Build in exact reference order / field set
    final = {}
    for key, ref_val in ref.items():
        val = g.get(key)
        if val is None:
            val = [] if isinstance(ref_val, list) else {} if isinstance(ref_val, dict) \
                else False if isinstance(ref_val, bool) else ""
        final[key] = val
    return final

def final_checks(ref, payload):
    issues = []
    if list(payload.keys()) != list(ref.keys()):
        issues.append("top-level fields differ from the reference payload")
    ids = []
    for v in payload.values():
        if isinstance(v, list):
            ids += [i["id"] for i in v if isinstance(i, dict) and "id" in i]
    if len(ids) != len(set(ids)):
        issues.append("duplicate UUIDs found")
    if payload.get("level") != ref.get("level") or payload.get("mode") != ref.get("mode"):
        issues.append("level/mode differ from reference")
    return issues
# --------------------------------------------------------------------------
# Short content and description
# --------------------------------------------------------------------------
SHORT_TEMPLATE_PATH = BASE / "short_template.txt"
SHORT_MIN, SHORT_MAX = 1700, 1900   # sample template is ~1,800 characters

def prompt_short(title, location, brand, extra, template):
    return f"""You write short SEO course-page content for a training institute.
NEW COURSE
Title: {title}
Location: {location}
Brand: {brand} (always spell exactly "techcadd")
Topics / tools / roles to use (from the user): {extra or "choose realistic ones for this course"}
TEMPLATE (a finished example for a DIFFERENT course):
\"\"\"
{template}
\"\"\"
TASK: Write the same kind of content for the NEW COURSE.
- Keep EXACTLY the same structure, section order, heading pattern, wording style and tone as the template.
- Each heading on its own line; each list item (topics, tools, roles) on its own line; paragraphs as one block.
- Replace the subject, topics (7), tools (5) and roles (4) to fit the new course. Reuse the new course's
  title wording in the headings the same way the template does.
- Nearby areas: use the real areas around "{location}" in the same way the template does.
- Keep the honesty lines: curriculum shared at enrolment, roles/salaries depend on the learner,
  placement support is assistance and not a guarantee, contact {brand} for fees/duration/batch timings.
- No invented salaries, statistics, reviews or guarantees.
- LENGTH: {SHORT_MIN}-{SHORT_MAX} characters including spaces and line breaks. Count carefully.
Return ONLY the plain text. No Markdown, no HTML, no code fences, no commentary."""

def validate_short(text):
    issues = []
    rng(text, SHORT_MIN, SHORT_MAX, "text", issues)
    for must in ["What you will learn", "Tools and technologies covered", "Career opportunities after",
                 "Why learn", "not a guarantee", "Contact"]:
        if must.lower() not in text.lower():
            issues.append(f'missing required part: "{must}"')
    if "techcadd" not in text:
        issues.append('brand must be spelled "techcadd"')
    if re.search(r"</?[a-zA-Z]+>|\*\*|```|^#", text, re.M):
        issues.append("plain text only: no HTML or Markdown")
    return issues
# --------------------------------------------------------------------------
# Short DESCRIPTION (HTML, same structure/length as short_template.txt)
# --------------------------------------------------------------------------

def prompt_desc_short(title, location, brand, extra, template):
    return f"""You write short SEO course-page content for a training institute.
NEW COURSE
Title: {title}
Location: {location}
Brand: {brand} (always spell exactly "techcadd")
Topics / tools / roles to use (from the user): {extra or "choose realistic ones for this course"}
TEMPLATE (plain-text example for a DIFFERENT course):
\"\"\"
{template}
\"\"\"
TASK: Write the "description" HTML field for the NEW COURSE in exactly the template's structure,
wording style and tone.
HTML FORMAT (only h2, p, strong tags):
<p>Intro paragraph with <strong>keywords</strong>.</p>
<h2><strong>About ... in city</strong></h2><p>...</p>
<h2><strong>What you will learn</strong></h2><p>Sentence, then the 7 topics separated by commas.</p>
<h2><strong>Tools and technologies covered</strong></h2><p>5 tools separated by commas.</p>
<h2><strong>Career opportunities after ...</strong></h2><p>Sentence, 4 roles separated by commas, then the salary honesty sentence.</p>
<h2><strong>Why learn ... at {brand} ...</strong></h2><p>...</p>
So: exactly 5 <h2> and exactly 6 <p> (the intro <p> comes first, before any heading).
- Keep the honesty lines: curriculum shared at enrolment, roles/salaries depend on the learner,
  placement support is assistance and not a guarantee, contact {brand} for fees/duration/batch timings.
- Nearby areas: real areas around "{location}", used as in the template.
- No invented salaries, statistics, reviews or guarantees. No Markdown.
- LENGTH: the VISIBLE TEXT (ignoring HTML tags) must be {SHORT_MIN}-{SHORT_MAX} characters.
Return ONLY the raw HTML, no code fences, no commentary."""

def validate_desc_short(html):
    issues = []
    visible = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()
    rng(visible, SHORT_MIN, SHORT_MAX, "visible text (tags excluded)", issues)
    if len(re.findall(r"<h2>", html)) != 5:
        issues.append("must contain exactly 5 <h2> sections")
    if len(re.findall(r"<p>", html)) != 6:
        issues.append("must contain exactly 6 <p> paragraphs (intro + one per section)")
    if re.search(r"</?(?!h2\b|p\b|strong\b)[a-zA-Z]+", html):
        issues.append("only h2, p and strong tags are allowed")
    if "**" in html or "```" in html:
        issues.append("no Markdown allowed")
    for must in ["What you will learn", "Tools and technologies covered", "Career opportunities after",
                 "not a guarantee", "Contact"]:
        if must.lower() not in visible.lower():
            issues.append(f'missing required part: "{must}"')
    if "techcadd" not in html:
        issues.append('brand must be spelled "techcadd"')
    return issues
# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def generate_payload(title, location, brand, extra):
    ref = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    ctx = context(title, location, brand, extra)
    llm_json, llm_text = make_llm(True), make_llm(False)
    problems = {}
    def parse_json(t):
        data = json.loads(strip_fences(t))
        if not isinstance(data, dict):
            raise ValueError("top-level JSON must be an object")
        return data
    def job_meta():
        return run_with_retries(llm_json, prompt_meta(ctx), parse_json, validate_meta)
    def job_arrays():
        return run_with_retries(llm_json, prompt_arrays(ctx, ref), parse_json, validate_arrays)
    def job_syllabus():
        return run_with_retries(llm_json, prompt_syllabus(ctx, ref), parse_json, validate_syllabus)
    def job_html():
        template = SHORT_TEMPLATE_PATH.read_text(encoding="utf-8").strip()
        desc, d_issues = run_with_retries(
            llm_text, prompt_desc_short(title, location, brand, extra, template),
            strip_fences, validate_desc_short)
        used = ", ".join(re.findall(r"<h2><strong>(.*?)</strong></h2>", desc or ""))
        ov_rules = OVERVIEW_RULES.format(used=used)
        ov, o_issues = run_with_retries(
            llm_text, prompt_html(ctx, "overview", 8, 1500, 2000, ov_rules),
            strip_fences, html_validator(1500, 2000, 8), length_hint=(1500, 2000))
        return desc, d_issues, ov, o_issues
    with ThreadPoolExecutor(max_workers=4) as pool:
        f_meta, f_arr = pool.submit(job_meta), pool.submit(job_arrays)
        f_syl, f_html = pool.submit(job_syllabus), pool.submit(job_html)
        meta, problems["meta"] = f_meta.result()
        arrays, problems["arrays"] = f_arr.result()
        syllabus, problems["syllabus"] = f_syl.result()
        desc, problems["description"], ov, problems["overview"] = f_html.result()
    payload = assemble(ref, title, location, meta or {}, arrays or {}, syllabus or {},
                       desc or "", ov or "")
    all_issues = [f"[{k}] {i}" for k, v in problems.items() for i in (v or [])]
    all_issues += final_checks(ref, payload)
    return payload, all_issues

# Course and connection routes
# ------------------------------------------------------------------------

@app.get("/")
def index():
    return render_template("index.html")

@app.post("/generate")
def generate():
    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "").strip()
    if not title:
        return jsonify(success=False, error="Title is required."), 400
    try:
        payload, issues = generate_payload(
            title=title,
            location=(data.get("location") or "Mohali, Punjab, India").strip(),
            brand=(data.get("brand") or "techcadd").strip(),
            extra=(data.get("keywords") or "").strip(),
        )
        return jsonify(success=True, payload=payload, issues=issues)
    except Exception as e:
        return jsonify(success=False, error=f"{type(e).__name__}: {e}"), 500

@app.post("/generate-short")
def generate_short():
    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "").strip()
    if not title:
        return jsonify(success=False, error="Title is required."), 400
    try:
        template = SHORT_TEMPLATE_PATH.read_text(encoding="utf-8").strip()
        prompt = prompt_short(title, (data.get("location") or "Mohali, Punjab, India").strip(),
                              (data.get("brand") or "techcadd").strip(),
                              (data.get("keywords") or "").strip(), template)
        text, issues = run_with_retries(make_llm(False), prompt, strip_fences, validate_short,
                                        length_hint=(SHORT_MIN, SHORT_MAX))
        return jsonify(success=True, text=text or "", length=len(text or ""), issues=issues)
    except Exception as e:
        return jsonify(success=False, error=f"{type(e).__name__}: {e}"), 500

@app.post("/publish")
def publish():
    """Post the (reviewed/edited) payload to the CMS API. Headers come from .env only."""
    data = request.get_json(silent=True) or {}
    payload = data.get("payload")
    if not isinstance(payload, dict) or not payload:
        return jsonify(success=False, error="No payload received."), 400
    url = os.getenv("CMS_API_URL", "https://cms.techcaddmohali.com/api/courses")
    headers = cms_headers('/courses/new', json_request=True, full_browser=True)
    try:
        r = requests.post(url, headers=headers, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), timeout=60)
        try:
            body = r.json()
        except ValueError:
            body = r.text
        return jsonify(success=r.ok, status=r.status_code, response=body), (200 if r.ok else 502)
    except requests.RequestException as e:
        return jsonify(success=False, error=f"{type(e).__name__}: {e}"), 502

@app.post("/test")
def test():
    """Quick connection check."""
    try:
        r = make_llm(False).invoke("Reply with: OpenAI connection successful.")
        return jsonify(success=True, model=os.getenv("OPENAI_MODEL"), reply=r.content)
    except Exception as e:
        return jsonify(success=False, model=os.getenv("OPENAI_MODEL"), error=str(e)), 500

# Image upload and pages
# ------------------------------------------------------------------------
API_URL = "https://cms.techcaddmohali.com/api/media"
with open(BASE / "pages_reference_payload.json", "r", encoding="utf-8") as f:
    pages_reference_payload = json.load(f)

@app.route("/imagesupload")
def imagesupload():
    return render_template(
        "imagesupload.html",
        res=session.get("image_credentials"),
        reference_payload=pages_reference_payload
    )
UPLOAD_FOLDER = os.path.abspath("uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/upload", methods=["POST"])
def upload_image():
    image = request.files["image"]
    filename = secure_filename(image.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    image.save(file_path)
    # Get complete absolute file path
    complete_path = os.path.abspath(file_path)
    print('=============',complete_path)
    image_upload_headers = cms_headers('/', json_request=False, full_browser=False)
    url = API_URL
    opened = []
    file_paths = [
            rf"{complete_path}"
        ]
    try:
        files = []
        for path in file_paths:
            if not os.path.exists(path):
                print(f"❌ File not found: {path}")
                continue
            f = open(path, "rb")
            opened.append(f)
            files.append(("files", (os.path.basename(path), f, "image/png")))
        response = requests.post(url, headers=image_upload_headers, files=files, timeout=120)
        print(response.status_code)
        print(response.text)
        image_data = response.json()[0]
        res = {
            "thumbnailId1": image_data['id'],
            "url1": image_data['url']
        }
        # Save credentials for the next request
        session["image_credentials"] = res
        # Store actual dimensions if returned by API
        session["image_dimensions"] = {
            key: image_data[key]
            for key in ("width", "height")
            if image_data.get(key) is not None
        }
        print("Image Credentials:", res)
        return render_template(
            "imagesupload.html",
            res=res,
            reference_payload=pages_reference_payload
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify(success=False, error=str(e)), 502
    finally:
        for f in opened:
            f.close()

@app.post('/generate-page')
def generate_page():
    data = request.get_json(silent=True) or {}
    title = str(data.get('title') or '').strip()
    extra = str(data.get('keywords') or '').strip()
    creds = session.get('image_credentials') or {}
    if not title:
        return jsonify(success=False, error='Topic is required.'), 400
    if not creds.get('thumbnailId1') or not creds.get('url1'):
        return jsonify(success=False, error='Upload an image first.'), 400
    try:
        ref = copy.deepcopy(pages_reference_payload)
        length = len(ref['content'])
        keyword_count = len(ref['seo']['keywords'])
        prompt = f'''Act as a senior SEO, semantic SEO, local SEO, AEO and GEO content specialist.
Topic: {title}
Brand: techcadd (always lowercase)
Additional instructions: {extra}
Write detailed, natural, student-focused content for the topic.
Use the reference page for structure, tone and approximate length, not its old subject.
Include practical learning, relevant local search intent, FAQs and an enquiry CTA.
Use valid HTML with h2, h3, p, ul, li, strong and relevant links.
Do not invent fees, placements, certificates, testimonials, rankings or batch dates.
Do not promise jobs, rankings or indexing.
Return one JSON object with exactly these keys:
content (HTML, approximately {length} characters), metaTitle (50-60 characters),
metaDescription (140-160 characters), keywords (exactly {keyword_count} strings),
alt (descriptive image alt text).
No Markdown fences. The backend assembles the full CMS payload.
REFERENCE CONTENT:
{ref['content']}
REFERENCE SEO:
{json.dumps(ref['seo'], ensure_ascii=False)}'''
        llm = make_llm(True)
        feedback = ''
        for attempt in range(MAX_ATTEMPTS):
            try:
                result = json.loads(strip_fences(llm.invoke(prompt + feedback).content))
                if not isinstance(result, dict):
                    raise ValueError('Expected a JSON object.')
                for key in ('content', 'metaTitle', 'metaDescription', 'alt'):
                    if not isinstance(result.get(key), str) or not result[key].strip():
                        raise ValueError(f'Missing or invalid {key}')
                if not isinstance(result.get('keywords'), list) or len(result['keywords']) != keyword_count or not all(isinstance(k, str) and k.strip() for k in result['keywords']):
                    raise ValueError(f'Expected {keyword_count} keyword strings')
                if not 45 <= len(result['metaTitle']) <= 65:
                    raise ValueError('metaTitle must be approximately 50-60 characters')
                if not 140 <= len(result['metaDescription']) <= 160:
                    raise ValueError('metaDescription must be 140-160 characters')
                if not 0.75 * length <= len(result['content']) <= 1.25 * length:
                    raise ValueError(f'Content must be close to {length} characters')
                if '<h2>' not in result['content'] or '<p>' not in result['content']:
                    raise ValueError('Content must contain HTML headings and paragraphs')
                break
            except Exception as exc:
                feedback = f'\nPrevious attempt failed: {exc}. Return the complete corrected JSON.'
        else:
            raise ValueError(f'Page generation failed after {MAX_ATTEMPTS} attempts: {feedback}')
        payload = copy.deepcopy(ref)
        payload['title'] = title
        payload['slug'] = slugify(title)
        payload['content'] = result['content']
        # Keep the reference publish date and all other system/navigation fields.
        payload['seo'].update(metaTitle=result['metaTitle'], metaDescription=result['metaDescription'], keywords=result['keywords'])
        for section in payload['sections']:
            if section.get('type') == 'rich-text':
                section['body'] = payload['content']
            elif section.get('type') == 'image':
                section['media'].update(id=creds['thumbnailId1'], url=creds['url1'], alt=result['alt'])
                for key in ('width', 'height'):
                    if key in (session.get('image_dimensions') or {}):
                        section['media'][key] = session['image_dimensions'][key]
        return jsonify(success=True, payload=payload)
    except Exception as exc:
        return jsonify(success=False, error=f'{type(exc).__name__}: {exc}'), 500

@app.post("/publish-page")
def publish_page():
    data = request.get_json(silent=True) or {}
    payload = data.get("payload")
    if not isinstance(payload, dict) or not payload:
        return jsonify(
            success=False,
            error="Invalid or empty JSON payload."
        ), 400
    url = "https://cms.techcaddmohali.com/api/pages"
    headers = cms_headers('/pages/new', json_request=True, full_browser=False)
    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=120
        )
        print("CMS Status:", response.status_code)
        print("CMS Response:", response.text)
        try:
            result = response.json()
        except ValueError:
            result = response.text
        return jsonify(
            success=response.ok,
            status=response.status_code,
            response=result
        ), 200 if response.ok else 502
    except requests.RequestException as e:
        return jsonify(
            success=False,
            error=str(e)
        ), 502

# Blog generation and publishing
# ------------------------------------------------------------------------
BLOG_CMS = 'https://cms.techcaddmohali.com'
BLOG_REFERENCE = Path(__file__).resolve().parent / 'blogs_reference_payload.json'
BLOG_ATTEMPTS = MAX_ATTEMPTS

def blog_headers(referer, json_request=False):
    headers = cms_headers(referer, json_request=True, full_browser=True)
    return headers

def blog_ref():
    return json.loads(BLOG_REFERENCE.read_text(encoding='utf-8'))

def blog_response(response):
    try:
        return response.json()
    except ValueError:
        return response.text
blogs_reference_payload = blog_ref()

@app.get('/blogsupload')
def blogsupload():
    return render_template('blogsupload.html', res=session.get('image_credentials'), reference_payload=blog_ref())

@app.post('/generate-blog')
def generate_blog():
    data = request.get_json(silent=True) or {}
    title = str(data.get('title') or '').strip()
    extra = str(data.get('keywords') or '').strip()
    creds = dict(session.get('image_credentials') or {})
    creds.update(session.get('image_dimensions') or {})
    if not title:
        return jsonify(success=False, error='Enter a blog topic.'), 400
    if not creds.get('thumbnailId1') or not creds.get('url1'):
        return jsonify(success=False, error='Upload a blog image first.'), 400
    try:
        ref = blog_ref()
        if not all(isinstance(creds.get(k), int) and creds[k] > 0 for k in ('width', 'height')):
            raise ValueError('Image dimensions missing from upload response. Check session image_dimensions.')
        size = len(ref['body'])
        tag_count = len(ref['tags'])
        keyword_count = len(ref['seo']['keywords'])
        prompt = f'''Act as a senior SEO strategist, semantic SEO, local SEO, AEO and GEO expert.
Write a complete student-focused blog for techcadd Mohali.
Title: {title}
Additional instructions: {extra}
Always spell the brand techcadd. Use Mohali and Chandigarh naturally when relevant.
Answer the main question early. Include a linked table of contents with matching heading IDs,
clear explanations, practical examples, learning roadmap where relevant, career context, FAQs and CTA.
Use valid HTML (p, h2, h3, ul, ol, li, strong, a). No Markdown or scripts.
Only reuse relevant internal URLs from the reference; never invent links, fees, salaries, certificates,
placements, reviews, batch dates, rankings or indexing guarantees. Label illustrative curricula clearly.
Use the reference for structure, detail and length, not its old topic.
Return ONLY one JSON object with these keys:
body (HTML, approximately {size} characters), excerpt (120-160 characters),
tags (exactly {tag_count} nonempty strings), alt (descriptive image alt text),
metaTitle (approximately 50-60 characters), metaDescription (140-160 characters),
keywords (exactly {keyword_count} nonempty strings).
Do not generate IDs, image URLs, status or other system fields; Python handles these.
REFERENCE BODY:
{ref['body']}
REFERENCE SEO:
{json.dumps(ref['seo'], ensure_ascii=False)}'''
        llm = make_llm(True)
        feedback = ''
        for attempt in range(BLOG_ATTEMPTS):
            try:
                result = json.loads(strip_fences(llm.invoke(prompt + feedback).content))
                if not isinstance(result, dict):
                    raise ValueError('Expected JSON object')
                for key, lo, hi in [('body', int(size*.7), int(size*1.3)),
                                    ('excerpt', 120, 160), ('alt', 10, 200),
                                    ('metaTitle', 45, 65), ('metaDescription', 140, 160)]:
                    val = result.get(key)
                    if not isinstance(val, str) or not lo <= len(val.strip()) <= hi:
                        raise ValueError(f'{key} must be {lo}-{hi} characters')
                for key, count in [('tags', tag_count), ('keywords', keyword_count)]:
                    val = result.get(key)
                    if not isinstance(val, list) or len(val) != count or not all(isinstance(x, str) and x.strip() for x in val):
                        raise ValueError(f'{key} must contain exactly {count} strings')
                if not re.search(r'<h2\b', result['body']) or '<p>' not in result['body']:
                    raise ValueError('Body requires HTML headings and paragraphs')
                if re.search(r'<\s*(script|iframe|object|embed)\b|\son\w+\s*=|javascript\s*:', result['body'], re.I):
                    raise ValueError('Unsafe HTML detected')
                break
            except Exception as exc:
                feedback = f'\nPrevious attempt failed: {exc}. Return the entire corrected JSON object.'
        else:
            raise ValueError(f'Blog generation failed after {BLOG_ATTEMPTS} attempts: {feedback}')
        payload = copy.deepcopy(ref)
        payload.update(title=title, slug=re.sub(r'[^a-z0-9]+', '-', title.lower().replace("'", '')).strip('-'),
                       tags=result['tags'], excerpt=result['excerpt'], body=result['body'],
                       publishDate=datetime.now(timezone(timedelta(hours=5, minutes=30))).date().isoformat())
        payload['coverImage'].update(id=creds['thumbnailId1'], url=creds['url1'], alt=result['alt'],
                                     width=creds['width'], height=creds['height'])
        payload['seo'].update(metaTitle=result['metaTitle'], metaDescription=result['metaDescription'], keywords=result['keywords'])
        return jsonify(success=True, payload=payload)
    except Exception as exc:
        return jsonify(success=False, error=str(exc)), 500

@app.post('/publish-blog')
def publish_blog():
    data = request.get_json(silent=True) or {}
    payload = data.get('payload')
    creds = dict(session.get('image_credentials') or {})
    creds.update(session.get('image_dimensions') or {})
    try:
        ref = blog_ref()
        if not isinstance(payload, dict) or set(payload) != set(ref):
            raise ValueError('Payload must have the original blog JSON fields.')
        for key in ('authorId', 'categoryId', 'status'):
            if payload[key] != ref[key]:
                raise ValueError(f'{key} must match the reference.')
        cover = payload.get('coverImage')
        if not creds or not isinstance(cover, dict) or cover.get('id') != creds.get('thumbnailId1') or cover.get('url') != creds.get('url1'):
            raise ValueError('Image credentials changed. Generate the blog again with the current image.')
        if not all(isinstance(creds.get(k), int) and creds[k] > 0 for k in ('width', 'height')):
            raise ValueError('Image dimensions missing from upload response.')
        cover.update(width=creds['width'], height=creds['height'])
        if not all(isinstance(payload.get(k), str) and payload[k].strip() for k in ('title', 'slug', 'body', 'excerpt')):
            raise ValueError('Title, slug, body and excerpt are required.')
        response = requests.post(BLOG_CMS + '/api/blogs',
                                 headers=blog_headers('/blogs/new', True),
                                 json=payload, timeout=120)
        return jsonify(success=response.ok, status=response.status_code,
                       response=blog_response(response)), 200 if response.ok else 502
    except ValueError as exc:
        return jsonify(success=False, error=str(exc)), 400
    except Exception as exc:
        return jsonify(success=False, error=str(exc)), 502
# =======================

# FAQ generation and publishing
# ------------------------------------------------------------------------
FAQ_CATEGORIES = (
    'Digital Marketing', 'Artificial Intelligence & Data',
    'Full-Stack Development', 'Cyber Security & Cloud',
    'CAD / CAM & Design', 'Core Programming', 'General',
)
FAQ_API_URL = 'https://cms.techcaddmohali.com/api/faqs'
FAQ_MAX_COUNT = 50

def faq_validate_items(items):
    if not isinstance(items, list) or not 1 <= len(items) <= FAQ_MAX_COUNT:
        raise ValueError('Expected 1 to 50 FAQs.')
    clean = []
    seen = set()
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f'FAQ {i+1} must be an object.')
        q, a = item.get('question'), item.get('answer')
        if not isinstance(q, str) or not isinstance(a, str) or not q.strip() or not a.strip():
            raise ValueError(f'FAQ {i+1} needs a question and answer.')
        q, a = q.strip(), a.strip()
        key = ' '.join(q.casefold().rstrip('?').split())
        if key in seen:
            raise ValueError(f'Duplicate question: {q}')
        seen.add(key)
        clean.append({'question': q, 'answer': a})
    return clean

@app.get('/faqsupload')
def faqsupload():
    return render_template('faqsupload.html', categories=FAQ_CATEGORIES, max_count=FAQ_MAX_COUNT)

@app.post('/generate-faqs')
def generate_faqs():
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify(success=False, error='Invalid request.'), 400
    topic = data.get('topic', '')
    category = data.get('categoryName')
    count = data.get('count')
    if not isinstance(topic, str) or not topic.strip() or len(topic) > 500:
        return jsonify(success=False, error='Enter a topic of 1–500 characters.'), 400
    if category not in FAQ_CATEGORIES or type(count) is not int or not 1 <= count <= FAQ_MAX_COUNT:
        return jsonify(success=False, error='Select a valid category and FAQ count.'), 400
    prompt = f'''Act as an SEO, AEO, GEO and semantic content specialist for techcadd Mohali.
Topic: {topic.strip()}
Category: {category}
Generate exactly {count} distinct student-focused FAQs. Use natural English and direct, helpful answers (approximately 40–80 words each where appropriate). Cover relevant student search intent, technical concepts, practical projects, eligibility, learning paths and career considerations. Use Mohali and nearby areas only where relevant. Always spell techcadd in lowercase. Do not invent fees, batch dates, certificates, placements, rankings or guarantees. Refer students to techcadd for unconfirmed details. No HTML or Markdown. Return only a JSON object with this shape: {{"items":[{{"question":"...","answer":"..."}}]}}.'''
    feedback = ''
    for attempt in range(MAX_ATTEMPTS):
        try:
            raw = make_llm(True).invoke(prompt + feedback).content
            parsed = json.loads(raw)
            items = faq_validate_items(parsed.get('items'))
            if len(items) != count:
                raise ValueError(f'Expected {count} FAQs, received {len(items)}.')
            payloads = [dict(item, categoryName=category, order=i, featured=False, status='published') for i, item in enumerate(items)]
            return jsonify(success=True, items=payloads)
        except Exception as exc:
            feedback = '\nPrevious output was invalid. Return the complete corrected JSON with exactly the requested count and distinct questions.'
            app.logger.warning('FAQ generation attempt %s failed: %s', attempt+1, type(exc).__name__)
    return jsonify(success=False, error='FAQ generation failed after three attempts. Check the model configuration and server logs.'), 502

@app.post('/publish-faq')
def publish_faq():
    # One CMS POST per FAQ. Browser sequences the calls and displays each result.
    data = request.get_json(silent=True) or {}
    payload = data.get('payload') if isinstance(data, dict) else None
    try:
        item = faq_validate_items([payload])[0]
        category = payload.get('categoryName')
        order = payload.get('order', 0)
        if category not in FAQ_CATEGORIES or type(order) is not int or order < 0:
            raise ValueError('Invalid category or order.')
        payload = dict(item, categoryName=category, order=order, featured=False, status='published')
    except ValueError as exc:
        return jsonify(success=False, error=str(exc)), 400
    headers = cms_headers('/faqs/new', json_request=True, full_browser=False)
    try:
        response = requests.post(FAQ_API_URL, headers=headers, json=payload, timeout=60, allow_redirects=False)
        try:
            result = response.json()
        except ValueError:
            result = response.text[:3000]
        success = 200 <= response.status_code < 300 and isinstance(result, (dict, list)) and not (isinstance(result, dict) and (result.get('fieldErrors') or result.get('success') is False))
        return jsonify(success=success, status=response.status_code, response=result), 200 if success else 502
    except requests.RequestException:
        return jsonify(success=False, uncertain=True, error='CMS connection failed. Verify whether the FAQ was created before retrying.'), 502

# Application entry point
# ------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=False, port=5000, threaded=True)
