#!/usr/bin/env python3
"""
Redrob Hackathon — Candidate Ranker for the "AI Engineering — Ranking &
Retrieval, Series A" role (Pune/Noida, India).

Architecture
------------
A hybrid scorer, CPU-only, no network, no GPU:

1. TEXT RELEVANCE (semantic-ish, but local): TF-IDF (word 1-2 grams) over
   each candidate's (headline + summary + all career-history descriptions +
   skill names), cosine-similarity against a small set of JD-derived query
   documents (one "ideal profile" doc built from the JD, plus a handful of
   targeted sub-queries: retrieval/embeddings, ranking/eval, LLM, hybrid
   search, production ML). This is our stand-in for dense embeddings —
   deployable in <5 min on 100K rows on CPU with zero downloads.

2. STRUCTURED / RULE-BASED FIT: explicit features pulled straight from the
   JD's "read between the lines" section — title relevance, years-of-exp
   band, company-type (product vs. pure-services, e.g. TCS/Infosys/Wipro/
   Accenture/Cognizant/Capgemini), disqualifiers (pure-research-only career,
   <12mo LangChain-only "AI experience", senior-but-no-code-in-18mo,
   title-chaser job-hopping, CV/speech/robotics-only, closed-source-only
   5+ yrs), location fit (Pune/Noida/Hyderabad/Mumbai/Delhi NCR/willing to
   relocate), tenure stability.

3. HONEYPOT / IMPOSSIBLE-PROFILE DETECTION: rule checks for internally
   inconsistent profiles (e.g. duration_months at a company exceeding the
   company's plausible age is not directly available, but we DO check:
   experience-years vs. career-history total-months mismatch, "expert"
   proficiency claimed with ~0 duration_months, overlapping/implausible
   date ranges, skill counts wildly inconsistent with years of experience,
   duplicate/degenerate text). Flagged candidates are hard-penalized
   (near-zero score) rather than special-cased into the ranking, so they
   naturally fall out of the top 100.

4. BEHAVIORAL SIGNAL MODIFIER: a multiplicative modifier built from
   redrob_signals — recency of activity, recruiter_response_rate,
   open_to_work_flag, notice_period, interview_completion_rate,
   verified contacts, profile_completeness. A perfect-on-paper but
   inactive/unresponsive candidate is down-weighted, not eliminated.

Final score = clip( TEXT_SCORE * 0.40 + RULE_SCORE * 0.45 + 0.15 ) fused,
then multiplied by BEHAVIORAL_MODIFIER, then honeypot penalty applied.
See combine_scores() for the exact formula.

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv
Runs end-to-end in well under 5 minutes on a CPU-only, 16GB machine, no
network access required (TF-IDF is fit locally on the candidate corpus).
"""

import argparse
import csv
import gzip
import json
import re
import sys
import time
from datetime import date

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TODAY = date(2026, 7, 2)

# ---------------------------------------------------------------------------
# JD-derived constants (from job_description.md — "how to read between the
# lines" + "things you absolutely need" + "explicitly do NOT want")
# ---------------------------------------------------------------------------

CORE_TITLE_TERMS = [
    "machine learning engineer", "ml engineer", "applied scientist",
    "ai engineer", "search engineer", "search relevance", "ranking engineer",
    "recommendation", "recommender", "information retrieval",
    "nlp engineer", "research engineer", "data scientist", "mlops",
]

DISQUALIFYING_PURE_SERVICES = {
    "tcs", "tata consultancy services", "infosys", "wipro", "accenture",
    "cognizant", "capgemini",
}

CV_SPEECH_ROBOTICS_TERMS = [
    "computer vision", "speech recognition", "speech-to-text", "robotics",
    "autonomous driving", "slam", "lidar", "image classification only",
]

NLP_IR_TERMS = [
    "nlp", "natural language processing", "information retrieval", "ir ",
    "search", "ranking", "retrieval", "embeddings", "semantic search",
    "recommendation", "recommender",
]

CORE_AI_SKILLS = {
    "embeddings", "sentence-transformers", "sentence transformers", "bge",
    "e5", "openai embeddings", "vector database", "pinecone", "weaviate",
    "qdrant", "milvus", "opensearch", "elasticsearch", "faiss",
    "hybrid search", "bm25", "learning to rank", "ltr", "xgboost",
    "ndcg", "mrr", "map", "a/b testing", "lora", "qlora", "peft",
    "fine-tuning llms", "llm fine-tuning", "rag", "retrieval augmented generation",
    "prompt engineering", "langchain", "vector search", "ann search",
    "hnsw", "dense retrieval", "semantic search",
}

TARGET_LOCATIONS_PRIMARY = {"pune", "noida"}
TARGET_LOCATIONS_SECONDARY = {"hyderabad", "mumbai", "delhi", "delhi ncr", "gurgaon", "gurugram", "new delhi"}
TARGET_COUNTRY = "india"

JD_QUERY_DOCS = [
    # Overall ideal-profile doc
    """Senior AI engineer owning the intelligence layer of a recruiting
    platform: ranking, retrieval, and candidate-JD matching systems at
    production scale. Deep technical depth in embeddings-based retrieval
    (sentence-transformers, BGE, E5, OpenAI embeddings) deployed to real
    users, handling embedding drift, index refresh, retrieval quality
    regression. Production experience with vector databases or hybrid
    search infrastructure such as Pinecone, Weaviate, Qdrant, Milvus,
    OpenSearch, Elasticsearch, or FAISS. Strong Python and code quality.
    Hands-on experience designing evaluation frameworks for ranking
    systems: NDCG, MRR, MAP, offline-to-online correlation, A/B test
    interpretation. Shipped an end-to-end ranking, search, or
    recommendation system to real users at meaningful scale at a product
    company, not a pure services firm. Scrappy product-engineering
    attitude, willing to ship a working ranker quickly. Understood
    retrieval and ranking before it was fashionable, with substantial
    pre-LLM-era production ML experience, not just recent LangChain/OpenAI
    prototypes.""",
    # Sub-query: retrieval/embeddings
    """embeddings based retrieval systems, dense retrieval, vector search,
    hybrid search combining BM25 and dense embeddings, index refresh,
    embedding drift monitoring, retrieval quality regression testing""",
    # Sub-query: ranking & eval
    """ranking systems, learning to rank, offline evaluation NDCG MRR MAP,
    online A/B testing, recruiter engagement metrics, search relevance
    engineering""",
    # Sub-query: LLM integration
    """LLM based re-ranking, prompt engineering for ranking, LoRA QLoRA PEFT
    fine-tuning of language models, retrieval augmented generation RAG
    systems in production""",
    # Sub-query: production ML / scale
    """production machine learning systems at scale, distributed systems,
    large scale inference optimization, mentoring engineers, owning
    architecture of a recommendation or search product""",
]


def load_candidates(path):
    opener = gzip.open if path.endswith(".gz") else open
    cands = []
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cands.append(json.loads(line))
    return cands


def safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def parse_date(s):
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except Exception:
        return None


def build_text_blob(c):
    prof = c.get("profile", {})
    parts = [
        prof.get("headline", ""),
        prof.get("summary", ""),
        prof.get("current_title", ""),
    ]
    for ch in c.get("career_history", []) or []:
        parts.append(ch.get("title", ""))
        parts.append(ch.get("description", ""))
    for sk in c.get("skills", []) or []:
        # weight skill names by repeating proportional to proficiency
        w = {"expert": 4, "advanced": 3, "intermediate": 2, "beginner": 1}.get(
            sk.get("proficiency", ""), 1
        )
        parts.append((sk.get("name", "") + " ") * w)
    return " \n ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Rule-based structured fit score (0..1) + hard disqualifier flags
# ---------------------------------------------------------------------------

def rule_score(c):
    prof = c.get("profile", {})
    career = c.get("career_history", []) or []
    skills = c.get("skills", []) or []
    skill_names = {s.get("name", "").lower() for s in skills}
    title = (prof.get("current_title") or "").lower()
    company = (prof.get("current_company") or "").lower()
    location = (prof.get("location") or "").lower()
    country = (prof.get("country") or "").lower()
    yoe = prof.get("years_of_experience") or 0

    score = 0.0
    max_score = 0.0
    disqualified = False
    reasons = []

    # --- Title relevance (weight 3) ---
    max_score += 3
    title_is_core = any(t in title for t in CORE_TITLE_TERMS)
    if title_is_core:
        score += 3
    elif any(t in title for t in ["engineer", "scientist", "developer"]):
        score += 1.0
    else:
        reasons.append("current title is not obviously an ML/AI/search role")

    # --- Years of experience band (weight 2): sweet spot 5-9, tolerate 4-11 ---
    max_score += 2
    if 5 <= yoe <= 9:
        score += 2
    elif 4 <= yoe <= 11:
        score += 1.2
    elif 2 <= yoe <= 13:
        score += 0.4

    # --- Company type: product vs. pure-services (weight 2) ---
    max_score += 2
    services_companies = [
        h.get("company", "").lower() for h in career
    ] + [company]
    all_services = len(career) > 0 and all(
        any(sc in comp for sc in DISQUALIFYING_PURE_SERVICES) for comp in services_companies if comp
    )
    has_product_exp = any(
        not any(sc in comp for sc in DISQUALIFYING_PURE_SERVICES)
        for comp in services_companies if comp
    )
    if all_services and len(career) >= 2:
        disqualified = True
        reasons.append("entire career at pure consulting/services firms (JD disqualifier)")
    elif has_product_exp:
        score += 2

    # --- Core AI skills present (weight 3), but title must back it up ---
    max_score += 3
    core_hits = sum(1 for s in skill_names if any(k in s for k in CORE_AI_SKILLS))
    # NLP/IR keyword presence in career descriptions as corroboration
    career_text = " ".join((h.get("description", "") or "") for h in career).lower()
    nlp_ir_hits = sum(1 for t in NLP_IR_TERMS if t in career_text)
    if core_hits >= 2 and nlp_ir_hits >= 1:
        score += 3
    elif core_hits >= 1 or nlp_ir_hits >= 1:
        score += 1.2

    # --- Keyword-stuffer trap: many AI skills but irrelevant title/history ---
    if core_hits >= 5 and nlp_ir_hits == 0 and not any(
        t in title for t in CORE_TITLE_TERMS
    ):
        score -= 2.5
        reasons.append("possible keyword-stuffer: many AI skills but no corroborating title/history")

    # --- Pre-LLM production ML experience disqualifier ---
    # "AI experience" under 12 months using only LangChain/OpenAI, no earlier ML
    total_months = sum(h.get("duration_months", 0) or 0 for h in career)
    recent_ai_only = False
    if career:
        recent = sorted(career, key=lambda h: h.get("start_date") or "", reverse=True)[0]
        recent_desc = (recent.get("description", "") or "").lower()
        recent_dur = recent.get("duration_months", 0) or 0
        if recent_dur <= 12 and ("langchain" in recent_desc and "openai" in recent_desc):
            earlier_ml = any(
                h is not recent
                and any(t in (h.get("description", "") or "").lower() for t in NLP_IR_TERMS)
                for h in career
            )
            if not earlier_ml:
                disqualified = True
                reasons.append("AI experience limited to <12mo LangChain/OpenAI wrapper work, no earlier production ML (JD disqualifier)")

    # --- Senior-but-no-recent-code disqualifier ---
    if career:
        latest = sorted(career, key=lambda h: h.get("start_date") or "", reverse=True)[0]
        latest_title = (latest.get("title") or "").lower()
        latest_desc = (latest.get("description", "") or "").lower()
        if any(t in latest_title for t in ["architect", "tech lead", "engineering manager", "director", "vp "]) \
           and not any(t in latest_desc for t in ["wrote code", "shipped", "implemented", "built", "coded", "hands-on"]):
            score -= 1.5
            reasons.append("recent role reads as pure architecture/management, may not be hands-on coding")

    # --- Pure research disqualifier ---
    all_text = (prof.get("summary", "") + " " + career_text).lower()
    if ("research scientist" in title or "research fellow" in title or "phd" in title) and \
       not any(t in all_text for t in ["production", "deployed", "shipped", "users", "scale"]):
        disqualified = True
        reasons.append("pure research background with no evidence of production deployment (JD disqualifier)")

    # --- CV/speech/robotics-only disqualifier ---
    cv_hits = sum(1 for t in CV_SPEECH_ROBOTICS_TERMS if t in all_text)
    if cv_hits >= 2 and nlp_ir_hits == 0:
        disqualified = True
        reasons.append("primary expertise appears to be CV/speech/robotics without NLP/IR exposure (JD disqualifier)")

    # --- Title-chasing / job-hopping (weight -1.5) ---
    if len(career) >= 3:
        durations = [h.get("duration_months", 0) or 0 for h in career]
        short_stints = sum(1 for d in durations if 0 < d < 18)
        if short_stints >= len(career) - 1 and len(career) >= 3:
            score -= 1.5
            reasons.append("job-hopping pattern (mostly <18 month stints), JD flags title-chasers")

    # --- Location fit (weight 2) ---
    max_score += 2
    willing_relocate = safe_get(c, "redrob_signals", "willing_to_relocate", default=False)
    if country == "india" and any(loc in location for loc in TARGET_LOCATIONS_PRIMARY):
        score += 2
    elif country == "india" and any(loc in location for loc in TARGET_LOCATIONS_SECONDARY):
        score += 1.5
    elif country == "india":
        score += 0.7
    elif willing_relocate:
        score += 0.3

    # --- Closed-source-only 5+ years disqualifier heuristic ---
    if yoe >= 5 and "open source" not in all_text and "github" not in all_text and \
       safe_get(c, "redrob_signals", "github_activity_score", default=-1) in (-1, 0) and \
       ("proprietary" in all_text or "closed-source" in all_text or "closed source" in all_text):
        score -= 0.5
        reasons.append("no external validation of technical work (no OSS/GitHub/talks)")

    frac = max(0.0, score) / max_score if max_score else 0.0
    return frac, disqualified, reasons


# ---------------------------------------------------------------------------
# Honeypot / impossible-profile detection
# ---------------------------------------------------------------------------

def honeypot_flags(c):
    prof = c.get("profile", {})
    career = c.get("career_history", []) or []
    skills = c.get("skills", []) or []
    yoe = prof.get("years_of_experience") or 0
    flags = []

    # Career total months vastly exceeds stated years of experience
    total_months = sum(h.get("duration_months", 0) or 0 for h in career)
    if yoe > 0 and total_months > 0:
        if total_months > (yoe * 12) * 1.6 + 12:
            flags.append("career-history duration far exceeds stated years_of_experience")
        if (total_months / 12.0) < yoe * 0.35 - 1 and yoe >= 4:
            flags.append("years_of_experience far exceeds sum of career-history durations")

    # "Expert" proficiency with ~0 duration
    expert_zero = sum(
        1 for s in skills
        if s.get("proficiency") == "expert" and (s.get("duration_months") is not None) and s.get("duration_months", 0) <= 1
    )
    if expert_zero >= 2:
        flags.append(f"{expert_zero} skills marked 'expert' with ~0 months duration")

    # Excessive number of "expert" skills relative to experience
    expert_count = sum(1 for s in skills if s.get("proficiency") == "expert")
    if expert_count >= 8:
        flags.append(f"implausibly high count of 'expert' skills ({expert_count})")
    if expert_count >= 5 and yoe < 3:
        flags.append("many 'expert' skills despite very low years_of_experience")

    # Overlapping date ranges in career history (candidate at 2 jobs full time)
    ranges = []
    for h in career:
        sd = parse_date(h.get("start_date"))
        ed = parse_date(h.get("end_date")) or TODAY
        if sd:
            ranges.append((sd, ed))
    ranges.sort()
    for i in range(len(ranges) - 1):
        if ranges[i][1] and ranges[i + 1][0] and ranges[i][1] > ranges[i + 1][0]:
            overlap_days = (ranges[i][1] - ranges[i + 1][0]).days
            if overlap_days > 90:
                flags.append("overlapping full-time roles in career history")
                break

    # Education years inconsistent with experience timeline
    edu = c.get("education", []) or []
    for e in edu:
        ey = e.get("end_year")
        if ey and career:
            earliest_start = min((parse_date(h.get("start_date")) for h in career if parse_date(h.get("start_date"))), default=None)
            if earliest_start and earliest_start.year < ey - 1:
                flags.append("career start predates education end year")
                break

    return flags


# ---------------------------------------------------------------------------
# Behavioral signal modifier (0.4 .. 1.15-ish multiplier)
# ---------------------------------------------------------------------------

def behavioral_modifier(c):
    sig = c.get("redrob_signals", {}) or {}
    m = 1.0

    last_active = parse_date(sig.get("last_active_date"))
    if last_active:
        days_inactive = (TODAY - last_active).days
        if days_inactive <= 14:
            m *= 1.10
        elif days_inactive <= 30:
            m *= 1.0
        elif days_inactive <= 90:
            m *= 0.85
        elif days_inactive <= 180:
            m *= 0.65
        else:
            m *= 0.45

    if not sig.get("open_to_work_flag", False):
        m *= 0.75

    rr = sig.get("recruiter_response_rate")
    if rr is not None:
        m *= (0.55 + 0.55 * rr)  # 0 -> 0.55x, 1 -> 1.10x

    notice = sig.get("notice_period_days")
    if notice is not None:
        if notice <= 30:
            m *= 1.05
        elif notice <= 60:
            m *= 1.0
        elif notice <= 90:
            m *= 0.92
        else:
            m *= 0.85

    icr = sig.get("interview_completion_rate")
    if icr is not None:
        m *= (0.8 + 0.2 * icr)

    if sig.get("verified_email") and sig.get("verified_phone"):
        m *= 1.03

    completeness = sig.get("profile_completeness_score")
    if completeness is not None:
        m *= (0.85 + 0.15 * (completeness / 100.0))

    return max(0.25, min(1.25, m))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top_k", type=int, default=100)
    args = ap.parse_args()

    t0 = time.time()
    cands = load_candidates(args.candidates)
    print(f"Loaded {len(cands)} candidates in {time.time()-t0:.1f}s", file=sys.stderr)

    # --- Text relevance via TF-IDF (CPU-only, no network) ---
    blobs = [build_text_blob(c) for c in cands]
    vectorizer = TfidfVectorizer(
        max_features=60000, ngram_range=(1, 2), stop_words="english", min_df=2
    )
    t1 = time.time()
    doc_matrix = vectorizer.fit_transform(blobs + JD_QUERY_DOCS)
    cand_matrix = doc_matrix[: len(blobs)]
    query_matrix = doc_matrix[len(blobs):]
    sims = cosine_similarity(cand_matrix, query_matrix)  # (n_cands, n_queries)
    # weight the main JD doc more, sub-queries less
    q_weights = np.array([0.45, 0.14, 0.14, 0.14, 0.13])
    text_scores = sims.dot(q_weights)
    if text_scores.max() > 0:
        text_scores = text_scores / text_scores.max()
    print(f"TF-IDF scored in {time.time()-t1:.1f}s", file=sys.stderr)

    # --- Rule-based + honeypot + behavioral ---
    results = []
    for i, c in enumerate(cands):
        rscore, disqualified, reasons = rule_score(c)
        hp_flags = honeypot_flags(c)
        mod = behavioral_modifier(c)

        fused = 0.55 * text_scores[i] + 0.45 * rscore
        if disqualified:
            fused *= 0.15
        if hp_flags:
            fused *= 0.05  # near-zero: honeypot / impossible profile
        final = fused * mod
        results.append((c, final, text_scores[i], rscore, disqualified, hp_flags, reasons, mod))

    results.sort(key=lambda x: (-x[1], x[0]["candidate_id"]))
    top = results[: args.top_k]

    # normalize scores to a clean 0-1 range for the top 100, non-increasing
    max_final = top[0][1] if top and top[0][1] > 0 else 1.0
    scored = []
    for (c, final, tscore, rscore, dq, hp, reasons, mod) in top:
        norm_score = round(min(0.999, final / max_final * 0.98 + 0.001), 4)
        scored.append((c, norm_score, tscore, rscore, mod, reasons))

    # re-sort by rounded score desc, candidate_id asc so ties break correctly
    scored.sort(key=lambda x: (-x[1], x[0]["candidate_id"]))

    # enforce strictly non-increasing normalized score by rank (guard against
    # float noise from the above transform)
    for i in range(1, len(scored)):
        if scored[i][1] > scored[i - 1][1]:
            c, _, tscore, rscore, mod, reasons = scored[i]
            scored[i] = (c, scored[i - 1][1], tscore, rscore, mod, reasons)

    rows = [(c, rank, s, tscore, rscore, mod, reasons)
            for rank, (c, s, tscore, rscore, mod, reasons) in enumerate(scored, start=1)]

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "rank", "score", "reasoning"])
        for c, rank, score, tscore, rscore, mod, reasons in rows:
            w.writerow([
                c["candidate_id"], rank, f"{score:.4f}",
                build_reasoning(c, tscore, rscore, mod, reasons),
            ])

    print(f"Wrote top {len(rows)} to {args.out} in {time.time()-t0:.1f}s total", file=sys.stderr)


def build_reasoning(c, tscore, rscore, mod, reasons):
    prof = c.get("profile", {})
    sig = c.get("redrob_signals", {}) or {}
    title = prof.get("current_title", "unknown title")
    company = prof.get("current_company", "unknown company")
    yoe = prof.get("years_of_experience", "?")
    location = prof.get("location", "unknown location")
    rr = sig.get("recruiter_response_rate")
    last_active = sig.get("last_active_date", "unknown")

    skills = c.get("skills", []) or []
    core = sorted(
        {s.get("name") for s in skills if s.get("name", "").lower() in CORE_AI_SKILLS},
    )
    skill_bit = f"; core skills: {', '.join(core[:3])}" if core else ""

    concern = ""
    if reasons:
        concern = f" Concern: {reasons[0]}."

    rr_bit = f"; recruiter response rate {rr:.2f}" if isinstance(rr, (int, float)) else ""
    body = (
        f"{title} at {company}, {yoe} yrs, based in {location}{skill_bit}"
        f"{rr_bit}; last active {last_active}.{concern}"
    )
    return body[:400]


if __name__ == "__main__":
    main()
