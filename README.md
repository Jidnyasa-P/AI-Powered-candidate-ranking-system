# Redrob Hackathon — Candidate Ranker

Hybrid, CPU-only, no-network ranker for the "AI Engineering — Ranking &
Retrieval" role at Redrob AI.

## Reproduce

```
pip install -r requirements.txt
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
python validate_submission.py ./submission.csv
```

Runs in ~45–90s on a CPU-only 16GB machine for the full 100K-candidate
pool (well inside the 5-minute / 16GB / CPU-only / no-network budget).
No GPU, no external API calls, no pre-training — TF-IDF is fit locally
on the candidate corpus at ranking time.

## Approach

Three fused signals, see `rank.py` module docstring for full detail:

1. **Text relevance** — TF-IDF (1–2 grams) over each candidate's
   headline/summary/career-history/skills, cosine-similarity against a
   JD-derived "ideal profile" query plus four targeted sub-queries
   (retrieval/embeddings, ranking & eval, LLM integration, production
   scale). Stands in for dense embeddings without needing network access
   or a model download.
2. **Rule-based structured fit** — explicit JD disqualifiers and
   preferences: title relevance, 5–9 yr experience band, product vs.
   pure-services company history (TCS/Infosys/Wipro/Accenture/Cognizant/
   Capgemini), pure-research-only career, <12mo LangChain-only "AI
   experience" with no earlier ML, senior-but-no-recent-code, job-hopping,
   CV/speech/robotics-only background, location fit (Pune/Noida
   preferred; Hyderabad/Mumbai/Delhi NCR secondary), and a keyword-stuffer
   penalty for candidates with many AI skill tags but no corroborating
   title or career-history evidence.
3. **Honeypot detection** — internal-consistency checks (years-of-
   experience vs. summed career-history months, "expert" proficiency
   with ~0 duration, implausible counts of "expert" skills, overlapping
   full-time roles, education/career timeline conflicts). Flagged
   profiles are hard-penalized (×0.05) so they fall out of the top 100
   rather than being special-cased.
4. **Behavioral modifier** — multiplicative adjustment from
   `redrob_signals`: recency of activity, `open_to_work_flag`,
   `recruiter_response_rate`, `notice_period_days`,
   `interview_completion_rate`, verified contact info, profile
   completeness. A strong-on-paper but inactive/unresponsive candidate is
   down-weighted, not eliminated.

Final score = (0.55 × text + 0.45 × rules) × disqualifier/honeypot
penalties × behavioral modifier, min-max normalized over the top 100.

## Files

- `rank.py` — the ranker (single entry point)
- `team_submission.csv` — sample output from a run against the full
  100K-candidate pool (rename/regenerate as your own submission)
- `requirements.txt`
