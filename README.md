# AI-Powered Candidate Ranking System

### 🚀 Intelligent Hybrid AI Recruiter for Semantic Candidate Ranking

<p align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge\&logo=python\&logoColor=white)
![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge\&logo=scikitlearn\&logoColor=white)
![CPU Only](https://img.shields.io/badge/CPU--Only-16GB-success?style=for-the-badge)
![No Network](https://img.shields.io/badge/No-Network-red?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)

</p>

---

# 📌 Overview

Traditional Applicant Tracking Systems (ATS) depend heavily on keyword matching, causing highly qualified candidates to be overlooked simply because they use different terminology.

This project was built for the **Redrob AI Hackathon** to solve that problem using a **hybrid AI ranking pipeline** that understands both **candidate context** and **job requirements**, while satisfying strict production constraints:

* ✅ CPU Only
* ✅ No Internet Access
* ✅ No External APIs
* ✅ Under 5-minute execution
* ✅ Fully Reproducible

Instead of relying on expensive LLM calls or dense embedding APIs, the system combines **semantic retrieval**, **rule-based intelligence**, **behavioral signals**, and **honeypot detection** to produce recruiter-quality rankings.

---

# 🎯 Problem Statement

Recruiters review hundreds of resumes every day.

Traditional ATS systems:

* Match keywords instead of meaning
* Ignore career progression
* Miss adjacent skills
* Cannot detect fake or inconsistent profiles
* Produce unreliable rankings

Our solution ranks candidates **the way an experienced recruiter would**, considering the entire professional profile instead of isolated keywords.

---

# ✨ Features

* 🔍 Semantic Candidate Retrieval
* 📄 Intelligent Job Description Understanding
* ⚡ Hybrid Ranking Engine
* 📊 Behavioral Signal Scoring
* 🛡️ Honeypot Candidate Detection
* 📈 Explainable Candidate Ranking
* 🚀 CPU-only Execution
* 🔒 Fully Offline (No Network Calls)
* 📁 CSV/XLSX Export
* ✅ Reproducible Pipeline

---

# 🏗️ System Architecture

```
                   Job Description
                          │
                          ▼
                Query Understanding Engine
                          │
                          ▼
            TF-IDF Semantic Query Generation
                          │
                          ▼
        Candidate Corpus Vector Representation
                          │
                          ▼
          Hybrid Multi-Signal Scoring Engine
          ├── Semantic Similarity
          ├── Rule-Based Scoring
          ├── Behavioral Signals
          └── Honeypot Detection
                          │
                          ▼
              Final Candidate Ranking
                          │
                          ▼
              Ranked CSV / XLSX Output
```

---

# ⚙️ Ranking Methodology

The ranking pipeline combines **four complementary scoring strategies**.

## 1️⃣ Semantic Text Relevance

The system builds TF-IDF vectors over:

* Resume Summary
* Headline
* Career History
* Skills
* Experience

The Job Description is converted into multiple semantic search queries including:

* Core Role
* Retrieval Systems
* Embeddings
* Ranking
* LLM Integration
* Production Scale AI

Each candidate receives a cosine similarity score.

---

## 2️⃣ Structured Rule-Based Scoring

Beyond text similarity, explicit hiring preferences are evaluated.

Examples include:

* Relevant Job Titles
* Experience Band (5–9 years)
* Product Company Experience
* Career Stability
* Leadership Growth
* AI/ML Experience
* Location Preference
* Domain Expertise

The engine also penalizes:

* Keyword Stuffing
* Pure Research Profiles
* Pure Service-Based Backgrounds
* Recent AI-only Experience
* CV-only / Robotics-only Profiles
* Excessive Job Hopping

---

## 3️⃣ Honeypot Detection

To prevent fake candidates from ranking highly, internal consistency checks identify suspicious profiles.

The detector flags:

* Impossible Career Timelines
* Overlapping Full-Time Jobs
* Expert Skills with Zero Experience
* Unrealistic Skill Counts
* Timeline Conflicts
* Inflated Experience

Flagged profiles receive a heavy ranking penalty.

---

## 4️⃣ Behavioral Signal Modifier

Professional activity also matters.

Signals considered include:

* Recruiter Response Rate
* Open to Work
* Interview Completion Rate
* Notice Period
* Profile Completeness
* Verified Contact Information
* Platform Activity

Strong but inactive candidates are slightly down-ranked instead of being removed.

---

# 🧮 Final Scoring Formula

```
Final Score

= (0.55 × Semantic Score
 + 0.45 × Rule Score)

× Behavioral Modifier

× Honeypot Penalty

→ Min-Max Normalization

→ Final Rank
```

---

# 📊 Candidate Evaluation Signals

| Signal               | Used |
| -------------------- | ---- |
| Semantic Similarity  | ✅    |
| Skills               | ✅    |
| Career History       | ✅    |
| Experience           | ✅    |
| Job Titles           | ✅    |
| Behavioral Signals   | ✅    |
| Notice Period        | ✅    |
| Recruiter Response   | ✅    |
| Profile Completeness | ✅    |
| Honeypot Detection   | ✅    |
| Timeline Consistency | ✅    |

---

# 🚀 Installation

```bash
git clone https://github.com/Jidnyasa-P/AI-Powered-candidate-ranking-system.git

cd AI-Powered-candidate-ranking-system

pip install -r requirements.txt
```

---

# ▶️ Run

Generate the ranked submission:

```bash
python rank.py \
--candidates ./candidates.jsonl \
--out ./submission.csv
```

Validate the submission:

```bash
python validate_submission.py ./submission.csv
```

---

# ⏱ Performance

| Metric            | Value          |
| ----------------- | -------------- |
| Runtime           | ~45–90 seconds |
| Hardware          | CPU Only       |
| RAM               | <16 GB         |
| Internet          | ❌ No           |
| GPU               | ❌ No           |
| API Calls         | ❌ No           |
| Submission Format | CSV            |
| Ranking Output    | Top 100        |

Fully compliant with the Redrob Hackathon compute constraints.

---

# 📦 Output

The system generates:

* Ranked Candidate CSV
* Candidate Scores
* Explainable Ranking
* XLSX Output (Optional)

---

# 🛠 Technologies Used

* Python 3.11
* Pandas
* NumPy
* Scikit-learn
* TF-IDF Vectorizer
* Cosine Similarity
* YAML
* JSONL

---

# 🎯 Why This Approach?

Unlike traditional ATS systems, our ranking engine combines:

* Semantic Retrieval
* Structured Intelligence
* Behavioral Analytics
* Fraud Detection

This hybrid approach produces recruiter-quality rankings while remaining lightweight, deterministic, reproducible, and suitable for production environments.

---

# 📚 Reproducibility

Install dependencies

```bash
pip install -r requirements.txt
```

Generate submission

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

Validate

```bash
python validate_submission.py ./submission.csv
```

---

# 👨‍💻 Team

**Team:** InnoQueens

**Hackathon:** Redrob AI Candidate Ranking Challenge

---

# 📄 License

This project is developed for the **Redrob AI Hackathon** and is intended for educational and evaluation purposes.
