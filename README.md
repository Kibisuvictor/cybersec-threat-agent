# Cybersecurity Threat Intelligence Agent — Text-to-SQL with LangChain + DuckDB

![Python](https://img.shields.io/badge/python-3.11-blue)
![LangGraph](https://img.shields.io/badge/langgraph-1.x-orange)
![Gemini](https://img.shields.io/badge/gemini-2.5--flash-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Ask questions about real-world cybersecurity threat data in plain English. A LangGraph agent writes and executes DuckDB SQL, returns results as tables, and auto-generates charts — no SQL knowledge required.

Built with **LangGraph**, **Gemini 2.5 Flash**, **DuckDB**, and **Gradio**.

---

![Demo — replace with a GIF of the agent answering a question](docs/demo.gif)

> *"Which malware families appear most frequently in the OTX threat data?"*
> → Agent writes SQL → queries DuckDB → returns ranked table + bar chart.

---

## What is a Text-to-SQL agent?

Large language models can translate natural language into SQL — but reliable results require more than a single prompt. This project uses a **LangGraph ReAct agent** that:

1. Receives the user's question alongside the live database schema
2. Calls the `execute_sql` tool with a generated DuckDB query
3. Inspects the result — if the query fails, it reads the error and retries
4. Returns a plain-English summary of the findings

A second Gemini call then inspects the result table and decides what chart to render — no hardcoded heuristics.

---

## Dataset

**Cybersecurity Threat Intelligence Dataset 2026** — sourced from AlienVault OTX, CISA's Known Exploited Vulnerabilities catalog, and the National Vulnerability Database (NVD). Originally curated and published by [chuneeb on Kaggle](https://www.kaggle.com/datasets/chuneeb/ai-cybersecurity-threat-dataset-2026).

4,312 total records across 4 tables:

| Table | Source | Records | Contains |
|---|---|---|---|
| `otx_threats` | AlienVault OTX | 2,365 | Malware families, targeted countries, attack techniques, APT groups |
| `cve_vulnerabilities` | CISA KEV + NVD | 1,585 | CVEs 2024–2026, CVSS scores, vendor/product, patch status |
| `malicious_domains` | OTX | 162 | Phishing, C2, malware domains with registrar and country |
| `malicious_ips` | OTX | 200 | IPs flagged for scanning, brute force, botnet, exfiltration |
| `cve_prioritized` | Derived view | — | CVEs with Critical/High/Medium/Low severity label |

**A note on "Unknown" values:** some fields contain "Unknown" — this reflects real-world threat intelligence where attribution is often incomplete. The agent is prompted to treat this as valid data rather than a missing value.

---

## Example questions

```
Which malware families appear most frequently?
Show me all Critical CVEs that are unpatched
Which countries are most targeted in OTX data?
Top 10 vendors with the most exploited vulnerabilities
Which domains are flagged for C2 activity?
Distribution of CVSS scores across all CVEs
Which adversary groups have the most threat pulses?
All malicious IPs associated with botnet activity
How many CVEs were published each year?
Which attack techniques appear most in OTX data?
```

---

## Architecture

```
User question (Gradio UI)
        │
        ▼
LangGraph ReAct agent
(Gemini 2.5 Flash + live schema injected into system prompt)
        │
        ├── inject schema ──► schema.py
        │                     (table names, columns, sample rows)
        │
        └── call tool ──────► execute_sql tool
                                    │
                                    ▼
                               DuckDB
                               (otx_threats, cve_vulnerabilities,
                                malicious_domains, malicious_ips)
                                    │
                                    ▼
                          Query results (DataFrame)
                                    │
                          ┌─────────┴──────────┐
                          ▼                    ▼
                    Agent answer        Chart spec LLM call
                    (plain English)     (Gemini decides chart type)
                          │                    │
                          └─────────┬──────────┘
                                    ▼
                             Gradio UI output
                         (answer + rendered chart)
```

**Agent retry loop:**
```
1. Receive question + schema context
2. LLM reasons: "I need CVEs with cvss_score >= 9.0 and patch_status = 'Unpatched'"
3. LLM calls execute_sql("SELECT cve_id, vendor, cvss_score FROM ...")
4. Tool runs query, returns markdown table
5. LLM summarises key finding in plain English
6. Second LLM call inspects result → returns JSON chart spec → chart renders
7. If SQL error at step 4 → LLM reads error, rewrites query, retries
```

---

## Tech stack

| Component | Tool | Why |
|---|---|---|
| LLM | [Gemini 2.5 Flash](https://ai.google.dev) | Fast, strong tool calling, generous free tier (1,500 req/day) |
| Agent framework | [LangGraph](https://langchain-ai.github.io/langgraph) | ReAct agent loop, replaces deprecated AgentExecutor |
| Orchestration | [LangChain 1.x](https://python.langchain.com) | Tool definitions, message types |
| Database | [DuckDB](https://duckdb.org) | Fast analytical SQL, zero config, reads CSVs natively |
| UI | [Gradio 6](https://gradio.app) | Clean chat interface, local and Hugging Face Spaces compatible |

---

## Project structure

```
cybersec-threat-agent/
├── agent.py                  # LangGraph ReAct agent + SQL tool
├── app.py                    # Gradio 6 UI with dynamic charting
├── schema.py                 # Live schema introspection from DuckDB
├── data/
│   ├── load_data.py          # Loads Kaggle CSVs into DuckDB
│   ├── raw/                  # Raw CSVs — git-ignored, re-download from Kaggle
│   └── cybersec.duckdb       # DuckDB database — git-ignored, built locally
├── tests/
│   └── test_sql_tool.py      # SQL tool + database integrity tests
├── .env.example              # Template — copy to .env and add your API key
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quickstart

### Prerequisites

- Python 3.11+
- A free Gemini API key from [Google AI Studio](https://aistudio.google.com) (no credit card needed)
- Kaggle account to download the dataset

### 1. Clone and install

```bash
git clone https://github.com/Kibisuvictor/cybersec-threat-agent.git
cd cybersec-threat-agent

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Add your API key

```bash
cp .env.example .env
```

Open `.env` and add your key:

```
GOOGLE_API_KEY=your_key_here
```

Get a free key at [aistudio.google.com](https://aistudio.google.com) — the free tier covers 1,500 requests/day, more than enough for local use.

### 3. Download the dataset

Download the four CSVs from [Kaggle](https://www.kaggle.com/datasets/chuneeb/ai-cybersecurity-threat-dataset-2026) and place them in `data/raw/`:

```
data/raw/otx_threat_intelligence.csv
data/raw/cve_vulnerabilities.csv
data/raw/malicious_domains.csv
data/raw/malicious_ips.csv
```

### 4. Load into DuckDB

```bash
python data/load_data.py
```

### 5. Verify the schema

```bash
python schema.py
```

You should see all four tables with their columns and a sample row each.

### 6. Run the app

```bash
python app.py
```

Open [http://localhost:7860](http://localhost:7860) and start querying.

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_API_KEY` | ✅ | Gemini API key from Google AI Studio |

Copy `.env.example` to `.env` and fill in your key. Never commit `.env` — it is git-ignored.

---

## Running tests

```bash
pytest tests/ -v
```

Tests skip gracefully when `cybersec.duckdb` hasn't been built yet — safe for environments without the Kaggle data.

---

## Design decisions

**Why LangGraph over the old AgentExecutor?**
LangChain 1.x fully deprecated `AgentExecutor` and `create_tool_calling_agent`. LangGraph's `create_react_agent` is the current standard — it handles the ReAct loop natively as a state machine, with cleaner error recovery and no need to configure `max_iterations` or `handle_parsing_errors` manually.

**Why dynamic chart generation instead of heuristics?**
The previous approach used keyword matching (`if "malware" in question`) to decide what chart to draw. This was brittle — it missed paraphrased questions and produced wrong chart types. The current approach makes a second Gemini call that receives the actual result table and returns a JSON chart spec (`{"type": "barh", "x": "malware_family", "y": "count"}`). The chart is always appropriate to the data returned, not the wording of the question.

**Why inject the live schema rather than hardcode it?**
`schema.py` introspects the database at runtime and includes a sample row from each table. This gives the agent accurate column names, data types, and real value examples — without any hardcoded prompt maintenance. Add a new table and the agent picks it up automatically.

**Why DuckDB over SQLite?**
DuckDB handles GROUP BY aggregations and analytical queries significantly faster than SQLite. It also reads CSV natively, removing a separate ingestion layer.

---

## What I would add next

- **Ollama fallback** — detect whether `GOOGLE_API_KEY` is set and fall back to a local Ollama model (e.g. `llama3.1`) for fully offline use
- **Hugging Face Spaces deployment** — add `GOOGLE_API_KEY` as a Space secret and push; the app runs without changes
- **Threat timeline** — plot CVE publication dates and OTX pulse dates together to visualise attack wave patterns
- **MITRE ATT&CK mapping** — tag OTX attack techniques against the MITRE framework for tactic/technique analysis
- **RAGAS evaluation** — automated agent evaluation using answer relevancy and faithfulness metrics

---

## Cost

This project runs on Gemini's free tier — no credit card required.

| Tier | Requests/day | Cost |
|---|---|---|
| Free (Google AI Studio) | 1,500 | $0 |
| Paid (if needed) | Unlimited | ~$0.001–0.002 per question |

Each question makes multiple API calls (Call 1: The Agent decides which tool to use.

Call 2: The Agent generates the SQL query.

Call 3: The Agent processes the DuckDB results.

Call 4 (Visualizer): Your UI makes an extra call to decide which chart to show.). One user interaction can easily consume 3–5 requests from your quota. I plan to implement a fallback for a local model that is free like Ollama or qwen2.5.

---

## Data attribution

Dataset curated by [chuneeb](https://www.kaggle.com/datasets/chuneeb/ai-cybersecurity-threat-dataset-2026).
Original sources: AlienVault OTX, CISA Known Exploited Vulnerabilities Catalog, NVD — all public and open access.

---

## License

MIT — see [LICENSE](LICENSE).

---

## Author

**Victor** — Data Scientist
[LinkedIn](https://linkedin.com/in/victor-kibisu) · [GitHub](https://github.com/Kibisuvictor)