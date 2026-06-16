# app.py — Gradio 6 compatible
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import re
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

load_dotenv()

from agent import build_agent, query, DB_PATH, current_llm_source

# ── Globals ───────────────────────────────────────────────────────────────────

agent_executor = build_agent()
chat_history   = []

EXAMPLE_QUESTIONS = [
    "Which malware families appear most frequently?",
    "Show me all Critical CVEs that are unpatched",
    "Which countries are most targeted in OTX data?",
    "Top 10 vendors with the most exploited vulnerabilities",
    "Which domains are flagged for C2 activity?",
    "Distribution of CVSS scores across all CVEs",
    "Which adversary groups have the most threat pulses?",
    "All malicious IPs associated with botnet activity",
    "How many CVEs were published each year?",
    "Which attack techniques appear most in OTX data?",
]

# ── Dynamic chart generation ──────────────────────────────────────────────────

CHART_SYSTEM_PROMPT = """You are a data visualisation assistant.
Given an analyst's question and a markdown table of results, decide if a chart
would help, and if so return a JSON chart specification.

Rules:
- Only return JSON, no explanation, no markdown fences.
- If no chart is useful return: {"chart": false}
- If a chart is useful return one of these structures:

Bar chart:
{"chart": true, "type": "bar", "x": "<column_name>", "y": "<column_name>", "title": "<title>"}

Horizontal bar chart (use when x-axis labels are long strings):
{"chart": true, "type": "barh", "x": "<column_name>", "y": "<column_name>", "title": "<title>"}

Line chart (use for time series / year columns):
{"chart": true, "type": "line", "x": "<column_name>", "y": "<column_name>", "title": "<title>"}

Pie chart (use only for distributions with <= 8 categories):
{"chart": true, "type": "pie", "labels": "<column_name>", "values": "<column_name>", "title": "<title>"}

Column names must exactly match the headers in the markdown table.
"""


def parse_table_from_markdown(markdown: str):
    lines = [l.strip() for l in markdown.strip().splitlines()
             if l.strip().startswith("|")]
    data_lines = [l for l in lines if not re.match(r"^\|[-| :]+\|$", l)]
    if len(data_lines) < 2:
        return None
    try:
        import pandas as pd
        headers = [h.strip() for h in data_lines[0].strip("|").split("|")]
        rows = []
        for line in data_lines[1:]:
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) == len(headers):
                rows.append(cells)
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=headers)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="ignore")
        return df
    except Exception:
        return None


def get_chart_llm():
    """Create a fallback LLM for chart generation (Gemini -> Ollama)."""
    class ChartFallbackLLM:
        def __init__(self):
            self.gemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
            self.ollama = ChatOllama(
                model=os.getenv("OLLAMA_MODEL", "mistral"),
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                temperature=0,
            )

        def invoke(self, input_data):
            try:
                return self.gemini.invoke(input_data)
            except Exception as e:
                print(f"⚠️  Gemini chart failed: {str(e)[:100]}. Falling back to Ollama...")
                try:
                    return self.ollama.invoke(input_data)
                except Exception as e2:
                    print(f"❌ Ollama chart also failed: {str(e2)}")
                    raise
    return ChartFallbackLLM()


def get_dynamic_chart(question: str, agent_answer: str):
    table_match = re.search(
        r"(\|.+\|[\s\S]+?\|[-| :]+\|[\s\S]+?)(?:\n\n|\Z)", agent_answer
    )
    if not table_match:
        return None

    table_str = table_match.group(1).strip()
    df = parse_table_from_markdown(table_str)
    if df is None or df.empty:
        return None

    chart_llm = get_chart_llm()
    user_msg  = f"Question: {question}\n\nResults table:\n{table_str}"

    response = chart_llm.invoke([
        {"role": "system", "content": CHART_SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ])

    raw = response.content.strip()
    raw = re.sub(r"^```json\s*|^```\s*|```$", "", raw, flags=re.MULTILINE).strip()

    try:
        spec = json.loads(raw)
    except json.JSONDecodeError:
        return None

    if not spec.get("chart"):
        return None

    return render_chart(df, spec)


def render_chart(df, spec: dict):
    chart_type = spec.get("type", "bar")

    BG      = "#0d1117"
    FG      = "#e6edf3"
    ACCENT  = "#f85149"
    GRID    = "#21262d"
    BAR_CLR = "#388bfd"

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.tick_params(colors=FG, labelsize=9)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)
    ax.grid(axis="y", color=GRID, linewidth=0.5, linestyle="--")

    try:
        import pandas as pd
        if chart_type == "bar":
            x_col, y_col = spec["x"], spec["y"]
            ax.bar(df[x_col].astype(str),
                   pd.to_numeric(df[y_col], errors="coerce"),
                   color=BAR_CLR, edgecolor=BG, linewidth=0.5)
            ax.set_xlabel(x_col, fontsize=9)
            ax.set_ylabel(y_col, fontsize=9)
            plt.xticks(rotation=35, ha="right", fontsize=8)

        elif chart_type == "barh":
            x_col, y_col = spec["x"], spec["y"]
            ax.barh(df[x_col].astype(str),
                    pd.to_numeric(df[y_col], errors="coerce"),
                    color=BAR_CLR, edgecolor=BG, linewidth=0.5)
            ax.set_xlabel(y_col, fontsize=9)
            ax.set_ylabel(x_col, fontsize=9)
            ax.grid(axis="x", color=GRID, linewidth=0.5, linestyle="--")
            ax.grid(axis="y", visible=False)
            ax.invert_yaxis()

        elif chart_type == "line":
            x_col, y_col = spec["x"], spec["y"]
            x_vals = df[x_col].astype(str)
            y_vals = pd.to_numeric(df[y_col], errors="coerce")
            ax.plot(x_vals, y_vals, color=ACCENT, linewidth=2,
                    marker="o", markersize=4, markerfacecolor=FG)
            ax.fill_between(range(len(x_vals)), y_vals,
                            alpha=0.15, color=ACCENT)
            ax.set_xticks(range(len(x_vals)))
            ax.set_xticklabels(x_vals, rotation=35, ha="right", fontsize=8)

        elif chart_type == "pie":
            lbl_col, val_col = spec["labels"], spec["values"]
            labels = df[lbl_col].astype(str).tolist()
            values = pd.to_numeric(df[val_col], errors="coerce").tolist()
            colors = plt.cm.Set2.colors
            wedges, texts, autotexts = ax.pie(
                values, labels=labels, autopct="%1.0f%%",
                colors=colors[:len(values)],
                wedgeprops={"edgecolor": BG, "linewidth": 1.5},
                textprops={"color": FG, "fontsize": 8},
            )
            for at in autotexts:
                at.set_color(BG)
                at.set_fontsize(7)

        ax.set_title(spec.get("title", ""), fontsize=11, pad=10, fontweight="bold")
        plt.tight_layout(pad=1.5)
        return fig

    except Exception:
        plt.close(fig)
        return None


# ── Response handler ──────────────────────────────────────────────────────────

def respond(question: str, history: list):
    global chat_history

    if not question.strip():
        return history, None

    answer = query(agent_executor, question, chat_history)
    
    # Add LLM source indicator
    from agent import current_llm_source
    llm_badge = "� Ollama (Local)" if current_llm_source == "ollama" else "🔵 Gemini (API)"
    answer_with_badge = f"{answer}\n\n---\n*Processing with: {llm_badge}*"

    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=answer))

    history = history or []
    
    history.append({"role": "user",      "content": question})
    history.append({"role": "assistant", "content": answer_with_badge})

    chart = get_dynamic_chart(question, answer)
    return history, chart


def clear_chat():
    global chat_history
    chat_history = []
    return [], None, ""


# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg-primary:   #0d1117;
    --bg-secondary: #161b22;
    --bg-tertiary:  #21262d;
    --border:       #30363d;
    --text-primary: #e6edf3;
    --text-secondary:#8b949e;
    --accent-red:   #f85149;
    --accent-blue:  #388bfd;
    --accent-green: #3fb950;
    --accent-orange:#d29922;
}

body, .gradio-container {
    background: var(--bg-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text-primary) !important;
}

.app-header {
    padding: 1.5rem 0 1rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
}
.app-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.3rem;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0;
}
.app-title span { color: var(--accent-red); }
.app-subtitle {
    font-size: 0.75rem;
    color: var(--text-secondary);
    margin: 4px 0 0;
    font-family: 'JetBrains Mono', monospace;
}
.status-row {
    display: flex;
    gap: 8px;
    margin-top: 10px;
    flex-wrap: wrap;
}
.pill {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    padding: 2px 10px;
    border-radius: 20px;
    border: 1px solid;
}
.pill-green  { border-color: var(--accent-green);  color: var(--accent-green);  }
.pill-blue   { border-color: var(--accent-blue);   color: var(--accent-blue);   }
.pill-orange { border-color: var(--accent-orange); color: var(--accent-orange); }

.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
}

/* Chatbot */
.chatbot { background: var(--bg-secondary) !important; border: 1px solid var(--border) !important; border-radius: 8px !important; }

/* Input */
textarea {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.875rem !important;
}
textarea:focus { border-color: var(--accent-blue) !important; }

/* Buttons */
.btn-ask {
    background: var(--accent-blue) !important;
    border: none !important;
    border-radius: 8px !important;
    color: #fff !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
}
.btn-ask:hover { opacity: 0.85 !important; }
.btn-clear {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-secondary) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
}
.btn-clear:hover { border-color: var(--accent-red) !important; color: var(--accent-red) !important; }
/* Add to CSS string */
.gradio-textbox textarea {
    min-height: 80px !important;
    font-size: 0.9rem !important;
    padding: 12px !important;
}

.btn-ask {
    min-height: 52px !important;
    width: 100% !important;
}

.btn-clear {
    min-height: 32px !important;
    width: 100% !important;
}
"""

# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Cybersec Threat Intel Agent") as demo:

    gr.HTML("""
    <div class="app-header">
        <p class="app-title"><span>//</span> cybersec threat intel agent</p>
        <p class="app-subtitle">4,312 records · AlienVault OTX · CISA KEV · NVD · Gemini 2.5 Flash</p>
        <div class="status-row">
            <span class="pill pill-green">● duckdb connected</span>
            <span class="pill pill-blue">● gemini-2.5-flash</span>
            <span class="pill pill-orange">● langgraph agent</span>
        </div>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=5):
            gr.HTML('<p class="section-label">Conversation</p>')

            chatbot = gr.Chatbot(
                height=360,
                show_label=False,
                placeholder=(
                    "<div style='text-align:center;font-family:JetBrains Mono,monospace;"
                    "font-size:0.8rem;color:#8b949e;padding:2rem'>"
                    "Ask a question about the threat intelligence database"
                    "</div>"
                ),
            )

            with gr.Row():
                question_input = gr.Textbox(
                    show_label=False,
                    placeholder="e.g. Which malware families appear most frequently?",
                    lines=2,
                    scale=5,
                    min_width=400,  #prevents it collapsing on narrow screens
                )
                with gr.Column(scale=1, min_width=80):
                    ask_btn   = gr.Button("Ask →",  elem_classes=["btn-ask"],   size="lg")
                    clear_btn = gr.Button("Clear",   elem_classes=["btn-clear"], size="sm")

            gr.HTML('<p class="section-label" style="margin-top:12px">Try these</p>')
            gr.Examples(
                examples=[[q] for q in EXAMPLE_QUESTIONS],
                inputs=question_input,
                label=None,
            )

        with gr.Column(scale=2):
            gr.HTML('<p class="section-label">Visualisation</p>')
            chart_output = gr.Plot(show_label=False, container=False)
            gr.HTML("""
            <p style="font-family:JetBrains Mono,monospace;font-size:0.68rem;
            color:#8b949e;margin-top:8px;text-align:center">
            Charts generated dynamically by the agent based on query results
            </p>
            """)

    # ── Events ────────────────────────────────────────────────────────────────
    ask_btn.click(
        respond,
        inputs=[question_input, chatbot],
        outputs=[chatbot, chart_output],
    ).then(lambda: "", outputs=question_input)

    question_input.submit(
        respond,
        inputs=[question_input, chatbot],
        outputs=[chatbot, chart_output],
    ).then(lambda: "", outputs=question_input)

    clear_btn.click(
        clear_chat,
        outputs=[chatbot, chart_output, question_input],
    )


if __name__ == "__main__":
    demo.launch(
        css=CSS,              # css moves to launch() in Gradio 6
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )