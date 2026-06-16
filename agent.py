# agent.py
import duckdb
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
load_dotenv()
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from schema import get_schema_context, DB_PATH

# Track which LLM is currently being used
current_llm_source = "gemini"  # "gemini" or "ollama"


# ── SQL tool ──────────────────────────────────────────────────────────────────

@tool
def execute_sql(query: str) -> str:
    """
    Execute a DuckDB SQL query and return the results as a markdown table.
    Use this to answer any question about the dataset.
    Always write valid DuckDB SQL. Limit results to 20 rows unless asked otherwise.
    """
    try:
        con = duckdb.connect(DB_PATH, read_only=True)
        df = con.execute(query).fetchdf()
        con.close()

        if df.empty:
            return "Query returned no results."

        return df.to_markdown(index=False)

    except Exception as e:
        return f"SQL error: {str(e)}. Rewrite the query and try again."


# ── LLM Fallback Wrapper ──────────────────────────────────────────────────────

class FallbackLLM:
    """Wrapper that tries Ollama first (local), falls back to Gemini (API) on error."""

    def __init__(self):
        self.ollama = ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "mistral"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0,
        )
        self.gemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        self.current_source = "ollama"
        self._current_model = self.ollama
        self._fallback_triggered = False

    def invoke(self, input_data):
        """Try Ollama first (local), fall back to Gemini (API) on error."""
        global current_llm_source
        
        # Try Ollama first (local, fast)
        if not self._fallback_triggered:
            try:
                result = self.ollama.invoke(input_data)
                self.current_source = "ollama"
                self._current_model = self.ollama
                current_llm_source = "ollama"
                return result
            except Exception as e:
                print(f"\n⚠️  Ollama failed: {str(e)[:150]}")
                print("   Falling back to Gemini API...\n")
                self._fallback_triggered = True
        
        # Use Gemini (fallback)
        try:
            result = self.gemini.invoke(input_data)
            self.current_source = "gemini"
            self._current_model = self.gemini
            current_llm_source = "gemini"
            print("✓ Processing with Gemini API (fallback)\n")
            return result
        except Exception as e:
            print(f"❌ Both Ollama and Gemini failed. Gemini error: {str(e)}")
            raise Exception(f"Both Ollama and Gemini failed. Last error: {str(e)}")

    def batch(self, input_list):
        """Batch invoke with fallback."""
        return [self.invoke(inp) for inp in input_list]

    def __getattr__(self, name):
        """Delegate unknown attributes to the current model."""
        return getattr(self._current_model, name)


# ── Agent builder ─────────────────────────────────────────────────────────────

def build_agent():
    schema = get_schema_context()

    system_prompt = f"""You are a cybersecurity threat intelligence analyst assistant.
You have access to a DuckDB database with 4,312 records from AlienVault OTX,
CISA's Known Exploited Vulnerabilities catalog, and the NVD.

DATABASE SCHEMA:
{schema}

TABLES:
- otx_threats        : Threat pulses - malware families, targeted countries,
                       attack techniques, adversary/APT groups (2,365 records)
- cve_vulnerabilities: Actively exploited CVEs 2024-2026, 
                       vendor/product (1,585 records)
- malicious_domains  : Domains flagged for phishing, C2, malware (162 records)
- malicious_ips      : IPs reported for scanning, brute force, botnet (200 records)

INSTRUCTIONS:
- Always use execute_sql to answer data questions
- Write clean DuckDB SQL - use ILIKE for case-insensitive string matching
- After results, summarise the key finding in 1-2 sentences
- Note when values are "Unknown" - this is real-world data, not a data error
- For rankings use ORDER BY ... DESC LIMIT 10
- For distributions use GROUP BY + COUNT(*)
- Never fabricate threat data - only report what the tool returns
"""

    llm = FallbackLLM()
    tools = [execute_sql]

    # create_react_agent is the LangGraph replacement for AgentExecutor +
    # create_tool_calling_agent. Works natively with LangChain 1.x.
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
    )

    return agent


# ── Query helper ──────────────────────────────────────────────────────────────

def query(agent, question: str, chat_history: list = None):
    messages = []

    # Replay chat history if provided (list of HumanMessage / AIMessage)
    if chat_history:
        messages.extend(chat_history)

    messages.append(HumanMessage(content=question))

    result = agent.invoke({"messages": messages})

    # LangGraph returns the full message list - last message is the answer
    return result["messages"][-1].content


if __name__ == "__main__":
    agent = build_agent()
    answer = query(
        agent,
        "Which is the most common vendor project?"
    )
    print(answer)