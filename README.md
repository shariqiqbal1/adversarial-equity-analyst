# Multi-Agent Financial Research System

## Overview
This application implements an adversarial agentic workflow to generate balanced investment theses. Unlike standard retrieval-augmented generation (RAG) pipelines that summarize consensus, this system instantiates two distinct agents with conflicting optimization goals (Bull vs. Bear) to debate a specific asset. A third arbitrator agent synthesizes these opposing viewpoints into a final decision.

![scr](https://github.com/user-attachments/assets/863aaad0-9680-4d92-947f-706f8e351686)

## Architecture
The system utilizes a "Fan-Out / Fan-In" orchestration pattern powered by LangGraph:

1.  **Data Ingestion:** Fetches real-time pricing (yfinance) and news sentiment (Tavily).
2.  **Parallel Execution:** The state graph branches into two concurrent nodes:
    * **Bull Agent:** Prompt-engineered to prioritize growth vectors, undervaluation, and market dominance.
    * **Bear Agent:** Prompt-engineered to prioritize macro headwinds, valuation risks, and accounting irregularities.
3.  **Arbitration:** A final "CIO" node evaluates both outputs against the raw data to produce a recommendation (Buy/Sell/Hold).

## Technical Stack
* **Orchestration:** LangGraph (Stateful multi-agent execution)
* **LLM:** Claude 3.5 Sonnet/Haiku (Anthropic API)
* **Data Sources:** Yahoo Finance API, Tavily Search API
* **Frontend:** Streamlit

## Installation

### Prerequisites
* Python 3.10+
* Anthropic API Key
* Tavily API Key

### Setup
1 - Clone the repository:
   ```bash git clone https://github.com/shariqiqbal1/bear-bull-researcher.git```

2 - Install dependencies: ```pip install -r requirements.txt```

3 - Create a .env file in the root directory:
```ANTHROPIC_API_KEY=sk-ant-...```
```TAVILY_API_KEY=tvly-...```

4 - Run locally: ```streamlit run app.py```
