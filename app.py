import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from typing import TypedDict
import yfinance as yf
from tavily import TavilyClient
import os
from dotenv import load_dotenv

# Load Environment
load_dotenv()

# --- CONFIG ---
# Ensure you set these in your environment or Streamlit secrets
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    # Fallback for Streamlit Cloud deployment
    TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY")

tavily = TavilyClient(api_key=TAVILY_API_KEY)

# Initialize LLM
# Note: Ensure ANTHROPIC_API_KEY is set in env or secrets
llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0.5)

# --- TOOLS ---
def get_stock_data(ticker):
    """Fetches price, P/E, and market cap."""
    try:
        stock = yf.Ticker(ticker)
        # Fast info fetch
        info = stock.info
        hist = stock.history(period="1mo")
        
        if hist.empty:
            return {"error": "No history found"}

        current_price = info.get('currentPrice', hist.iloc[-1]['Close'])
        market_cap = info.get('marketCap', "N/A")
        pe_ratio = info.get('forwardPE', "N/A")
        sector = info.get('sector', "N/A")
        
        start_price = hist.iloc[0]['Close']
        end_price = hist.iloc[-1]['Close']
        trend = "Up" if end_price > start_price else "Down"
        
        return {
            "current_price": current_price,
            "market_cap": market_cap,
            "pe_ratio": pe_ratio,
            "sector": sector,
            "price_trend": trend
        }
    except Exception as e:
        return {"error": str(e)}

def get_news(ticker):
    """Searches for recent news headlines."""
    try:
        response = tavily.search(query=f"{ticker} stock news analysis", search_depth="basic", max_results=3)
        return "\n".join([f"- {r['content']}" for r in response['results']])
    except Exception as e:
        return f"Could not fetch news: {str(e)}"

# --- AGENT STATE ---
class AgentState(TypedDict):
    ticker: str
    stock_data: dict
    news_summary: str
    bull_case: str
    bear_case: str
    final_verdict: str

# --- NODES ---

def research_node(state: AgentState):
    """Gather data first so both agents see the same reality."""
    ticker = state["ticker"]
    data = get_stock_data(ticker)
    news = get_news(ticker)
    return {"stock_data": data, "news_summary": news}

def bull_agent_node(state: AgentState):
    """The Optimist."""
    prompt = ChatPromptTemplate.from_template(
        """You are a Wall Street Bull Analyst. You specialize in finding growth stories and undervalued gems.
        Analyze the following data for {ticker} and write a compelling, optimistic investment thesis.
        Focus on upside potential, innovation, and market dominance. Minimize risks.
        
        Data: {data}
        News: {news}
        
        Keep it punchy (3 bullet points max)."""
    )
    chain = prompt | llm
    response = chain.invoke({"ticker": state["ticker"], "data": state["stock_data"], "news": state["news_summary"]})
    return {"bull_case": response.content}

def bear_agent_node(state: AgentState):
    """The Pessimist."""
    prompt = ChatPromptTemplate.from_template(
        """You are a Short Seller / Bear Analyst. You specialize in finding overhyped stocks and accounting risks.
        Analyze the following data for {ticker} and write a scathing, pessimistic investment thesis.
        Focus on overvaluation, slowing growth, and macro headwinds. Ignore the hype.
        
        Data: {data}
        News: {news}
        
        Keep it punchy (3 bullet points max)."""
    )
    chain = prompt | llm
    response = chain.invoke({"ticker": state["ticker"], "data": state["stock_data"], "news": state["news_summary"]})
    return {"bear_case": response.content}

def arbitrator_node(state: AgentState):
    """The Decision Maker."""
    prompt = ChatPromptTemplate.from_template(
        """You are the Chief Investment Officer. You have heard the Bull and Bear cases for {ticker}.
        
        Bull Case: {bull}
        Bear Case: {bear}
        Data: {data}
        
        Synthesize these conflicting views into a final verdict.
        Who makes the stronger argument? 
        Give a recommendation: BUY, SELL, or HOLD.
        """
    )
    chain = prompt | llm
    response = chain.invoke({
        "ticker": state["ticker"], 
        "bull": state["bull_case"], 
        "bear": state["bear_case"],
        "data": state["stock_data"]
    })
    return {"final_verdict": response.content}

# --- GRAPH ---
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("research", research_node)
workflow.add_node("bull", bull_agent_node)
workflow.add_node("bear", bear_agent_node)
workflow.add_node("arbitrator", arbitrator_node)

# Set Edges
workflow.set_entry_point("research")
# Fan Out: Research -> Bull AND Bear (Parallel)
workflow.add_edge("research", "bull")
workflow.add_edge("research", "bear")
# Fan In: Both -> Arbitrator
workflow.add_edge("bull", "arbitrator")
workflow.add_edge("bear", "arbitrator")
workflow.add_edge("arbitrator", END)

app = workflow.compile()

# --- UI ---
st.set_page_config(page_title="Bear vs Bull", layout="wide")
st.title("🐂 Bear vs. Bull Investment Researcher")
st.markdown("Two AI agents debate a stock. One makes the decision.")

ticker = st.text_input("Enter Ticker Symbol (e.g., AAPL, TSLA, GLD)", "AAPL").upper()

if st.button("Start Debate"):
    with st.spinner("Gathering data and starting agents..."):
        initial_state = {"ticker": ticker}
        result = app.invoke(initial_state)
        
        # Display Data
        st.subheader(f"Financial Snapshot: {ticker}")
        metrics = result['stock_data']
        
        if "error" in metrics:
            st.error(f"Error fetching data: {metrics['error']}")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Price", f"${metrics.get('current_price', 'N/A')}")
            c2.metric("P/E Ratio", metrics.get('pe_ratio', 'N/A'))
            c3.metric("Trend (1mo)", metrics.get('price_trend', 'N/A'))
            
            st.divider()
            
            # Display The Debate
            col1, col2 = st.columns(2)
            
            with col1:
                st.success("### 🐂 The Bull Case")
                st.markdown(result['bull_case'])
                
            with col2:
                st.error("### 🐻 The Bear Case")
                st.markdown(result['bear_case'])
                
            st.divider()
            
            # Final Verdict
            st.info("### ⚖️ CIO Verdict")
            st.markdown(result['final_verdict'])
