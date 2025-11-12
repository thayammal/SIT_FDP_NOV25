from google.adk.agents import Agent

# root_agent = Agent(
#     name="my_voice_agent",
#     model="gemini-2.0-flash-live-001",#"gemini-2.0-flash-001",#"gemini-2.5-flash",   # choose appropriate model
#     description="Voice assistant for students at Deep Dive Technologies",
#     instruction=(
#         "You are a helpful voice assistant for AI/ML students. "
#         "Listen to user queries, respond clearly, and optionally ask clarifying questions."
#     ),
#     tools=[]  # If you have tools, list them here
# )


# #app1
# from google.adk.agents import Agent

# root_agent = Agent(
#     name="ai_news_agent_simple",
#     model="gemini-2.0-flash-live-001", # Essential for live voice interaction
#     instruction="You are an AI News Assistant.",
# )



# #app2
# #from google.adk.agents import Agent
# from google.adk.tools import google_search

# root_agent = Agent(
#     name="ai_news_agent_simple",
#     model="gemini-2.0-flash-live-001", # Essential for live voice interaction
#     instruction="You are an AI News Assistant. Use Google Search to find recent AI news.",
#     tools=[google_search]
# )


# #app3
# from google.adk.agents import Agent
# from google.adk.tools import google_search

# root_agent = Agent(
#     name="ai_news_agent_strict",
#     model="gemini-2.0-flash-live-001",
#     instruction="""
#     **Your Core Identity and Sole Purpose:**
#     You are a specialized AI News Assistant. Your sole and exclusive purpose is to find and summarize recent news (from the last few weeks) about Artificial Intelligence.

#     **Strict Refusal Mandate:**
#     If a user asks about ANY topic that is not recent AI news, you MUST refuse.
#     For off-topic requests, respond with the exact phrase: "Sorry, I can't answer anything about this. I am only supposed to answer about the latest AI news."

#     **Required Workflow for Valid Requests:**
#     1. You MUST use the `google_search` tool to find information.
#     2. You MUST base your answer strictly on the search results.
#     3. You MUST cite your sources.
#     """,
#     tools=[google_search]
# )




#app4
from typing import Dict, List
from google.adk.agents import Agent
from google.adk.tools import google_search
import yfinance as yf

def get_financial_context(tickers: List[str]) -> Dict[str, str]:
    """
    Fetches the current stock price and daily change for a list of stock tickers
    using the yfinance library.

    Args:
        tickers: A list of stock market tickers (e.g., ["GOOG", "NVDA"]).

    Returns:
        A dictionary mapping each ticker to its formatted financial data string.
    """
    financial_data: Dict[str, str] = {}
    for ticker_symbol in tickers:
        try:
            # Create a Ticker object
            stock = yf.Ticker(ticker_symbol)

            # Fetch the info dictionary
            info = stock.info

            # Safely access the required data points
            price = info.get("currentPrice") or info.get("regularMarketPrice")
            change_percent = info.get("regularMarketChangePercent")

            if price is not None and change_percent is not None:
                # Format the percentage and the final string
                change_str = f"{change_percent * 100:+.2f}%"
                financial_data[ticker_symbol] = f"${price:.2f} ({change_str})"
            else:
                # Handle cases where the ticker is valid but data is missing
                financial_data[ticker_symbol] = "Price data not available."

        except Exception:
            # This handles invalid tickers or other yfinance errors gracefully
            financial_data[ticker_symbol] = "Invalid Ticker or Data Error"

    return financial_data



root_agent = Agent(
    name="ai_news_chat_assistant",
    model="gemini-2.0-flash-live-001",
    instruction="""
    You are an AI News Analyst specializing in recent AI news about US-listed companies. Your primary goal is to be interactive and transparent about your information sources.

    **Your Workflow:**

    1.  **Clarify First:** If the user makes a general request for news (e.g., "give me AI news"), your very first response MUST be to ask for more details.
        *   **Your Response:** "Sure, I can do that. How many news items would you like me to find?"
        *   Wait for their answer before doing anything else.

    2.  **Search and Enrich:** Once the user specifies a number, perform the following steps:
        *   Use the `google_search` tool to find the requested number of recent AI news articles.
        *   For each article, identify the US-listed company and its stock ticker.
        *   Use the `get_financial_context` tool to retrieve the stock data for the identified tickers.

    3.  **Present Headlines with Citations:** Display the findings as a concise, numbered list. You MUST cite your tools.
        *   **Start with:** "Using `google_search` for news and `get_financial_context` (via yfinance) for market data, here are the top headlines:"
        *   **Format:**
            1.  [Headline 1] - [Company Stock Info]
            2.  [Headline 2] - [Company Stock Info]

    4.  **Engage and Wait:** After presenting the headlines, prompt the user for the next step.
        *   **Your Response:** "Which of these are you interested in? Or should I search for more?"

    5.  **Discuss One Topic:** If the user picks a headline, provide a more detailed summary for **only that single item**. Then, re-engage the user.

    **Strict Rules:**
    *   **Stay on Topic:** You ONLY discuss AI news related to US-listed companies. If asked anything else, politely state your purpose: "I can only provide recent AI news for US-listed companies."
    *   **Short Turns:** Keep your responses brief and always hand the conversation back to the user. Avoid long monologues.
    *   **Cite Your Tools:** Always mention `google_search` when presenting news and `get_financial_context` when presenting financial data.
    """,
    tools=[google_search, get_financial_context],
)