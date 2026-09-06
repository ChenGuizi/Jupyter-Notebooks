import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from tavily import TavilyClient
from langchain_core.tools import tool
from deepagents import create_deep_agent

load_dotenv()

if not os.getenv("GEMINI_API_KEY"):
    st.error("GEMINI_API_KEY is missing from .env")
    st.stop()

if not os.getenv("TAVILY_API_KEY"):
    st.error("TAVILY_API_KEY is missing from .env")
    st.stop()


tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


@tool
def tavily_search(query: str) -> str:
    """Search the web for current consumer credit risk information."""
    response = tavily_client.search(
        query=query,
        search_depth="advanced",
        max_results=5,
        include_answer=True,
    )

    results = []

    if response.get("answer"):
        results.append(f"Summary:\n{response['answer']}")

    for item in response.get("results", []):
        results.append(
            f"Title: {item.get('title', '')}\n"
            f"URL: {item.get('url', '')}\n"
            f"Content: {item.get('content', '')}"
        )

    return "\n\n".join(results) or "No search results found."


SKILL_DIR = Path(__file__).parent / "skills" / "consumer_credit_risk"

agent = create_deep_agent(
    model="google_genai:gemini-2.5-flash",
    tools=[tavily_search],
    skills=[str(SKILL_DIR.resolve())],
    system_prompt=(
        "You are a consumer credit risk analyst. "
        "Use the consumer_credit_risk skill and Tavily search. "
        "Cite source titles and URLs. "
        "Never make individual lending decisions. "
        "Your output is for qualified human review only."
    ),
)


st.set_page_config(
    page_title="Consumer Credit Risk Analyst",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Consumer Credit Risk Analyst")
st.caption("Research assistant powered by Deep Agents, Gemini, and Tavily")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Controls")

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

    st.info(
        "This tool provides analytical research only. "
        "It must not be used as the sole basis for an individual credit decision."
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input(
    "Ask about consumer credit risk, delinquency, rates, employment, or regulation..."
)

if prompt:
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Researching with Tavily..."):
            try:
                result = agent.invoke(
                    {"messages": st.session_state.messages}
                )

                answer = result["messages"][-1].content

                if isinstance(answer, list):
                    answer = "\n".join(
                        item.get("text", str(item))
                        if isinstance(item, dict)
                        else str(item)
                        for item in answer
                    )

                st.markdown(answer)

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

            except Exception as error:
                st.error(f"Agent error: {error}")