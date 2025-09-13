# app.py
import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage

# -------------------------------
# Load API keys safely
# -------------------------------
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY", os.getenv("TAVILY_API_KEY"))
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

if not (GROQ_API_KEY and TAVILY_API_KEY and OPENAI_API_KEY):
    st.error("❌ API keys missing! Add them in `.streamlit/secrets.toml` or as environment variables.")
    st.stop()

# -------------------------------
# Function: Get AI Response
# -------------------------------
def get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider):
    # Initialize LLM
    if provider == "Groq":
        llm = ChatGroq(model=llm_id, api_key=GROQ_API_KEY)
    elif provider == "OpenAI":
        llm = ChatOpenAI(model=llm_id, api_key=OPENAI_API_KEY)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    # Setup tools
    tools = [TavilySearch(api_key=TAVILY_API_KEY, max_results=2)] if allow_search else []

    # Create agent
    agent = create_react_agent(model=llm, tools=tools)

    # Prepare state
    state = {
        "messages": [
            ("system", system_prompt),
            ("user", query)
        ]
    }

    # Invoke agent
    response = agent.invoke(state)

    # Extract AI messages
    messages = response.get("messages", [])
    ai_messages = [m.content for m in messages if isinstance(m, AIMessage)]
    return ai_messages[-1] if ai_messages else "No AI response."

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="LangGraph Agent UI", layout="centered")
st.title("🤖 LangGraph AI Chatbot")
st.write("Interact with AI Agents powered by LangGraph, Groq, and OpenAI!")

# System prompt input
system_prompt = st.text_area(
    "📝 Define your AI Agent:",
    height=70,
    placeholder="Type your system prompt here..."
)

# Model selection
MODEL_NAMES_GROQ = ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"]
MODEL_NAMES_OPENAI = ["gpt-4o-mini"]

provider = st.radio("⚡ Select Provider:", ("Groq", "OpenAI"))

if provider == "Groq":
    selected_model = st.selectbox("Select Groq Model:", MODEL_NAMES_GROQ)
else:
    selected_model = st.selectbox("Select OpenAI Model:", MODEL_NAMES_OPENAI)

# Web search toggle
allow_web_search = st.checkbox("🌐 Allow Web Search")

# User query input
user_query = st.text_area(
    "💬 Enter your query:",
    height=150,
    placeholder="Ask anything!"
)

# Submit button
if st.button("🚀 Ask Agent!"):
    if user_query.strip():
        try:
            response = get_response_from_ai_agent(
                llm_id=selected_model,
                query=user_query,
                allow_search=allow_web_search,
                system_prompt=system_prompt,
                provider=provider
            )
            st.subheader("🤖 Agent Response")
            st.markdown(response)
        except Exception as e:
            st.error(f"❌ Agent error: {e}")
    else:
        st.warning("✋ Please enter a query before asking the agent.")
