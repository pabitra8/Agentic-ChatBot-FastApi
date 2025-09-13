#Step1: Setup API Keys for Groq, OpenAI and Tavily
import os
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage

GROQ_API_KEY=os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY=os.environ.get("TAVILY_API_KEY")
OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY")

def get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider):
    # Select LLM
    if provider=="Groq":
        llm=ChatGroq(model=llm_id)
    elif provider=="OpenAI":
        llm=ChatOpenAI(model=llm_id)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    # Optional tool
    tools=[TavilySearch(max_results=2)] if allow_search else []

    # Create agent (⚠️ no state_modifier here)
    agent=create_react_agent(
        model=llm,
        tools=tools
    )

    # Pass system + user messages explicitly
    state={
        "messages": [
            ("system", system_prompt),
            ("user", query)
        ]
    }

    response=agent.invoke(state)
    messages=response.get("messages", [])
    ai_messages=[m.content for m in messages if isinstance(m, AIMessage)]
    return ai_messages[-1] if ai_messages else "No AI response."
