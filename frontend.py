import streamlit as st
import requests

st.set_page_config(page_title="LangGraph Agent UI", layout="centered")
st.title("🤖 AI Chatbot Agents")
st.write("Create and interact with LangGraph-powered AI Agents!")

# -----------------------
# Agent Setup UI
# -----------------------
system_prompt = st.text_area(
    "📝 Define your AI Agent:",
    height=70,
    placeholder="Type your system prompt here..."
)

MODEL_NAMES_GROQ = ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"]
MODEL_NAMES_OPENAI = ["gpt-4o-mini"]

provider = st.radio("⚡ Select Provider:", ("Groq", "OpenAI"))

if provider == "Groq":
    selected_model = st.selectbox("Select Groq Model:", MODEL_NAMES_GROQ)
elif provider == "OpenAI":
    selected_model = st.selectbox("Select OpenAI Model:", MODEL_NAMES_OPENAI)

allow_web_search = st.checkbox("🌐 Allow Web Search")

user_query = st.text_area(
    "💬 Enter your query:",
    height=150,
    placeholder="Ask anything!"
)

# -----------------------
# API Config
# -----------------------
API_URL = "http://127.0.0.1:8000/chat"

# -----------------------
# Submit Button
# -----------------------
if st.button("🚀 Ask Agent!"):
    if user_query.strip():
        payload = {
            "model_name": selected_model,
            "model_provider": provider,
            "system_prompt": system_prompt,
            "messages": [user_query],
            "allow_search": allow_web_search
        }

        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                response_data = response.json()
                if "error" in response_data:
                    st.error(response_data["error"])
                elif "response" in response_data:
                    st.subheader("🤖 Agent Response")
                    st.markdown(f"{response_data['response']}")
                else:
                    st.warning("⚠️ No valid response received from backend.")
            else:
                st.error(f"⚠️ Backend error: {response.status_code}")
        except Exception as e:
            st.error(f"❌ Failed to connect to backend: {e}")
    else:
        st.warning("✋ Please enter a query before asking the agent.")
