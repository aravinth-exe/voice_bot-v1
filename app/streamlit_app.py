import os
import sys
import streamlit as st

# 🔧 System Path Setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
ffmpeg_path = os.path.abspath("ffmpeg")
os.environ["PATH"] += os.pathsep + ffmpeg_path

# 🧠 Internal Imports
from app.audio_io import listen_once
from app.coqui_tts import speak_text
from app.assistant_chain import query_agent
from app.rag_ingest import ingest_docs

# 📄 Helper to list processed files
def list_uploaded_docs(folder="data/docs"):
    return [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))] if os.path.exists(folder) else []

# ⚙️ Page Setup
st.set_page_config(page_title="AI Voice Agent", layout="wide")
st.title("🎤 AI Voice Assistant")

# 📤 Upload Section
uploaded_files = st.file_uploader("📄 Upload your documents", type=["pdf", "docx", "txt"], accept_multiple_files=True)
if uploaded_files:
    os.makedirs("data/docs", exist_ok=True)
    for file in uploaded_files:
        save_path = os.path.join("data/docs", file.name)
        with open(save_path, "wb") as f:
            f.write(file.read())
    ingest_docs("data/docs")
    st.success("✅ Files ingested into the vector DB!")

# 📂 Display Uploaded Documents
st.markdown("### 📄 Processed Documents")
docs = list_uploaded_docs("data/docs")
if docs:
    for doc in docs:
        file_url = f"/data/docs/{doc}"  # Must be statically served
        st.markdown(f"✅ **{doc}** <a href='{file_url}' target='_blank'>🔗 Open</a>", unsafe_allow_html=True)
else:
    st.info("No documents uploaded yet.")

# 📞 Call Controls
if "call_active" not in st.session_state:
    st.session_state.call_active = False

col1, col2 = st.columns(2)
with col1:
    if not st.session_state.call_active and st.button("📞 Call Agent"):
        st.session_state.call_active = True
        st.rerun()
with col2:
    if st.session_state.call_active and st.button("🔴 End Call"):
        st.session_state.call_active = False
        st.rerun()

# 🎤 Voice Interaction (One turn per run)
if st.session_state.call_active:
    st.info("🎙️ Agent is listening... Say 'exit' or press 'End Call' to stop.")
    
    query = listen_once()

    if query.strip():
        st.write(f"🧑‍💬 You: {query}")
        
        if query.lower() in {"exit", "quit", "stop"}:
            speak_text("Goodbye!")
            st.session_state.call_active = False
            st.rerun()
        else:
            st.info("🤖 Thinking...")
            try:
                answer = query_agent(query)
                st.write(f"🤖 Assistant: {answer}")
                speak_text(answer)
                st.rerun()  # 🔁 Continue conversation
            except Exception as e:
                st.error(f"❌ Error: {e}")
    else:
        st.warning("⚠️ No speech detected. Please try again.")
        st.rerun()
