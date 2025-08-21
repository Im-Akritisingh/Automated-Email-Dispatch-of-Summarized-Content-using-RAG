import streamlit as st
from dotenv import load_dotenv
import os
import tempfile
from datetime import datetime, timedelta, time
import threading
import pytz   # ✅ Added
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
import smtplib
from email.mime.text import MIMEText
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"


# Load environment variables
load_dotenv()
EMAIL_ADDRESS = os.getenv("EMAIL_USER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASS")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="Smart Email Summarizer", layout="centered")
st.title("📩 Auto Email Summarizer")

# Sidebar inputs
st.sidebar.header("📨 Email Settings")
user_email = st.sidebar.text_input("Receiver Email", value="example@gmail.com")
send_time = st.sidebar.time_input("⏰ Send Summary At")

# Upload section
st.subheader("📄 Upload a file or paste text")
uploaded_file = st.file_uploader("Upload PDF or TXT file", type=["pdf", "txt"])
text_input = st.text_area("Or paste your content here")

def summarize_and_send(file_bytes, file_name, pasted_text, email_to):
    try:
        # Write to temp file
        if file_bytes:
            suffix = file_name.split('.')[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            loader = PyPDFLoader(tmp_path) if suffix == "pdf" else TextLoader(tmp_path)
        elif pasted_text.strip():
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as tmp:
                tmp.write(pasted_text)
                tmp_path = tmp.name
            loader = TextLoader(tmp_path)
        else:
            print("No content to summarize.")
            return

        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(chunks, embedding)

        llm = ChatGroq(api_key=GROQ_API_KEY, model_name="llama3-70b-8192")
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
        response = qa_chain.invoke("Summarize this document for email update.")
        summary = response["result"] if isinstance(response, dict) else str(response)

        msg = MIMEText(summary)

        msg["Subject"] = "Summary Update"
        msg["From"] = EMAIL_ADDRESS
        msg["To"] = email_to

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)

        print(f"Email sent to {email_to}")
    except Exception as e:
        print(f" Email failed: {e}")

def schedule_email_once(file_bytes, file_name, pasted_text, email_to):
    ist = pytz.timezone("Asia/Kolkata")  # ✅ Force IST timezone
    now = datetime.now(ist)
    target = datetime.combine(now.date(), send_time)
    target = ist.localize(target)  # ✅ Localize target to IST

    if target < now:
        target += timedelta(days=1)

    delay = (target - now).total_seconds()

    def run_task():
        print(f" Waiting {int(delay)} seconds...")
        threading.Event().wait(delay)
        summarize_and_send(file_bytes, file_name, pasted_text, email_to)

    threading.Thread(target=run_task, daemon=True).start()
    st.info(f" Email scheduled in {int(delay // 60)} min {int(delay % 60)} sec (IST)")

# Trigger
if st.button(" Schedule Email"):
    if not (uploaded_file or text_input.strip()):
        st.warning("Please upload a file or paste content.")
    elif not user_email:
        st.warning("Please enter a valid email.")
    else:
        file_bytes = uploaded_file.read() if uploaded_file else None
        file_name = uploaded_file.name if uploaded_file else None
        pasted_text = text_input
        schedule_email_once(file_bytes, file_name, pasted_text, user_email)
        st.success(f" Email will be sent at {send_time.strftime('%H:%M')} IST to {user_email}")
