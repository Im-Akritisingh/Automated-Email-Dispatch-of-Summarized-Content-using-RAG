import streamlit as st
import smtplib, ssl

st.title("📧 Email Tester")

# Read secrets
try:
    email_user = st.secrets["EMAIL_USER"]
    email_pass = st.secrets["EMAIL_PASS"]
    st.success("✅ Secrets loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load secrets: {e}")
    st.stop()

receiver = st.text_input("Receiver Email", "test@example.com")

if st.button("Send Test Mail"):
    try:
        msg = "Subject: Streamlit Test Mail\n\nThis is a test mail from Streamlit Cloud."
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(email_user, email_pass)
            server.sendmail(email_user, receiver, msg)
        st.success(f"✅ Mail sent to {receiver}")
    except Exception as e:
        st.error(f"❌ Error: {e}")
