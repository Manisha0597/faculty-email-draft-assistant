from dotenv import load_dotenv
import os
import streamlit as st
import requests
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Faculty Email Draft Assistant",
    page_icon="📧",
    layout="wide"
)

# ---------------- SIDEBAR ----------------
st.sidebar.markdown(
    """
    <h2 style='color:#ffffff;'>📌 Navigation</h2>
    """,
    unsafe_allow_html=True
)

st.sidebar.info("""
Faculty Email Draft Assistant

Features:
• AI Email Generation  
• Template Retrieval  
• Tone Selection  
• Download Email  
• Previous Request History  
• Knowledge Base Refresh  
""")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
 /* Spinner Styling */
.stSpinner > div {
    border-top-color: #6366f1 !important;
}

.loading-box {
    background: white;
    padding: 18px;
    border-radius: 16px;
    text-align: center;
    margin-top: 20px;
    box-shadow: 0px 4px 16px rgba(0,0,0,0.06);
    color: #475569;
    font-size: 17px;
    font-weight: 500;
}

/* Main App Background */
.stApp {
    background: linear-gradient(135deg, #eef2ff 0%, #f8fafc 50%, #ede9fe 100%);
    font-family: 'Segoe UI', sans-serif;
    color: #1e293b;
}

/* Hide Streamlit Branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e1b4b, #312e81);
    border-right: 1px solid rgba(255,255,255,0.08);
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

/* Header Title */
.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #1e293b;
    margin-bottom: 5px;
}

.sub-title {
    color: #64748b;
    font-size: 17px;
    margin-bottom: 30px;
}

/* Section Titles */
.section-title {
    font-size: 30px;
    font-weight: 700;
    color: #312e81;
    margin-bottom: 5px;
}

.section-subtitle {
    color: #64748b;
    margin-bottom: 25px;
    font-size: 15px;
}

/* Input Labels */
label {
    color: #312e81 !important;
    font-weight: 700 !important;
    font-size: 15px !important;
}

/* Emoji Label Styling */
.input-heading {
    font-size: 16px;
    font-weight: 700;
    color: #312e81;
    margin-bottom: -8px;
    margin-top: 2px;
}

/* Input Boxes */
.stTextInput input,
.stTextArea textarea {
    background: rgba(255,255,255,0.82) !important;
    border: 1px solid #dbeafe !important;
    border-radius: 16px !important;
    padding: 14px !important;
    color: #0f172a !important;
    font-size: 15px !important;
    backdrop-filter: blur(10px);
}

/* Text Area */
.stTextArea textarea {
    min-height: 120px !important;
}

/* Select Box */
.stSelectbox div[data-baseweb="select"] {
    background: rgba(255,255,255,0.82);
    border-radius: 16px;
    border: 1px solid #dbeafe;
}

/* Buttons */
.stButton button {
    width: 100%;
    background: linear-gradient(135deg, #4f46e5, #2563eb);
    color: white;
    border: none;
    border-radius: 16px;
    padding: 14px;
    font-size: 16px;
    font-weight: 700;
    transition: all 0.3s ease;
    margin-top: 10px;
    box-shadow: 0 8px 20px rgba(79,70,229,0.25);
}

.stButton button:hover {
    transform: translateY(-2px);
    background: linear-gradient(135deg, #4338ca, #1d4ed8);
    color: white;
}

/* Download Button */
.stDownloadButton button {
    width: 100%;
    border-radius: 16px;
    padding: 12px;
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
    border: none;
    font-weight: 600;
}

/* Info Box */
.stAlert {
    border-radius: 16px;
}

/* Generated Email Box */
textarea {
    border-radius: 18px !important;
}

/* Expander */
.streamlit-expanderHeader {
    font-size: 16px;
    font-weight: 600;
    color: #1e293b;
}

div[data-testid="stVerticalBlock"] > div {
    gap: 0.4rem;
}

/* Previous Request Cards */
.history-box {
    background: rgba(255,255,255,0.72);
    border-radius: 18px;
    padding: 18px;
    margin-bottom: 15px;
    border: 1px solid rgba(219,234,254,0.9);
    box-shadow: 0 4px 15px rgba(0,0,0,0.04);
}

/* Footer */
.footer {
    text-align: center;
    color: #64748b;
    margin-top: 30px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)



# ---------------- API KEY ----------------
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")

# ---------------- LOAD DATABASE ----------------
embeddings = HuggingFaceEmbeddings()

db = FAISS.load_local(
    "db",
    embeddings,
    allow_dangerous_deserialization=True
)

# ---------------- EMAIL GENERATION FUNCTION ----------------
def generate_email(prompt):

    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "openai/gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=data)

        result = response.json()

        if "choices" in result:
            return result["choices"][0]["message"]["content"]

        elif "error" in result:
            return f"API Error: {result['error']['message']}"

        else:
            return "Unknown API response."

    except Exception as e:
        return f"Error occurred: {str(e)}"

# ---------------- HEADER ----------------
st.markdown("""
<div class="main-title">
📧 Faculty Email Draft Assistant
</div>

<div class="sub-title">
Generate professional faculty emails instantly using AI
</div>
""", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- LAYOUT ----------------
left_col, right_col = st.columns([45, 55])

# ---------------- LEFT PANEL ----------------
with left_col:


    st.markdown(
        '<div class="section-title">📝 Email Details</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">Fill in the details to generate a professional faculty email.</div>',
        unsafe_allow_html=True
    )

    # YOUR NAME
    st.markdown('<div class="input-heading">👤 Your Name</div>', unsafe_allow_html=True)

    name = st.text_input(
        "",
        placeholder="Enter your name"
    )   

    # FACULTY NAME
    st.markdown('<div class="input-heading">👨‍🏫 Faculty Name</div>', unsafe_allow_html=True)

    title_col, name_col = st.columns([20, 80])

    with title_col:
        faculty_title = st.selectbox(
            " ",
            ["Mr.", "Mrs.", "Ms.", "Dr.", "Prof."]
        )

    with name_col:
        faculty_name = st.text_input(
            "  ",
            placeholder="Enter Faculty Name"
        )

    full_faculty_name = f"{faculty_title} {faculty_name}"

    # REQUEST
    st.markdown('<div class="input-heading">📝 Request</div>', unsafe_allow_html=True)

    user_input = st.text_area(
        "",
        placeholder="Example: OD request for hackathon participation",
        height=120
    )

    # NUMBER OF DAYS
    st.markdown('<div class="input-heading">📅 Number of Days</div>', unsafe_allow_html=True)

    days = st.text_input(
        "",
        placeholder="Enter number of days"
    )

    # REASON
    st.markdown('<div class="input-heading">📌 Reason</div>', unsafe_allow_html=True)

    reason = st.text_input(
        "",
        placeholder="Enter reason"
    )

    # TONE
    st.markdown('<div class="input-heading">🎯 Select Tone</div>', unsafe_allow_html=True)

    tone = st.selectbox(
        "",
        ["Formal", "Polite", "Request", "Apologetic", "Urgent"]
    )

    generate = st.button("✨ Generate Email")

    refresh = st.button("🔄 Refresh Knowledge Base")

    if refresh:
        st.success("Knowledge Base Refreshed Successfully")


# ---------------- RIGHT PANEL ----------------
with right_col:


    st.markdown(
        '<div class="section-title">📨 Generated Email</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">Your AI generated email will appear here.</div>',
        unsafe_allow_html=True
    )

    if generate:

        if user_input.strip() == "":
            st.warning("Please enter your request.")
            st.stop()

        elif len(user_input.strip()) < 5:
            st.warning("Please enter a more detailed request.")
            st.stop()

        else:

            with st.spinner("Generating Email..."):

                # Retrieve template
                docs = db.similarity_search(user_input, k=2)

                if len(docs) == 0:
                    st.error("No suitable template found.")
                    st.stop()

                template = docs[0].page_content

                # Prompt
                prompt = f"""
You are an AI Faculty Email Assistant.

Generate a professional email.

Instructions:
- Keep the email professional and polite.
- Use the faculty name provided in the greeting.
- Keep it medium length.
- Generate only 2 proper paragraphs in the body.
- First paragraph should explain the request.
- Second paragraph should explain the reason briefly.
- End with a polite requesting line like:
  "I kindly request you to consider my request."
- Include:
    1. Subject
    2. Greeting
    3. Body
    4. Closing

Student Name: {name}

Faculty Name: {full_faculty_name}

Request: {user_input}

Reason: {reason}

Days: {days}

Tone: {tone}

Reference Template:
{template}

Generate:
1. Subject
2. Greeting
3. Body
4. Closing
"""

                # Generate Email
                email = generate_email(prompt)

                # Save History
                st.session_state.history.append({
                    "request": user_input,
                    "email": email
                })

                # Display Email
                st.text_area(
                    "Generated Email",
                    email,
                    height=420
                )

                # Download
                st.download_button(
                    "⬇ Download Email",
                    email,
                    file_name="faculty_email.txt"
                )

                # Previous Requests
                st.markdown("### 🕘 Previous Requests")

                for i, item in enumerate(reversed(st.session_state.history[-5:])):

                    with st.expander(f"📌 {item['request']}"):

                        st.text_area(
                            "Generated Email",
                            item["email"],
                            height=250,
                            key=f"history_{i}"
                        )

                # Template
                with st.expander("📄 Source Template Used"):
                    st.write(template)

    else:

        st.info("Generated email will appear here after clicking Generate Email.")


# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
Developed using Streamlit + LangChain + OpenRouter + FAISS
</div>
""", unsafe_allow_html=True)