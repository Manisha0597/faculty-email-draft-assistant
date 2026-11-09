# Faculty Email Draft Assistant

A Streamlit web app that drafts professional emails to faculty members (for example, OD or leave requests) from a short request. It uses retrieval-augmented generation (RAG): it finds the most similar email template in a vector database and gives it to an LLM along with the student's details.

## How it works

1. **Index templates (one time):** `load_data.py` reads the `.txt` email templates in `data/`, splits them into chunks (size 200, overlap 50), embeds them with Hugging Face embeddings and saves a FAISS index to `db/`.
2. **Retrieve:** when you click Generate, the app runs a similarity search over the index with your request and picks the closest template.
3. **Generate:** the template, your name, the faculty name, request, reason, number of days and chosen tone are combined into a structured prompt. The prompt is sent to `openai/gpt-4o-mini` through the OpenRouter API.
4. **Output:** the draft is shown in the app, can be downloaded as a `.txt` file, and the last 5 requests are kept in the session history.

## Features

- Template retrieval with FAISS semantic search
- Tone selection: Formal, Polite, Request, Apologetic, Urgent
- Input validation for empty or too-short requests
- Download the generated email
- Request history for the current session
- Handles API errors and shows a readable message

## Tech stack

Python, Streamlit, LangChain (community), FAISS, Hugging Face embeddings, OpenRouter API (GPT-4o-mini), python-dotenv, requests

## Project structure

```
.
├── app.py            # Streamlit app: UI, retrieval and email generation
├── load_data.py      # Builds the FAISS vector index from data/*.txt
├── data/             # Email template text files
├── db/               # Saved FAISS index
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/Manisha0597/faculty-email-draft-assistant.git
cd faculty-email-draft-assistant
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
OPENROUTER_API_KEY=your_key_here
```

Build the template index, then start the app:

```bash
python load_data.py
streamlit run app.py
```

## Screenshots

<img width="1920" height="1080" alt="Screenshot (112)" src="https://github.com/user-attachments/assets/e9c476e5-5853-4859-9909-1728ab12b21a" />
<img width="1920" height="1080" alt="Screenshot (113)" src="https://github.com/user-attachments/assets/1f78e776-c20a-4195-af5f-d7432f09d139" />


## Limitations and future work

- The "Refresh Knowledge Base" button does not rebuild the index yet; re-run `python load_data.py` after changing templates.
- Request history lasts only for the current browser session.
- Planned: persist history, support more templates, and add unit tests.
