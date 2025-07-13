#the imports
from flask import Flask, render_template, request, redirect, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
from groq import Groq
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document
import sqlite3
import os
import json
import PyPDF2
from io import BytesIO
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from urllib.parse import urlparse, parse_qs

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
secret_key = os.getenv("SECRET_KEY", "supersecretkey")

if not groq_api_key:
    raise RuntimeError("Missing GROQ_API_KEY in environment.")

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = secret_key

#groq client for chat
groq_client = Groq(api_key=groq_api_key)

#add LangChain Groq client for summarization
langchain_llm = ChatGroq(model="llama3-70b-8192", groq_api_key=groq_api_key)

#system prompt for chat
system_prompt = {
    "role": "system",
    "content": (
        "You're a Gen Z tutor who explains things clearly and makes learning fun and snappy. "
        "Keep it simple, complete, and never miss key info."
        "Be extremely smart and well informed and always get things right and relevant"
    )
}

#summarization prompt template
summarization_prompt_template = """
Provide a comprehensive summary of the following YouTube video content. 
Include the main topics, key points, and important takeaways.
Structure the summary with clear sections if the content covers multiple topics:

Content: {text}

Summary:
"""

summarization_prompt = PromptTemplate(
    template=summarization_prompt_template,
    input_variables=["text"]
)

DB_PATH = "learning_app.db"

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            prompt TEXT,
            response TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id))''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS flashcards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            question TEXT,
            answer TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id))''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            type TEXT,
            content TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id))''')
        conn.commit()

def extract_pdf_text(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"PDF extraction error: {e}")
        return None
    
def extract_video_id(url):
    if "youtu.be" in url:
        return url.split("/")[-1]
    elif "youtube.com" in url:
        parsed_url = urlparse(url)
        video_id = parse_qs(parsed_url.query).get("v")
        if video_id:
            return video_id[0]
    return None

def get_youtube_transcript(video_url):
    try:
        video_id = extract_video_id(video_url)
        if not video_id:
            return None, "Invalid YouTube URL"

        print(f"[DEBUG] Extracted video ID: {video_id}")

        # fetch the transcript
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        full_text = " ".join([entry['text'] for entry in transcript_list])

        if not full_text.strip():
            return None, "Transcript is empty"

        return full_text, f"Transcript from video ID: {video_id}"

    except TranscriptsDisabled:
        return None, "Transcripts are disabled for this video"
    except NoTranscriptFound:
        return None, "No transcript found for this video"
    except Exception as e:
        print(f"[Transcript Fetch Error] {e}")
        return None, "Something went wrong fetching the transcript"


def summarize_youtube_content(content_text):
    try:
        chain = load_summarize_chain(langchain_llm, chain_type="stuff", prompt=summarization_prompt)
        docs = [Document(page_content=content_text)]
        return chain.run(docs)
    except Exception as e:
        print(f"Summarization error: {e}")
        return None


def summarize_text(text, max_length=2000):
    try:
        # Split text into chunks if too long
        if len(text) > max_length:
            words = text.split()
            chunks = []
            chunk_size = max_length // 4  # Smaller chunks for processing
            
            for i in range(0, len(words), chunk_size):
                chunk = ' '.join(words[i:i + chunk_size])
                chunks.append(chunk)
            
            # Summarize each chunk
            summaries = []
            for chunk in chunks:
                prompt = f"Summarize this text concisely:\n\n{chunk}"
                response = groq_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that creates concise summaries."},
                        {"role": "user", "content": prompt}
                    ],
                    model="llama3-70b-8192",
                    temperature=0.3
                )
                summaries.append(response.choices[0].message.content)
            
            # Final summary of all chunks
            combined_summaries = "\n\n".join(summaries)
            final_prompt = f"Create a comprehensive summary from these partial summaries:\n\n{combined_summaries}"
            final_response = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates comprehensive summaries."},
                    {"role": "user", "content": final_prompt}
                ],
                model="llama3-70b-8192",
                temperature=0.3
            )
            return final_response.choices[0].message.content
        else:
            # Direct summarization for shorter text
            prompt = f"Summarize this text concisely:\n\n{text}"
            response = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates concise summaries."},
                    {"role": "user", "content": prompt}
                ],
                model="llama3-70b-8192",
                temperature=0.3
            )
            return response.choices[0].message.content
    except Exception as e:
        print(f"Summarization error: {e}")
        return None

init_db()

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("SELECT id, password FROM users WHERE username=?", (username,))
            user = c.fetchone()
            if user and check_password_hash(user[1], password):
                session["user_id"] = user[0]
                session["username"] = username
                session['current_chat'] = []
                return redirect('/')
            else:
                flash("Invalid login credentials", "error")
    return render_template("login.html")

@app.route('/signup', methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        password = generate_password_hash(request.form["password"])
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            try:
                c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
                conn.commit()
                return redirect("/login")
            except sqlite3.IntegrityError:
                flash("Username already exists!", "error")
    return render_template("signup.html")

@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out!", "info")
    return redirect("/")

@app.route('/chat', methods=["GET", "POST"])
def chat():
    if "user_id" not in session:
        flash("Please log in first", "info")
        return redirect("/login")

    if "current_chat" not in session:
        session["current_chat"] = []

    if request.method == "POST":
        prompt = request.form.get("prompt", "").strip()
        if not prompt:
            flash("Enter a prompt to chat!", "error")
            return redirect("/chat")

        messages = [system_prompt]

        # Add up to 10 past exchanges
        recent_chat = session["current_chat"][-10:]
        for item in recent_chat:
            messages.append({"role": "user", "content": item["prompt"]})
            messages.append({"role": "assistant", "content": item["reply"]})

        messages.append({"role": "user", "content": prompt})

        try:
            response = groq_client.chat.completions.create(
                messages=messages,
                model="llama3-70b-8192",
                temperature=0.7,
                max_tokens=1000
            )
            reply = response.choices[0].message.content.strip()

            # Append to session chat
            session["current_chat"].append({"prompt": prompt, "reply": reply})
            session.modified = True

            # Save to DB
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute(
                    "INSERT INTO chat_history (user_id, prompt, response) VALUES (?, ?, ?)",
                    (session["user_id"], prompt, reply)
                )
                conn.commit()

        except Exception as e:
            flash("Error generating response. Please try again.", "error")
            print("Chat error:", e)

    return render_template("chat.html", chat=session["current_chat"])

@app.route('/clear_chat')
def clear_chat():
    if "user_id" in session:
        session["current_chat"] = []
        session.modified = True
        flash("Chat cleared!", "info")
    return redirect("/chat")

@app.route('/pdf', methods=["GET", "POST"])
def pdf():
    if "user_id" not in session:
        flash("Login first!", "info")
        return redirect("/login")

    summary = None
    if request.method == "POST":
        pdf_file = request.files.get("pdf_file")
        if pdf_file and pdf_file.filename.endswith('.pdf'):
            try:
                # Extract text from PDF
                text = extract_pdf_text(pdf_file)
                if text:
                    # Summarize the text
                    summary = summarize_text(text)
                    if summary:
                        # Save to database
                        with sqlite3.connect(DB_PATH) as conn:
                            conn.execute("INSERT INTO summaries (user_id, type, content) VALUES (?, 'pdf', ?)",
                                         (session["user_id"], summary))
                            conn.commit()
                    else:
                        flash("Failed to generate summary", "error")
                else:
                    flash("Failed to extract text from PDF", "error")
            except Exception as e:
                flash(f"PDF processing failed: {str(e)}", "error")
                print(f"PDF error: {e}")
        else:
            flash("Please upload a valid PDF file", "error")

    return render_template("pdf.html", summary=summary)

@app.route('/url', methods=["GET", "POST"])
def url():
    if "user_id" not in session:
        flash("Login first!", "info")
        return redirect("/login")

    summary = None
    video_title = None
    transcript_info = None

    if request.method == "POST":
        youtube_url = request.form.get("youtube_url", "").strip()
        if not youtube_url:
            flash("Please enter a YouTube URL", "error")
            return render_template("url.html")

        transcript_text, transcript_info = get_youtube_transcript(youtube_url)

        if transcript_text:
            video_title = transcript_info or "YouTube Video Summary"

            try:
                summary = summarize_youtube_content(transcript_text)

                if summary:
                    # Save to DB
                    full_summary = f"{transcript_info}\n\n{summary}"
                    with sqlite3.connect(DB_PATH) as conn:
                        conn.execute(
                            "INSERT INTO summaries (user_id, type, content) VALUES (?, 'youtube', ?)",
                            (session["user_id"], full_summary)
                        )
                        conn.commit()

                    flash("Summary generated successfully!", "success")
                else:
                    flash("Summary generation failed.", "error")

            except Exception as e:
                print("Summarization failed:", e)
                flash("Error summarizing YouTube video.", "error")
        else:
            flash(f"Could not fetch transcript: {transcript_info}", "error")
            flash("Tip: Try videos with subtitles/captions.", "info")

    return render_template("url.html", summary=summary, video_title=video_title, transcript_info=transcript_info)


@app.route('/flashcards', methods=["GET", "POST"])
def flashcards():
    if "user_id" not in session:
        flash("Login first!", "info")
        return redirect("/login")

    flashcards = []
    if request.method == "POST":
        input_text = request.form.get("input_text", "").strip()
        if not input_text:
            flash("Please enter some text to generate flashcards", "error")
            return render_template("flashcards.html", flashcards=flashcards)
            
        try:
            no_of_cards = int(request.form.get("no_of_cards", 5))
            no_of_cards = max(1, min(no_of_cards, 20))  # Limit between 1-20
        except (ValueError, TypeError):
            no_of_cards = 5

        prompt = (
            f"Generate exactly {no_of_cards} flashcards based on the following content. "
            f"Return ONLY a valid JSON array in this exact format: "
            f'[{{"question": "question text here", "answer": "answer text here"}}]. '
            f"Do not include any explanations, markdown, or additional text. Just the JSON array:\n\n{input_text}"
        )

        try:
            response = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You generate flashcards in JSON format. Return only valid JSON arrays."},
                    {"role": "user", "content": prompt}
                ],
                model="llama3-70b-8192",
                temperature=0.5
            )
            
            reply = response.choices[0].message.content.strip()
            
            # Clean up the response to extract JSON
            if "```json" in reply:
                reply = reply.split("```json")[1].split("```")[0].strip()
            elif "```" in reply:
                reply = reply.split("```")[1].split("```")[0].strip()
            
            # Ensure it starts with [ and ends with ]
            if not reply.startswith("["):
                # Try to find JSON array in the response
                start_idx = reply.find("[")
                if start_idx != -1:
                    reply = reply[start_idx:]
            
            if not reply.endswith("]"):
                end_idx = reply.rfind("]")
                if end_idx != -1:
                    reply = reply[:end_idx + 1]
            
            # Parse JSON
            flashcards = json.loads(reply)
            
            # Validate the structure
            if not isinstance(flashcards, list):
                raise ValueError("Response is not a list")
                
            for fc in flashcards:
                if not isinstance(fc, dict) or "question" not in fc or "answer" not in fc:
                    raise ValueError("Invalid flashcard structure")
            
            # Save to database
            with sqlite3.connect(DB_PATH) as conn:
                for fc in flashcards:
                    conn.execute("INSERT INTO flashcards (user_id, question, answer) VALUES (?, ?, ?)",
                                (session["user_id"], fc['question'], fc['answer']))
                conn.commit()
                
        except json.JSONDecodeError as e:
            flash(f"Failed to parse flashcards response. Please try again.", "error")
            print(f"JSON error: {e}, Response: {reply}")
            flashcards = []
        except Exception as e:
            flash(f"Flashcard generation failed: {str(e)}", "error")
            print(f"Flashcard error: {e}")
            flashcards = []

    return render_template("flashcards.html", flashcards=flashcards)


if __name__ == "__main__":
    app.run(debug=True)