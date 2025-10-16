# Learning Website App

## Project Overview
A Flask web application that helps users learn more efficiently by offering:
- AI-powered chat with a Gen Z-style tutor,
- PDF and YouTube summarization,
- Custom flashcard generation,
- Secure user authentication and histories.

The app uses Groq LLM (Llama-3) and LangChain for language understanding, with all data securely stored in an SQLite database.

## Core Features

- **User Authentication:** Sign up, log in, log out securely.
- **AI Tutor Chat:** Interactive chat driven by large language model, keeps explanations clear and fun.
- **PDF Summarizer:** Upload a PDF to get an AI-generated summary.
- **YouTube Summarizer:** Paste a YouTube URL and receive a concise text summary using subtitles/transcripts.
- **Flashcard Generator:** Create custom flashcards from any text for study and memory aid.
- **Personalized Data:** Each user's activity, histories, and resources are private and persisted (in SQLite).

## Workflow

1. **User Management**
    - Registration and secure password hashing
    - Login/logout/session handling

2. **Learning Tools**
    - **Chat:** LLM-powered answers, chat history saved per user
    - **PDF Summarization:** Extracts and summarizes large documents
    - **YouTube Summarization:** Extracts video transcript, summarizes core points
    - **Flashcards:** Converts user text into JSON-formatted Q&A flashcards

3. **Database**
    - Users, chat history, flashcards, and summaries all stored in `learning_app.db`

## Results

- **Intuitive, all-in-one learning dashboard**: Chat, summarize, and quiz yourself all through a single login.
- **Modern Gen Z experience:** Engaging, simplified, AI-powered.
- **Handles various file/text inputs** with robust error handling and data privacy.

***

*This app is your AI-powered study partner, designed for modern learners. All results and summaries are generated and stored securely, with easy extension for new learning features.*

***

