# 🚀 RAG-Based Video GPT — Convert Any Video Into an Intelligent Q&A System

This project transforms long videos into a fully searchable Video-GPT system using RAG (Retrieval Augmented Generation).
Ask any question about the video → Get the exact timestamp, video number, and a clear explanation retrieved from your processed dataset.

Built using:

🎙 Whisper for speech-to-text

🔍 Cosine Similarity + Embeddings for semantic search

🤖 Gemini / LLMs for refined answers

🧩 A clean multi-stage pipeline

This project is inspired by and partially based on the excellent Data Science Course by Code With Harry.
Full credit to Code With Harry for the foundational concepts in Whisper, embeddings, and project structure.


# ⚙️ Pipeline Overview

1️⃣ Extract audio from video

1_process_video.py

Cuts video → extracts audio → stores in /audios.

2️⃣ Split audio into segments

2_audio_to_segments.py

Whisper transcribes → saves timestamps + text into JSON.

3️⃣ Create embeddings

3_embeddings.py

Converts each transcript segment to vector embeddings.

4️⃣ Run cosine similarity search

4_model_output.py

User asks a question

System returns best matching timestamps + explanation.

5️⃣ Improve with Gemini / LLM

5_using_genai_for_better_output.py

LLM refines output

Produces clean "VideoGPT-style" response.

# 🧠 Example Output
Ask a question: what is machine learning? where it had thought

video_number = 1
start_time = 127.16
end_time   = 144.16
explanation = "Machine learning is a science of getting computers to learn without being explicitly programmed."

✨ Features

✔ Converts any video into a Q&A tool
✔ Searches via embeddings + cosine similarity
✔ Retrieves exact timestamps
✔ LLM-powered refined answers
✔ Well-structured modular pipeline
✔ Perfect for RAG-based learning apps, chatbot systems, and educational tools

# 🔥 Credits

Special thanks & full credit to:

Code With Harry — Data Science Master Course

This project is heavily inspired by the techniques, ideas, and structure taught in the course.
Some scripts and logic are derived or adapted from the course lessons.


# 👨‍💻 Author

Krushna Palekar
