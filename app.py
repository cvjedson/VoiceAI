import os
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from gtts import gTTS

st.set_page_config(page_title="Vimal's VoiceAI", page_icon=":microphone:", layout="centered")

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []


def generate_ai_response(prompt):
    model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Keep the full conversation context from this session and answer each new question based on the earlier discussion."}
    ]
    messages.extend(st.session_state.conversation_history)
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.7,
        max_tokens=300
    )

    ai_response = response.choices[0].message.content.strip()
    st.session_state.conversation_history.extend([
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": ai_response},
    ])
    return ai_response

def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    audio_bytes = BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    return audio_bytes

st.title("Vimal's VoiceAI")

st.write(
    "Welcome to Vimal's VoiceAI!\n"
    "Convert your voice into text with advanced speech recognition and receive AI-powered responses in both text and speech.\n"
    "Free Version: Responses are limited to 300 tokens, and AI voice generation may take a few moments. "
    "For the best results, please ask your questions clearly and simply."
)

for message in st.session_state.conversation_history:
    if message["role"] == "user":
        st.markdown(f"**You:** {message['content']}")
    else:
        st.markdown(f"**AI:** {message['content']}")

st.markdown("---")

audio_value = st.audio_input("Record your voice:", key="voice_input")

if audio_value is not None:
    st.audio(audio_value)

    if "last_processed_audio" not in st.session_state:
        st.session_state.last_processed_audio = None

    if st.session_state.last_processed_audio != audio_value:
        st.session_state.last_processed_audio = audio_value
        with st.spinner("Transcribing your voice..."):
            transcription = client.audio.transcriptions.create(
                file=audio_value,
                model="whisper-large-v3-turbo",
                response_format="json",
                temperature=0.0
            )

            transcribed_text = transcription.text.strip()
            st.subheader("you asked:")

            if transcribed_text:
                st.write(transcribed_text)
                ai_response = generate_ai_response(transcribed_text)
                st.subheader("AI Response:")
                st.write(ai_response)

                st.markdown("\n\nAsk your next question Please.")
                st.markdown("---")
                audio_bytes = text_to_speech(ai_response)
                st.audio(audio_bytes, format="audio/mp3", start_time=0)
                st.markdown("---")
                st.audio_input("Record your voice:", key="voice_input_next") 