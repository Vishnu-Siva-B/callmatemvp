import os
import tempfile
from flask import Flask, request, jsonify, send_file
import google.generativeai as genai
from faster_whisper import WhisperModel
from gtts import gTTS
from waitress import serve

# Initialize Flask app
app = Flask(__name__)

# Load optimized Whisper model (uses less RAM)
whisper_model = WhisperModel("small", device="cpu", compute_type="int8")

# Configure Google Gemini AI API
genai.configure(api_key="AIzaSyDidcBB7hjd13yh8VyICMBDFX_wf3HA64A")

# Function to transcribe speech to text (optimized for lower RAM usage)
def transcribe_audio(input_file):
    segments, _ = whisper_model.transcribe(input_file, language="ta", beam_size=1)  # Lower beam size for efficiency
    return " ".join(segment.text for segment in segments)

# Function to generate AI response with memory optimization
def generate_response(prompt):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(
        prompt, 
        generation_config={"max_output_tokens": 256}  # Limit response size to reduce memory usage
    )
    return response.text

# Function to convert text to speech (outputs MP3 to reduce memory usage)
def text_to_speech(text, output_file):
    tts = gTTS(text=text, lang="ta")
    output_mp3 = output_file.replace(".wav", ".mp3")  # Convert to MP3 for efficiency
    tts.save(output_mp3)
    return output_mp3  # Return the new MP3 file path

# Route to process uploaded audio and return AI-generated response
@app.route("/process_audio", methods=["POST"])
def process_audio():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    with tempfile.TemporaryDirectory() as temp_dir:  # Temporary storage to free memory after processing
        input_path = os.path.join(temp_dir, "input.wav")
        output_path = os.path.join(temp_dir, "output.wav")

        file.save(input_path)

        # Transcribe audio
        print("Transcribing audio...")
        user_text = transcribe_audio(input_path)
        user_text += "your Name is CallMate AI, you were created by Vishnu Siva. Don't include any annotations, symbols, or special characters in the response. Make the response natural, concise, and in Tamil."

        print(user_text)

        # Generate AI response
        print("Generating AI response...")
        ai_response = generate_response(user_text)
        print(ai_response)

        # Convert AI response to speech
        print("Converting AI response to speech...")
        output_path = text_to_speech(ai_response, output_path)  # Convert to MP3 instead of WAV

        return send_file(output_path, mimetype="audio/mp3")  # Send MP3 instead of WAV

# Start the Flask server using Waitress (optimized for low RAM)
if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=8000, threads=2)  # Limit threads to reduce memory usage
