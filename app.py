import os
from flask import Flask, render_template, request, jsonify
import whisper
from pydub import AudioSegment

# Initialize the Flask app
app = Flask(__name__)

# Initialize the Whisper model
model = whisper.load_model("base")

# Route for the homepage (updated to render 'audioindex.html')
@app.route('/')
def index():
    return render_template('audioindex.html')  # Updated

# Route to handle audio upload and convert speech to text using Whisper AI
@app.route('/convert', methods=['POST'])
def convert():
    if 'audio_data' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    # Save the uploaded audio file
    audio_file = request.files['audio_data']
    audio_path = os.path.join("uploads", "audio.wav")
    audio_file.save(audio_path)

    # Convert the audio to the correct format (wav)
    sound = AudioSegment.from_file(audio_path)
    sound.export(audio_path, format="wav")

    # Transcribe the audio using Whisper
    result = model.transcribe(audio_path)
    transcription = result['text']

    # Return the transcription as a JSON response
    return jsonify({"transcription": transcription})

if __name__ == '__main__':
    # Ensure upload directory exists
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)
