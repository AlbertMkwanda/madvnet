from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import tempfile

BASE_DIR = os.path.dirname(__file__)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

app = Flask(__name__)
CORS(app)  # Allows React (port 3000) to talk to Flask (port 5000)

# Lazy-load processing utilities and optional whisper transcription
DeceptionProcessor = None
processor = None
processor_error_message = None
whisper_model = None
whisper_load_attempted = False
backend_initialized = False

# Model weight paths relative to this file
MODEL_DIR = os.path.join(BASE_DIR, 'models')
V_WEIGHTS = os.path.join(MODEL_DIR, 'visual_brain_90plus.pth')
T_WEIGHTS = os.path.join(MODEL_DIR, 'linguistic_brain_best.pth')
F_WEIGHTS = os.path.join(MODEL_DIR, 'fusion_brain_best.pth')


def load_whisper_model():
    global whisper_model, whisper_load_attempted
    whisper_load_attempted = True
    try:
        import whisper
        whisper_model = whisper.load_model("medium")
    except Exception:
        whisper_model = None


def get_processor():
    global DeceptionProcessor, processor, processor_error_message

    if DeceptionProcessor is None:
        try:
            from processor import DeceptionProcessor as DP
            DeceptionProcessor = DP
            processor_error_message = None
        except Exception as e:
            DeceptionProcessor = None
            processor_error_message = str(e)

    if processor is None and DeceptionProcessor is not None:
        try:
            processor = DeceptionProcessor(V_WEIGHTS, T_WEIGHTS, F_WEIGHTS)
            processor_error_message = None
        except Exception as e:
            processor = None
            processor_error_message = str(e)

    return processor


@app.before_request
def initialize_backend():
    global backend_initialized
    if backend_initialized:
        return

    print('Initializing backend models... this may take a while.')
    load_whisper_model()
    get_processor()
    if processor is None:
        print('WARNING: Processor failed to initialize:', processor_error_message)
    backend_initialized = True


@app.route('/health', methods=['GET'])
def health():
    status = 'ready' if processor is not None else 'starting'
    return jsonify({'status': status, 'processor_error': processor_error_message}), 200


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'no file uploaded'}), 400

    file = request.files['file']
    # Save to a temporary file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    try:
        file.save(tmp.name)

        # Transcribe with whisper if available
        transcript_text = ''
        if whisper_model is not None:
            try:
                result = whisper_model.transcribe(tmp.name, fp16=False, language='en', condition_on_previous_text=False)
                transcript_text = result.get('text', '').strip()
            except Exception:
                transcript_text = ''

        # Ensure processor is initialized (attempt lazy init for clearer errors)
        global processor
        if processor is None:
            processor = get_processor()
            if processor is None:
                if DeceptionProcessor is None:
                    return jsonify({'error': 'processor_not_available', 'message': 'DeceptionProcessor class could not be imported', 'detail': processor_error_message}), 500
                return jsonify({'error': 'processor_initialization_failed', 'message': 'Processor failed to initialize; ensure all model files and dependencies are available', 'detail': processor_error_message}), 500

        try:
            analysis = processor.analyze(tmp.name, transcript_text)
        except Exception as e:
            return jsonify({'error': 'processing_error', 'message': str(e)}), 500

        return jsonify({'status': 'ok', 'analysis': analysis})
    finally:
        try:
            tmp.close()
            os.unlink(tmp.name)
        except Exception:
            pass


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)