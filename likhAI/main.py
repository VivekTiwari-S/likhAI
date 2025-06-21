from flask import Flask, request, jsonify, send_from_directory
import base64
import io
from PIL import Image
from ocr import extract_text  # Assuming extract_text is properly implemented in ocr.py
from translator import translate_text  # Assuming this is implemented
from summarize import summarize_text  # Assuming this is implemented
from paraphrase import paraphrase_text  # Assuming this is implemented

# Initialize Flask app
app = Flask(__name__)

# Enable CORS if needed (to handle cross-origin requests)
from flask_cors import CORS
CORS(app)

@app.route('/')
def index():
    return send_from_directory('templates', 'index2.html')

# @app.route('/extract_text', methods=['POST'])
# def process_image():
#     data = request.json
#     try:
#         # Extract and decode base64 image data
#         image_data = data['image'].split(',')[1]  # Remove "data:image/jpeg;base64,"
#         image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        
#         # Process the image using the extract_text function from ocr.py
#         text = extract_text(image)
        
#         return jsonify({'text': text}), 200
    
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

@app.route('/extract_text', methods=['POST'])
def process_image():
    data = request.json
    try:
        # Extract and decode base64 image data
        image_data = data['image'].split(',')[1]  # Remove "data:image/jpeg;base64,"
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))

        # Process the image using the extract_text function from ocr.py
        text = extract_text(image)

        return jsonify({'text': text}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/translate_text', methods=['POST'])
def translate():
    data = request.json
    try:
        text = data['text']
        language = data['language']
        
        # Call the translation function from translator.py
        translated = translate_text(text, language)
        
        return jsonify({'result': translated}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/summarize_text', methods=['POST'])
def summarize():
    data = request.json
    try:
        text = data['text']
        
        # Call the summarize function from summarize.py
        summary = summarize_text(text)
        
        return jsonify({'result': summary}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/paraphrase_text', methods=['POST'])
def paraphrase():
    data = request.json
    try:
        text = data['text']
        
        # Call the paraphrase function from paraphrase.py
        paraphrased = paraphrase_text(text)
        
        return jsonify({'result': paraphrased}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)