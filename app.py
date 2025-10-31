from flask import Flask, request, jsonify
from flask_cors import CORS  # Importa CORS
import base64
import numpy as np
import onnxruntime as ort
import json
from PIL import Image
import io
import os

app = Flask(__name__)

# Configura CORS per accettare richieste da qualsiasi origine
CORS(app, origins=["http://localhost:3000", "https://your-frontend-domain.vercel.app"])

# Oppure per accettare da tutte le origini (meno sicuro ma funzionale per testing)
# CORS(app, origins="*")

# Carica il modello e metadata
def load_model():
    model_path = "models/captcha_model.onnx"
    metadata_path = "models/captcha_model_metadata.json"
    
    session = ort.InferenceSession(model_path)
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return session, metadata

model_session, model_metadata = load_model()
char_set = model_metadata.get('char_set', 'abcdefghijklmnopqrstuvwxyz0123456789')

def preprocess_image(base64_string):
    """Preprocessa l'immagine base64 per il modello"""
    try:
        # Rimuovi l'header se presente
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decodifica base64
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        
        # Converti in RGB se necessario
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Ridimensiona alle dimensioni attese dal modello
        target_width = model_metadata.get('image_width', 128)
        target_height = model_metadata.get('image_height', 64)
        image = image.resize((target_width, target_height))
        
        # Converti in array numpy e normalizza
        image_array = np.array(image).astype(np.float32) / 255.0
        
        # Transponi per formato CHW (Channel, Height, Width)
        image_array = np.transpose(image_array, (2, 0, 1))
        
        # Aggiungi dimensione batch
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    except Exception as e:
        raise ValueError(f"Errore nel preprocessing: {str(e)}")

def predict_captcha(image_array):
    """Esegue la predizione del captcha"""
    try:
        # Get input name
        input_name = model_session.get_inputs()[0].name
        
        # Esegui inference
        outputs = model_session.run(None, {input_name: image_array})
        
        # Il modello restituisce probabilmente una forma [1, lunghezza_captcha, num_caratteri]
        # Adatta questa parte in base al formato di output del tuo modello
        predictions = outputs[0]
        
        # Decodifica le predizioni (questa parte dipende dal tuo modello)
        captcha_text = decode_predictions(predictions, char_set)
        
        return captcha_text
    except Exception as e:
        raise ValueError(f"Errore nella predizione: {str(e)}")

def decode_predictions(predictions, char_set):
    """Decodifica le predizioni in testo"""
    try:
        # Assumendo che le predizioni siano di forma [batch, sequence_length, num_chars]
        # Prendi l'argmax lungo l'asse dei caratteri
        predicted_indices = np.argmax(predictions[0], axis=1)
        
        # Converti gli indici in caratteri
        captcha_text = ''.join([char_set[idx] for idx in predicted_indices if idx < len(char_set)])
        
        return captcha_text
    except Exception as e:
        raise ValueError(f"Errore nella decodifica: {str(e)}")

@app.route('/solve-captcha', methods=['POST', 'OPTIONS'])
def solve_captcha():
    """Endpoint per risolvere i captcha"""
    if request.method == 'OPTIONS':
        # Gestisci preflight request
        return '', 200
    
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'Nessuna immagine fornita'
            }), 400
        
        base64_image = data['image']
        
        # Preprocessa l'immagine
        processed_image = preprocess_image(base64_image)
        
        # Predici il captcha
        captcha_code = predict_captcha(processed_image)
        
        print(f"Captcha risolto: {captcha_code}")
        
        return jsonify({
            'success': True,
            'captcha_code': captcha_code,
            'message': 'Captcha risolto con successo'
        })
        
    except Exception as e:
        print(f"Errore nel risolvere il captcha: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint per health check"""
    return jsonify({
        'status': 'healthy',
        'message': 'Captcha solver service is running',
        'model_loaded': True
    })

@app.route('/')
def home():
    return jsonify({
        'message': 'Captcha Solver API',
        'endpoints': {
            'POST /solve-captcha': 'Risolvi un captcha',
            'GET /health': 'Health check'
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)