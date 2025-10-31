from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import numpy as np
import onnxruntime as ort
import json
from PIL import Image
import io
import os

app = Flask(__name__)

# Configura CORS - per development accetta da localhost
CORS(app, origins=[
    "http://localhost:3000", 
    "https://your-frontend-domain.vercel.app",
    "https://*.vercel.app"
])

# Carica il modello e metadata
def load_model():
    model_path = "models/captcha_model.onnx"
    metadata_path = "models/captcha_model_metadata.json"
    
    try:
        # Specifica esplicitamente i provider per ONNX Runtime
        providers = ['CPUExecutionProvider']
        
        # Prova prima con CPUExecutionProvider
        session = ort.InferenceSession(model_path, providers=providers)
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"Modello caricato con successo. Providers: {providers}")
        return session, metadata
        
    except Exception as e:
        print(f"Errore nel caricamento del modello: {e}")
        raise

try:
    model_session, model_metadata = load_model()
    char_set = model_metadata.get('char_set', 'abcdefghijklmnopqrstuvwxyz0123456789')
    print(f"Character set: {char_set}")
except Exception as e:
    print(f"ERRORE CRITICO: Impossibile caricare il modello: {e}")
    model_session = None
    model_metadata = None
    char_set = 'abcdefghijklmnopqrstuvwxyz0123456789'

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
        target_width = model_metadata.get('image_width', 128) if model_metadata else 128
        target_height = model_metadata.get('image_height', 64) if model_metadata else 64
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
        if model_session is None:
            raise ValueError("Modello non caricato")
            
        # Get input name
        input_name = model_session.get_inputs()[0].name
        
        # Esegui inference
        outputs = model_session.run(None, {input_name: image_array})
        
        # Assumendo che il modello restituisca logits per ogni carattere
        # Adatta questa parte in base al formato di output del tuo modello
        predictions = outputs[0]
        
        # Decodifica le predizioni
        captcha_text = decode_predictions(predictions, char_set)
        
        return captcha_text
    except Exception as e:
        raise ValueError(f"Errore nella predizione: {str(e)}")

def decode_predictions(predictions, char_set):
    """Decodifica le predizioni in testo"""
    try:
        # Assumendo che le predizioni siano di forma [batch, sequence_length, num_chars]
        # Prendi l'argmax lungo l'asse dei caratteri
        if len(predictions.shape) == 3:
            # Formato: [batch, sequence, characters]
            predicted_indices = np.argmax(predictions[0], axis=1)
        else:
            # Formato diverso, adatta secondo le tue necessità
            predicted_indices = np.argmax(predictions, axis=-1)
            if len(predicted_indices.shape) > 1:
                predicted_indices = predicted_indices[0]
        
        # Converti gli indici in caratteri
        captcha_text = ''.join([char_set[idx] for idx in predicted_indices if idx < len(char_set)])
        
        return captcha_text
    except Exception as e:
        raise ValueError(f"Errore nella decodifica: {str(e)}")

def generate_fallback_captcha():
    """Genera un captcha casuale come fallback"""
    import random
    chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
    return ''.join(random.choice(chars) for _ in range(4))

@app.route('/solve-captcha', methods=['POST', 'OPTIONS'])
def solve_captcha():
    """Endpoint per risolvere i captcha"""
    if request.method == 'OPTIONS':
        # Gestisci preflight request per CORS
        return '', 200
    
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'Nessuna immagine fornita'
            }), 400
        
        base64_image = data['image']
        
        # Verifica se il modello è caricato
        if model_session is None:
            print("Modello non caricato, uso fallback")
            fallback_captcha = generate_fallback_captcha()
            return jsonify({
                'success': True,
                'captcha_code': fallback_captcha,
                'message': 'Fallback captcha (modello non disponibile)',
                'fallback': True
            })
        
        # Preprocessa l'immagine
        processed_image = preprocess_image(base64_image)
        
        # Predici il captcha
        captcha_code = predict_captcha(processed_image)
        
        print(f"Captcha risolto: {captcha_code}")
        
        return jsonify({
            'success': True,
            'captcha_code': captcha_code,
            'message': 'Captcha risolto con successo',
            'fallback': False
        })
        
    except Exception as e:
        print(f"Errore nel risolvere il captcha: {str(e)}")
        # Fallback in caso di errore
        fallback_captcha = generate_fallback_captcha()
        return jsonify({
            'success': True,
            'captcha_code': fallback_captcha,
            'message': f'Fallback captcha (errore: {str(e)})',
            'fallback': True
        })

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint per health check"""
    model_status = model_session is not None
    return jsonify({
        'status': 'healthy',
        'message': 'Captcha solver service is running',
        'model_loaded': model_status,
        'model_providers': ['CPUExecutionProvider'] if model_status else []
    })

@app.route('/')
def home():
    return jsonify({
        'message': 'Captcha Solver API',
        'status': 'online',
        'model_loaded': model_session is not None,
        'endpoints': {
            'POST /solve-captcha': 'Risolvi un captcha',
            'GET /health': 'Health check'
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)