from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import numpy as np
import onnxruntime as ort
import json
from PIL import Image
import io
import os
import logging

app = Flask(__name__)

# Configura CORS
CORS(app, origins=[
    "http://localhost:3000", 
    "https://your-frontend-domain.vercel.app",
    "https://*.vercel.app"
])

# Configura logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Carica il modello e metadata
def load_model():
    model_path = "models/captcha_model.onnx"
    metadata_path = "models/captcha_model_metadata.json"
    
    try:
        # Specifica esplicitamente i provider
        providers = ['CPUExecutionProvider']
        
        session = ort.InferenceSession(model_path, providers=providers)
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"Modello caricato con successo. Input shape: {metadata.get('input_shape', 'N/A')}")
        return session, metadata
        
    except Exception as e:
        logger.error(f"Errore nel caricamento del modello: {e}")
        raise

try:
    model_session, model_metadata = load_model()
    
    # Estrai le dimensioni corrette dal metadata
    if 'input_shape' in model_metadata:
        # Formato: [batch, channels, height, width]
        input_shape = model_metadata['input_shape']
        expected_channels = input_shape[1]
        expected_height = input_shape[2]
        expected_width = input_shape[3]
    else:
        # Valori di default basati sull'errore
        expected_channels = 1
        expected_height = 40
        expected_width = 150
    
    char_set = model_metadata.get('char_set', 'abcdefghijklmnopqrstuvwxyz0123456789')
    logger.info(f"Model expects: {expected_channels} channels, {expected_height}x{expected_width}")
    
except Exception as e:
    logger.error(f"ERRORE CRITICO: Impossibile caricare il modello: {e}")
    model_session = None
    model_metadata = None
    expected_channels = 1
    expected_height = 40
    expected_width = 150
    char_set = 'abcdefghijklmnopqrstuvwxyz0123456789'

def preprocess_image(base64_string):
    """Preprocessa l'immagine base64 per il modello con le dimensioni corrette"""
    try:
        # Rimuovi l'header se presente
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decodifica base64
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        
        logger.info(f"Immagine originale: {image.mode}, {image.size}")
        
        # Converti nel formato colore corretto
        if expected_channels == 1:
            # Il modello si aspetta grayscale
            if image.mode != 'L':
                image = image.convert('L')
        else:
            # Il modello si aspetta RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
        
        # Ridimensiona alle dimensioni attese dal modello
        image = image.resize((expected_width, expected_height), Image.LANCZOS)
        logger.info(f"Immagine ridimensionata: {image.mode}, {image.size}")
        
        # Converti in array numpy
        image_array = np.array(image).astype(np.float32) / 255.0
        
        # Gestisci i canali in base alle aspettative del modello
        if expected_channels == 1:
            # Se il modello si aspetta 1 canale ma l'immagine è RGB, converti
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                image_array = np.mean(image_array, axis=2)  # Converti RGB in grayscale
            # Assicurati che sia 2D
            if len(image_array.shape) == 2:
                image_array = np.expand_dims(image_array, axis=0)  # Aggiungi dimensione canale
        else:
            # Se il modello si aspetta 3 canali ma l'immagine è grayscale, converti
            if len(image_array.shape) == 2:
                image_array = np.stack([image_array] * 3, axis=0)  # Converti in RGB
        
        # Formato finale: [channels, height, width]
        logger.info(f"Shape dopo elaborazione: {image_array.shape}")
        
        # Aggiungi dimensione batch: [batch, channels, height, width]
        image_array = np.expand_dims(image_array, axis=0)
        logger.info(f"Shape finale per il modello: {image_array.shape}")
        
        return image_array
        
    except Exception as e:
        logger.error(f"Errore nel preprocessing: {str(e)}")
        raise

def predict_captcha(image_array):
    """Esegue la predizione del captcha"""
    try:
        if model_session is None:
            raise ValueError("Modello non caricato")
            
        # Verifica che le dimensioni siano corrette
        expected_shape = (1, expected_channels, expected_height, expected_width)
        if image_array.shape != expected_shape:
            raise ValueError(f"Dimensioni errate: atteso {expected_shape}, ottenuto {image_array.shape}")
        
        # Get input name
        input_name = model_session.get_inputs()[0].name
        
        # Esegui inference
        outputs = model_session.run(None, {input_name: image_array})
        
        # Decodifica le predizioni
        captcha_text = decode_predictions(outputs, char_set)
        
        return captcha_text
    except Exception as e:
        raise ValueError(f"Errore nella predizione: {str(e)}")

def decode_predictions(outputs, char_set):
    """Decodifica le predizioni in testo"""
    try:
        # Il modello probabilmente restituisce 4 output (uno per ogni carattere)
        # o un singolo output con shape [batch, sequence_length, num_chars]
        
        if len(outputs) == 4:
            # Caso: 4 output separati per ogni carattere
            predicted_text = ""
            for i in range(4):
                if i < len(outputs):
                    char_probs = outputs[i][0]  # [num_chars]
                    predicted_idx = np.argmax(char_probs)
                    if predicted_idx < len(char_set):
                        predicted_text += char_set[predicted_idx]
                    else:
                        predicted_text += '?'
                else:
                    predicted_text += '?'
            return predicted_text
            
        elif len(outputs) == 1:
            # Caso: singolo output con multiple dimensioni
            predictions = outputs[0]  # Shape: [batch, sequence, num_chars] o [batch, num_chars, sequence]
            
            if len(predictions.shape) == 3:
                # [batch, sequence, num_chars]
                predicted_indices = np.argmax(predictions[0], axis=1)
            else:
                # [batch, num_chars, sequence] - trasponi se necessario
                if predictions.shape[1] == len(char_set):
                    predicted_indices = np.argmax(predictions[0], axis=0)
                else:
                    predicted_indices = np.argmax(predictions[0], axis=1)
            
            predicted_text = ''.join([char_set[idx] for idx in predicted_indices if idx < len(char_set)])
            return predicted_text
            
        else:
            # Cerca di interpretare l'output in modo generico
            logger.warning(f"Numero inaspettato di output: {len(outputs)}")
            # Prova con il primo output
            predictions = outputs[0]
            if len(predictions.shape) >= 2:
                predicted_indices = np.argmax(predictions[0], axis=-1)
                predicted_text = ''.join([char_set[idx] for idx in predicted_indices if idx < len(char_set)])
                return predicted_text
            else:
                return "????"
                
    except Exception as e:
        logger.error(f"Errore nella decodifica: {str(e)}")
        return "????"

def generate_fallback_captcha():
    """Genera un captcha casuale come fallback"""
    import random
    chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
    return ''.join(random.choice(chars) for _ in range(4))

@app.route('/solve-captcha', methods=['POST', 'OPTIONS'])
def solve_captcha():
    """Endpoint per risolvere i captcha"""
    if request.method == 'OPTIONS':
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
            logger.warning("Modello non caricato, uso fallback")
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
        
        logger.info(f"Captcha risolto: {captcha_code}")
        
        return jsonify({
            'success': True,
            'captcha_code': captcha_code,
            'message': 'Captcha risolto con successo',
            'fallback': False
        })
        
    except Exception as e:
        logger.error(f"Errore nel risolvere il captcha: {str(e)}")
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
        'model_details': {
            'channels': expected_channels,
            'height': expected_height,
            'width': expected_width,
            'char_set_length': len(char_set)
        } if model_status else None
    })

@app.route('/model-info', methods=['GET'])
def model_info():
    """Endpoint per informazioni dettagliate sul modello"""
    if model_session is None:
        return jsonify({'error': 'Modello non caricato'}), 500
    
    inputs_info = []
    for input in model_session.get_inputs():
        inputs_info.append({
            'name': input.name,
            'shape': input.shape,
            'type': input.type
        })
    
    outputs_info = []
    for output in model_session.get_outputs():
        outputs_info.append({
            'name': output.name,
            'shape': output.shape,
            'type': output.type
        })
    
    return jsonify({
        'inputs': inputs_info,
        'outputs': outputs_info,
        'metadata': model_metadata
    })

@app.route('/')
def home():
    return jsonify({
        'message': 'Captcha Solver API',
        'status': 'online',
        'model_loaded': model_session is not None,
        'expected_dimensions': {
            'channels': expected_channels,
            'height': expected_height,
            'width': expected_width
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)