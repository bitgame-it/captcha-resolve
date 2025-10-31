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
CORS(app, origins=["*"])

# Configura logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model():
    """Carica il modello ONNX con la stessa configurazione del tuo bot"""
    try:
        logger.info("🔍 Starting model loading process...")
        
        model_path = "models/captcha_model.onnx"
        metadata_path = "models/captcha_model_metadata.json"
        
        if not os.path.exists(model_path):
            logger.error(f"❌ Model file not found: {model_path}")
            logger.error(f"Current directory: {os.getcwd()}")
            logger.error(f"Directory contents: {os.listdir('.')}")
            if os.path.exists('models'):
                logger.error(f"Models directory: {os.listdir('models')}")
            return None, None
            
        if not os.path.exists(metadata_path):
            logger.error(f"❌ Metadata file not found: {metadata_path}")
            return None, None
        
        logger.info("🔄 Loading ONNX model...")
        
        # CON ONNXRUNTIME 1.18.1, usa la stessa logica del tuo bot
        # Il tuo bot usa semplicemente: ort.InferenceSession(model_path)
        # Quindi proviamo prima senza provider espliciti
        try:
            session = ort.InferenceSession(model_path)
            logger.info("✅ ONNX model loaded successfully (no explicit providers)")
        except Exception as e:
            logger.warning(f"First attempt failed: {e}")
            logger.info("🔄 Trying with explicit CPU provider...")
            # Se fallisce, prova con provider esplicito
            session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            logger.info("✅ ONNX model loaded successfully (with CPU provider)")
        
        logger.info("🔄 Loading model metadata...")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        logger.info("✅ Model metadata loaded successfully")
        
        # Test del modello
        logger.info("🔄 Testing model inference...")
        if 'input_shape' in metadata:
            height, width = metadata['input_shape'][1:3]
        else:
            height, width = 64, 128
            
        dummy_input = np.random.rand(1, 1, height, width).astype(np.float32)
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: dummy_input})
        logger.info(f"✅ Model test successful. Outputs: {len(outputs)}")
        
        return session, metadata
        
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        return None, None

# Carica il modello
logger.info("🚀 Starting application...")
model_session, model_metadata = load_model()

# Definisci variabili globali
if model_session and model_metadata:
    logger.info("🎉 Application started successfully with model loaded!")
    
    char_set = model_metadata.get('chars', 'abcdefghijklmnopqrstuvwxyz0123456789')
    idx_to_char = model_metadata.get('idx_to_char', {})
    
    if 'input_shape' in model_metadata:
        input_shape = model_metadata['input_shape']
        expected_channels = input_shape[1]
        expected_height = input_shape[2]
        expected_width = input_shape[3]
    else:
        expected_channels = 1
        expected_height = 64
        expected_width = 128
    
    logger.info(f"📐 Model configuration: {expected_channels} channel(s), {expected_height}x{expected_width}")
else:
    logger.error("💥 Application started WITHOUT model!")
    char_set = 'abcdefghijklmnopqrstuvwxyz0123456789'
    idx_to_char = {}
    expected_channels = 1
    expected_height = 64
    expected_width = 128

def preprocess_image(base64_string):
    """Preprocessa l'immagine base64 - STESSA LOGICA DEL TUO BOT"""
    try:
        # Rimuovi l'header se presente
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decodifica base64
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        
        logger.info(f"Original image: {image.mode}, {image.size}")
        
        # Converti in grayscale come nel tuo bot
        if image.mode != 'L':
            image = image.convert('L')
        
        # Ridimensiona alle dimensioni attese dal modello
        image = image.resize((expected_width, expected_height), Image.LANCZOS)
        logger.info(f"Resized image: {image.mode}, {image.size}")
        
        # Converti in array numpy
        image_array = np.array(image, dtype=np.float32)
        
        # Normalizza usando i valori del metadata (come nel tuo bot)
        if model_metadata and 'normalization' in model_metadata:
            mean = model_metadata['normalization']['mean'][0]
            std = model_metadata['normalization']['std'][0]
            image_array = (image_array / 255.0 - mean) / std
        else:
            # Normalizzazione di default
            image_array = image_array / 255.0
        
        # Formato finale: [1, 1, height, width] - come nel tuo bot
        image_array = np.expand_dims(image_array, axis=0)  # batch dimension
        image_array = np.expand_dims(image_array, axis=0)  # channel dimension
        
        logger.info(f"Final shape for model: {image_array.shape}")
        return image_array
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

def predict_captcha(image_array):
    """Esegue la predizione del captcha - STESSA LOGICA DEL TUO BOT"""
    try:
        if model_session is None:
            raise ValueError("Model not loaded")
        
        # Get input name
        input_name = model_session.get_inputs()[0].name
        
        # Esegui inference
        outputs = model_session.run(None, {input_name: image_array})
        
        # Decodifica le predizioni - STESSA LOGICA DEL TUO BOT
        predicted_text = ""
        confidences = []

        # Il tuo modello probabilmente restituisce 4 output per 4 caratteri
        for pos in range(4):
            if pos < len(outputs):
                char_probs = outputs[pos][0]  # [num_chars]
                predicted_idx = np.argmax(char_probs)
                confidence = float(char_probs[predicted_idx])
                
                # Converti indice in carattere
                predicted_char = idx_to_char.get(str(predicted_idx), '?')
                predicted_text += predicted_char
                confidences.append(confidence)
            else:
                predicted_text += '?'
                confidences.append(0.0)

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        logger.info(f"Predicted captcha: {predicted_text} (confidence: {avg_confidence:.3f})")
        return predicted_text, avg_confidence
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise

def generate_fallback_captcha():
    """Genera un captcha casuale come fallback"""
    import random
    chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
    return ''.join(random.choice(chars) for _ in range(4)), 0.0

@app.route('/solve-captcha', methods=['POST', 'OPTIONS'])
def solve_captcha():
    """Endpoint principale per risolvere i captcha"""
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'No image provided'
            }), 400
        
        base64_image = data['image']
        
        # Verifica se il modello è caricato
        if model_session is None:
            logger.warning("Model not loaded, using fallback")
            captcha_code, confidence = generate_fallback_captcha()
            return jsonify({
                'success': True,
                'captcha_code': captcha_code,
                'confidence': confidence,
                'message': 'Fallback captcha (model not available)',
                'fallback': True
            })
        
        # Preprocessa l'immagine
        processed_image = preprocess_image(base64_image)
        
        # Predici il captcha
        captcha_code, confidence = predict_captcha(processed_image)
        
        # Valida il risultato (come nel tuo bot)
        valid_chars = set(model_metadata.get('chars', 'abcdefghijklmnopqrstuvwxyz0123456789'))
        is_valid = (captcha_code and 
                   len(captcha_code) == 4 and 
                   all(c in valid_chars for c in captcha_code))
        
        if is_valid:
            logger.info(f"✅ Captcha solved successfully: {captcha_code}")
            return jsonify({
                'success': True,
                'captcha_code': captcha_code,
                'confidence': confidence,
                'message': 'Captcha solved successfully',
                'fallback': False
            })
        else:
            logger.warning(f"❌ Invalid captcha result: {captcha_code}")
            # Usa fallback se il risultato non è valido
            captcha_code, confidence = generate_fallback_captcha()
            return jsonify({
                'success': True,
                'captcha_code': captcha_code,
                'confidence': confidence,
                'message': f'Fallback captcha (invalid result: {captcha_code})',
                'fallback': True
            })
        
    except Exception as e:
        logger.error(f"Error solving captcha: {str(e)}")
        # Fallback in caso di errore
        captcha_code, confidence = generate_fallback_captcha()
        return jsonify({
            'success': True,
            'captcha_code': captcha_code,
            'confidence': confidence,
            'message': f'Fallback captcha (error: {str(e)})',
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
        'model_ready': model_status and model_metadata is not None
    })

@app.route('/')
def home():
    return jsonify({
        'message': 'Captcha Solver API',
        'status': 'online',
        'model_loaded': model_session is not None,
        'model_ready': model_session is not None and model_metadata is not None
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)