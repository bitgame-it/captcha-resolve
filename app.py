from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
import os
import sys
import asyncio
import logging

# Aggiungi utils al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from gift_captcha_solver import GiftCaptchaSolver

app = Flask(__name__)
CORS(app)

# Configura logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inizializza il solver
solver = GiftCaptchaSolver(save_images=0)

@app.route('/')
def home():
    return jsonify({
        "status": "WOS Captcha Solver API is running", 
        "version": "3.0",
        "model": "ONNX"
    })

@app.route('/health')
def health():
    return jsonify({"status": "healthy"})

@app.route('/solve', methods=['POST', 'OPTIONS'])
async def solve_captcha():
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'Image data is required'
            }), 400

        image_b64 = data['image']
        player_id = data.get('playerId', 'unknown')
        
        logger.info(f"Solving captcha for player: {player_id}")
        
        # Decodifica base64
        image_bytes = base64.b64decode(image_b64)
        
        # Risolvi captcha
        solved_code, success, method, confidence, image_path = await solver.solve_captcha(
            image_bytes, player_id
        )
        
        logger.info(f"Captcha result: {solved_code} (success: {success}, confidence: {confidence})")
        
        return jsonify({
            'success': success,
            'text': solved_code,
            'method': method,
            'confidence': confidence
        })
        
    except Exception as e:
        logger.error(f"Error solving captcha: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)