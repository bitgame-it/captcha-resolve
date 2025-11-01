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
import threading
import time
import queue
import uuid
from datetime import datetime
import requests
import hashlib
import random

app = Flask(__name__)
CORS(app, origins=["*"])

# Configura logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurazione Supabase
from supabase import create_client, Client

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

supabase_client = None

if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("✅ Supabase client initialized successfully")
    except Exception as e:
        logger.warning(f"❌ Supabase initialization failed: {e}")
else:
    logger.warning("❌ Supabase credentials not provided - running in limited mode")
    
# Variabili globali per i worker
worker_queue = queue.Queue()
active_workers = {}
worker_results = {}

def load_model():
    """Carica il modello ONNX"""
    try:
        logger.info("🔍 Starting model loading process...")
        
        model_path = "models/captcha_model.onnx"
        metadata_path = "models/captcha_model_metadata.json"
        
        if not os.path.exists(model_path):
            logger.error(f"❌ Model file not found: {model_path}")
            return None, None
            
        if not os.path.exists(metadata_path):
            logger.error(f"❌ Metadata file not found: {metadata_path}")
            return None, None
        
        logger.info("🔄 Loading ONNX model...")
        
        try:
            session = ort.InferenceSession(model_path)
            logger.info("✅ ONNX model loaded successfully")
        except Exception as e:
            logger.warning(f"First attempt failed: {e}")
            session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            logger.info("✅ ONNX model loaded successfully (with CPU provider)")
        
        logger.info("🔄 Loading model metadata...")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        logger.info("✅ Model metadata loaded successfully")
        
        return session, metadata
        
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        return None, None

# Carica il modello
logger.info("🚀 Starting application...")
model_session, model_metadata = load_model()

# Configurazione dimensioni modello
if model_session and model_metadata:
    logger.info("🎉 Application started successfully with model loaded!")
    
    char_set = model_metadata.get('chars', 'abcdefghijklmnopqrstuvwxyz0123456789')
    idx_to_char = model_metadata.get('idx_to_char', {})
    
    if 'input_shape' in model_metadata:
        input_shape = model_metadata['input_shape']
        if len(input_shape) == 4:
            expected_channels = input_shape[1]
            expected_height = input_shape[2]
            expected_width = input_shape[3]
        elif len(input_shape) == 3:
            expected_channels = input_shape[0]
            expected_height = input_shape[1]
            expected_width = input_shape[2]
        else:
            expected_channels = 1
            expected_height = 64
            expected_width = 128
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
    """Preprocessa l'immagine base64"""
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        
        if image.mode != 'L':
            image = image.convert('L')
        
        image = image.resize((expected_width, expected_height), Image.LANCZOS)
        
        image_array = np.array(image, dtype=np.float32)
        
        if model_metadata and 'normalization' in model_metadata:
            mean = model_metadata['normalization']['mean'][0]
            std = model_metadata['normalization']['std'][0]
            image_array = (image_array / 255.0 - mean) / std
        else:
            image_array = image_array / 255.0
        
        image_array = np.expand_dims(image_array, axis=0)
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

def predict_captcha(image_array):
    """Esegue la predizione del captcha"""
    try:
        if model_session is None:
            raise ValueError("Model not loaded")
        
        input_name = model_session.get_inputs()[0].name
        outputs = model_session.run(None, {input_name: image_array})
        
        predicted_text = ""
        confidences = []

        for pos in range(4):
            if pos < len(outputs):
                char_probs = outputs[pos][0]
                predicted_idx = np.argmax(char_probs)
                confidence = float(char_probs[predicted_idx])
                
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
    chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
    return ''.join(random.choice(chars) for _ in range(4)), 0.0

class BulkRedeemWorker:
    def __init__(self, record_id, player_list, gift_code, worker_id):
        self.worker_id = worker_id
        self.record_id = record_id
        self.player_list = player_list
        self.gift_code = gift_code
        self.status = "pending"
        self.progress = 0
        self.total = len(player_list)
        self.results = []
        self.start_time = None
        self.end_time = None
    
    def encode_wos_data(self, data):
        """Encoding per le richieste WOS API"""
        secret = "tB87#kPtkxqOS2"
        sorted_keys = sorted(data.keys())
        encoded_data = "&".join([f"{key}={data[key]}" for key in sorted_keys])
        sign = hashlib.md5((encoded_data + secret).encode()).hexdigest()
        return {**data, "sign": sign}
    
    def get_wos_session(self, player_id):
        """Ottieni una sessione WOS autenticata"""
        try:
            timestamp = str(int(time.time()))
            data_to_encode = {
                "fid": player_id,
                "time": timestamp
            }
            
            encoded_data = self.encode_wos_data(data_to_encode)
            
            headers = {
                "accept": "application/json, text/plain, */*",
                "content-type": "application/x-www-form-urlencoded",
                "origin": "https://wos-giftcode.centurygame.com",
            }
            
            session = requests.Session()
            session.headers.update(headers)
            
            body_params = "&".join([f"{key}={value}" for key, value in encoded_data.items()])
            
            logger.info(f"🔄 Authenticating player {player_id} with WOS API...")
            response = session.post(
                "https://wos-giftcode-api.centurygame.com/api/player",
                data=body_params,
                timeout=30
            )
            
            auth_data = response.json()
            
            if auth_data.get("msg") == "success":
                logger.info(f"✅ Player {player_id} authenticated successfully")
                return session
            else:
                logger.error(f"❌ Failed to authenticate player {player_id}: {auth_data.get('msg')}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error authenticating player {player_id}: {e}")
            return None

    def solve_captcha_for_wos(self, player_id, session):
        """Risolvi il captcha usando una sessione autenticata"""
        try:
            timestamp = str(int(time.time() * 1000))
            data_to_encode = {
                "fid": player_id,
                "time": timestamp,
                "init": "0"
            }
            
            encoded_data = self.encode_wos_data(data_to_encode)
            body_params = "&".join([f"{key}={value}" for key, value in encoded_data.items()])
            
            logger.info("🔄 Requesting captcha from WOS API...")
            response = session.post(
                "https://wos-giftcode-api.centurygame.com/api/captcha",
                data=body_params,
                timeout=30
            )
            
            captcha_data = response.json()
            
            if captcha_data.get("msg") == "SUCCESS" and captcha_data.get("data", {}).get("img"):
                base64_image = captcha_data["data"]["img"]
                
                captcha_code, confidence = solve_captcha_internal(base64_image)
                logger.info(f"🎯 Solved captcha for player {player_id}: {captcha_code} (confidence: {confidence:.3f})")
                return captcha_code
            else:
                error_msg = captcha_data.get('msg', 'Unknown error')
                logger.error(f"❌ Failed to load captcha: {error_msg}")
                raise Exception(f"Failed to load captcha: {error_msg}")
            
        except Exception as e:
            logger.error(f"❌ Error in captcha process for player {player_id}: {e}")
            captcha_code, _ = generate_fallback_captcha()
            return captcha_code

    def redeem_gift_code_for_player(self, player_id, session, captcha_code):
        """Riscatta il codice regalo usando una sessione autenticata"""
        try:
            timestamp = str(int(time.time()))
            redeem_data = {
                "fid": player_id,
                "cdk": self.gift_code,
                "captcha_code": captcha_code,
                "time": timestamp
            }
            
            encoded_redeem_data = self.encode_wos_data(redeem_data)
            body_params = "&".join([f"{key}={value}" for key, value in encoded_redeem_data.items()])
            
            logger.info("🔄 Sending redeem request...")
            response = session.post(
                "https://wos-giftcode-api.centurygame.com/api/gift_code",
                data=body_params,
                timeout=30
            )
            
            result_data = response.json()
            
            return self.interpret_redeem_result(player_id, result_data, captcha_code)
            
        except Exception as e:
            logger.error(f"❌ Error redeeming gift code for player {player_id}: {e}")
            return {
                'player_id': player_id,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def interpret_redeem_result(self, player_id, result_data, captcha_code):
        """Interpreta la risposta del redeem"""
        message = result_data.get("msg", "")
        error_code = result_data.get("err_code")
        
        if message == "SUCCESS":
            return {
                'player_id': player_id,
                'success': True,
                'message': 'Codice riscattato con successo',
                'timestamp': datetime.now().isoformat()
            }
        elif message == "RECEIVED." and error_code == 40008:
            return {
                'player_id': player_id,
                'success': True, 
                'message': 'Codice già riscattato precedentemente',
                'timestamp': datetime.now().isoformat()
            }
        elif message == "CDK NOT FOUND." and error_code == 40014:
            return {
                'player_id': player_id,
                'success': False,
                'error': 'Codice regalo non valido',
                'timestamp': datetime.now().isoformat()
            }
        elif message == "TIME ERROR." and error_code == 40007:
            return {
                'player_id': player_id,
                'success': False,
                'error': 'Codice regalo scaduto',
                'timestamp': datetime.now().isoformat()
            }
        elif message == "CAPTCHA ERROR.":
            return {
                'player_id': player_id,
                'success': False,
                'error': f'Errore captcha (soluzione provata: {captcha_code})',
                'timestamp': datetime.now().isoformat()
            }
        elif message == "NOT LOGIN.":
            return {
                'player_id': player_id,
                'success': False,
                'error': 'Errore autenticazione: sessione non valida',
                'timestamp': datetime.now().isoformat()
            }
        else:
            return {
                'player_id': player_id,
                'success': False,
                'error': f"{message} (code: {error_code})",
                'timestamp': datetime.now().isoformat()
            }
    
    def update_supabase_progress(self):
        """Aggiorna il progresso nel database Supabase"""
        if not supabase_client:
            if self.progress % 10 == 0 or self.progress == self.total:
                logger.info(f"📊 Progress: {self.progress}/{self.total} (Supabase not available)")
            return
            
        try:
            result = supabase_client.table("bulk_redeem_requests").update({
                "progress_current": self.progress,
                "progress_total": self.total,
                "updated_at": datetime.now().isoformat()
            }).eq("id", self.record_id).execute()
            
        except Exception as e:
            logger.error(f"Error updating Supabase progress: {e}")
    
    def run(self):
        """Esegue il worker per il riscatto bulk"""
        self.status = "running"
        self.start_time = datetime.now()
        active_workers[self.worker_id] = self
        
        try:
            if supabase_client:
                try:
                    # Aggiorna lo stato a "running"
                    supabase_client.table("bulk_redeem_requests").update({
                        "status": "running",
                        "started_at": self.start_time.isoformat(),
                        "worker_id": self.worker_id,
                        "updated_at": self.start_time.isoformat()
                    }).eq("id", self.record_id).execute()
                    logger.info(f"✅ Updated Supabase record to running: {self.record_id}")
                except Exception as e:
                    logger.error(f"❌ Failed to update Supabase record: {e}")
            
            for i, player_id in enumerate(self.player_list):
                if self.status == "stopped":
                    logger.info(f"🛑 Worker {self.worker_id} stopped by user")
                    break
                
                logger.info(f"🔄 Processing player {i+1}/{self.total}: {player_id}")
                
                try:
                    session = self.get_wos_session(player_id)
                    if not session:
                        result = {
                            'player_id': player_id,
                            'success': False,
                            'error': 'Failed to authenticate with WOS API',
                            'timestamp': datetime.now().isoformat()
                        }
                        self.results.append(result)
                        self.progress = i + 1
                        self.update_supabase_progress()
                        continue
                    
                    captcha_code = self.solve_captcha_for_wos(player_id, session)
                    result = self.redeem_gift_code_for_player(player_id, session, captcha_code)
                    
                except Exception as e:
                    logger.error(f"❌ Error processing player {player_id}: {e}")
                    result = {
                        'player_id': player_id,
                        'success': False,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    }
                
                self.results.append(result)
                self.progress = i + 1
                self.update_supabase_progress()
                
                success_status = "✅" if result.get('success') else "❌"
                logger.info(f"{success_status} Player {player_id} processed: {result.get('message', result.get('error', 'Unknown'))}")
                
                time.sleep(2)
            
            if self.status == "stopped":
                final_status = "stopped"
            else:
                final_status = "completed"
                self.status = "completed"
                successful = len([r for r in self.results if r.get('success', False)])
                logger.info(f"✅ Worker {self.worker_id} completed: {successful}/{self.total} successful")
            
        except Exception as e:
            logger.error(f"❌ Error in bulk redeem worker {self.worker_id}: {e}")
            self.status = "failed"
            final_status = "failed"
            
        finally:
            self.end_time = datetime.now()
            duration = (self.end_time - self.start_time).total_seconds() if self.start_time else 0
            
            worker_results[self.worker_id] = {
                'status': self.status,
                'results': self.results,
                'summary': {
                    'total': self.total,
                    'successful': len([r for r in self.results if r.get('success', False)]),
                    'failed': len([r for r in self.results if not r.get('success', True)])
                },
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'end_time': self.end_time.isoformat() if self.end_time else None,
                'duration_seconds': duration
            }
            
            if supabase_client:
                try:
                    supabase_client.table("bulk_redeem_requests").update({
                        "status": final_status,
                        "updated_at": self.end_time.isoformat(),
                        "results": self.results
                    }).eq("id", self.record_id).execute()
                    logger.info(f"✅ Updated final status in Supabase: {final_status}")
                except Exception as e:
                    logger.error(f"❌ Failed to update final status in Supabase: {e}")
            
            active_workers.pop(self.worker_id, None)
            logger.info(f"🗑️ Worker {self.worker_id} removed from active workers")

def worker_manager():
    """Gestisce i worker in background"""
    while True:
        try:
            worker = worker_queue.get()
            if worker is None:
                break
            worker.run()
            worker_queue.task_done()
        except Exception as e:
            logger.error(f"Error in worker manager: {e}")

worker_thread = threading.Thread(target=worker_manager, daemon=True)
worker_thread.start()

def solve_captcha_internal(base64_image):
    """Funzione interna per risolvere captcha"""
    try:
        if model_session is None:
            return generate_fallback_captcha()
        
        processed_image = preprocess_image(base64_image)
        return predict_captcha(processed_image)
    except Exception as e:
        logger.error(f"Error in internal captcha solving: {e}")
        return generate_fallback_captcha()

# API Routes
@app.route('/solve-captcha', methods=['POST', 'OPTIONS'])
def solve_captcha_endpoint():
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
        
        if model_session is None:
            captcha_code, confidence = generate_fallback_captcha()
            return jsonify({
                'success': True,
                'captcha_code': captcha_code,
                'confidence': confidence,
                'message': 'Fallback captcha (model not available)',
                'fallback': True
            })
        
        processed_image = preprocess_image(base64_image)
        captcha_code, confidence = predict_captcha(processed_image)
        
        valid_chars = set(model_metadata.get('chars', 'abcdefghijklmnopqrstuvwxyz0123456789'))
        is_valid = (captcha_code and 
                   len(captcha_code) == 4 and 
                   all(c in valid_chars for c in captcha_code))
        
        if is_valid:
            return jsonify({
                'success': True,
                'captcha_code': captcha_code,
                'confidence': confidence,
                'message': 'Captcha solved successfully',
                'fallback': False
            })
        else:
            captcha_code, confidence = generate_fallback_captcha()
            return jsonify({
                'success': True,
                'captcha_code': captcha_code,
                'confidence': confidence,
                'message': f'Fallback captcha (invalid result: {captcha_code})',
                'fallback': True
            })
        
    except Exception as e:
        captcha_code, confidence = generate_fallback_captcha()
        return jsonify({
            'success': True,
            'captcha_code': captcha_code,
            'confidence': confidence,
            'message': f'Fallback captcha (error: {str(e)})',
            'fallback': True
        })

@app.route('/api/start-bulk-redeem', methods=['POST'])
def start_bulk_redeem():
    try:
        data = request.get_json()
        
        if not data or 'players' not in data or 'gift_code' not in data:
            return jsonify({'error': 'Missing required fields: players and gift_code'}), 400
        
        player_list = data['players']
        gift_code = data['gift_code']
        record_id = data.get('record_id', str(uuid.uuid4()))
        
        # Assicurati che record_id sia un UUID valido
        try:
            uuid.UUID(record_id)
        except ValueError:
            # Se non è un UUID valido, creane uno nuovo
            record_id = str(uuid.uuid4())
        
        if not isinstance(player_list, list) or len(player_list) == 0:
            return jsonify({'error': 'Invalid player list'}), 400
        
        if len(player_list) > 100:
            return jsonify({'error': 'Maximum 100 players allowed'}), 400
        
        # Crea il record nel database prima di avviare il worker
        if supabase_client:
            try:
                supabase_client.table("bulk_redeem_requests").insert({
                    "id": record_id,
                    "gift_code": gift_code,
                    "player_list": player_list,
                    "status": "starting",
                    "progress_current": 0,
                    "progress_total": len(player_list),
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat()
                }).execute()
                logger.info(f"✅ Created Supabase record: {record_id}")
            except Exception as e:
                logger.error(f"❌ Failed to create Supabase record: {e}")
                # Continua comunque, anche senza record nel database
        
        worker_id = str(uuid.uuid4())
        worker = BulkRedeemWorker(record_id, player_list, gift_code, worker_id)
        
        worker_queue.put(worker)
        
        logger.info(f"🚀 Started bulk redeem worker {worker_id} for {len(player_list)} players")
        
        return jsonify({
            'success': True,
            'worker_id': worker_id,
            'record_id': record_id,
            'total_players': len(player_list),
            'supabase_available': supabase_client is not None,
            'message': f'Started processing {len(player_list)} players'
        })
        
    except Exception as e:
        logger.error(f"❌ Error starting bulk redeem: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/worker-status/<worker_id>', methods=['GET'])
def get_worker_status(worker_id):
    try:
        worker = active_workers.get(worker_id)
        result = worker_results.get(worker_id)
        
        if not worker and not result:
            return jsonify({'error': 'Worker not found'}), 404
        
        response = {
            'worker_id': worker_id,
            'status': result['status'] if result else worker.status,
            'progress': worker.progress if worker else result['summary']['total'],
            'total': worker.total if worker else result['summary']['total'],
            'percentage': round((worker.progress / worker.total * 100) if worker else 100, 1)
        }
        
        if result:
            response.update({
                'results': result['results'],
                'summary': result['summary'],
                'start_time': result['start_time'],
                'end_time': result['end_time']
            })
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error getting worker status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stop-worker/<worker_id>', methods=['POST'])
def stop_worker(worker_id):
    try:
        worker = active_workers.get(worker_id)
        if worker:
            worker.status = "stopped"
            return jsonify({'success': True, 'message': 'Worker stopped'})
        else:
            return jsonify({'error': 'Worker not found or already completed'}), 404
            
    except Exception as e:
        logger.error(f"Error stopping worker: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/active-workers', methods=['GET'])
def get_active_workers():
    active = []
    completed = []
    
    for worker_id, worker in active_workers.items():
        if worker.status in ['running', 'pending']:
            active.append({
                'worker_id': worker_id,
                'status': worker.status,
                'progress': worker.progress,
                'total': worker.total,
                'percentage': round((worker.progress / worker.total * 100), 1)
            })
    
    for worker_id, result in worker_results.items():
        completed.append({
            'worker_id': worker_id,
            'status': result['status'],
            'summary': result['summary'],
            'start_time': result['start_time'],
            'end_time': result['end_time']
        })
    
    return jsonify({
        'active': active,
        'completed': completed
    })

@app.route('/health', methods=['GET'])
def health_check():
    model_status = model_session is not None
    supabase_status = supabase_client is not None
    
    health_status = "healthy"
    if not model_status:
        health_status = "degraded"
    if not supabase_status:
        health_status = "degraded"
    
    return jsonify({
        'status': health_status,
        'message': 'Captcha solver service is running',
        'model_loaded': model_status,
        'model_ready': model_status and model_metadata is not None,
        'supabase_connected': supabase_status,
        'active_workers': len(active_workers),
        'worker_queue_size': worker_queue.qsize(),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/')
def home():
    return jsonify({
        'message': 'Captcha Solver API',
        'status': 'online',
        'model_loaded': model_session is not None,
        'model_ready': model_session is not None and model_metadata is not None,
        'bulk_redeem_supported': True,
        'supabase_available': supabase_client is not None
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)