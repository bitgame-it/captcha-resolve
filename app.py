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

# Configura CORS
CORS(app, origins=["*"])

# Configura logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurazione Supabase - CORREGGI QUESTA PARTE
from supabase import create_client, Client

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

supabase_client = None

if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("✅ Supabase client initialized successfully")

        # Test di connessione
        try:
            result = supabase_client.table("bulk_redeem_requests").select("*").limit(1).execute()
            logger.info("✅ Supabase connection test successful")
        except Exception as test_error:
            logger.warning(f"⚠️ Supabase connection test failed: {test_error}")

    except Exception as e:
        logger.warning(f"❌ Supabase initialization failed: {e}")
else:
    logger.warning("❌ Supabase credentials not provided - running in limited mode")
    
# Variabili globali per i worker
worker_queue = queue.Queue()
active_workers = {}
worker_results = {}

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
        
        try:
            session = ort.InferenceSession(model_path)
            logger.info("✅ ONNX model loaded successfully (no explicit providers)")
        except Exception as e:
            logger.warning(f"First attempt failed: {e}")
            logger.info("🔄 Trying with explicit CPU provider...")
            session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            logger.info("✅ ONNX model loaded successfully (with CPU provider)")
        
        logger.info("🔄 Loading model metadata...")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        logger.info("✅ Model metadata loaded successfully")
        
        # Test del modello
        logger.info("🔄 Testing model inference...")
        if 'input_shape' in metadata:
            input_shape = metadata['input_shape']
            logger.info(f"📐 Testing with input_shape: {input_shape}")
            
            # Determina dimensioni per il test
            if len(input_shape) >= 3:
                height = input_shape[-2] if len(input_shape) >= 2 else 64
                width = input_shape[-1] if len(input_shape) >= 2 else 128
            else:
                height, width = 64, 128
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

# Definisci variabili globali CON CONTROLLO DELLE DIMENSIONI
if model_session and model_metadata:
    logger.info("🎉 Application started successfully with model loaded!")
    
    char_set = model_metadata.get('chars', 'abcdefghijklmnopqrstuvwxyz0123456789')
    idx_to_char = model_metadata.get('idx_to_char', {})
    
    # CONTROLLO SICURO delle dimensioni dell'input
    if 'input_shape' in model_metadata:
        input_shape = model_metadata['input_shape']
        logger.info(f"📐 Raw input_shape: {input_shape}")
        logger.info(f"📐 Input shape length: {len(input_shape)}")
        
        # Estrai dimensioni in modo sicuro
        if len(input_shape) == 4:
            # Formato: [batch, channels, height, width]
            expected_channels = input_shape[1]
            expected_height = input_shape[2]
            expected_width = input_shape[3]
        elif len(input_shape) == 3:
            # Formato: [channels, height, width] 
            expected_channels = input_shape[0]
            expected_height = input_shape[1]
            expected_width = input_shape[2]
        elif len(input_shape) == 2:
            # Formato: [height, width]
            expected_channels = 1  # Assume grayscale
            expected_height = input_shape[0]
            expected_width = input_shape[1]
        else:
            # Formato sconosciuto, usa default
            logger.warning(f"Unknown input_shape format: {input_shape}")
            expected_channels = 1
            expected_height = 64
            expected_width = 128
    else:
        # Se non c'è input_shape nel metadata, usa valori di default
        logger.warning("No input_shape in metadata, using defaults")
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
    
    def solve_captcha_for_wos(self, player_id):
        """Risolvi il captcha per WOS usando il modello esistente"""
        try:
            timestamp = str(int(time.time() * 1000))
            data_to_encode = {
                "fid": player_id,
                "time": timestamp,
                "init": "0"
            }
            
            encoded_data = self.encode_wos_data(data_to_encode)
            
            # Carica il captcha dall'API WOS
            response = requests.post(
                "https://wos-giftcode-api.centurygame.com/api/captcha",
                headers={
                    "accept": "application/json, text/plain, */*",
                    "content-type": "application/x-www-form-urlencoded",
                    "origin": "https://wos-giftcode.centurygame.com"
                },
                data=encoded_data
            )
            
            captcha_data = response.json()
            
            if captcha_data.get("msg") == "SUCCESS" and captcha_data.get("data", {}).get("img"):
                base64_image = captcha_data["data"]["img"]
                
                # Usa la funzione esistente per risolvere il captcha
                captcha_code, confidence = solve_captcha_internal(base64_image)
                logger.info(f"Solved captcha for player {player_id}: {captcha_code} (confidence: {confidence:.3f})")
                return captcha_code
            else:
                raise Exception(f"Failed to load captcha: {captcha_data.get('msg')}")
                
        except Exception as e:
            logger.error(f"Error in captcha process for player {player_id}: {e}")
            # Fallback
            captcha_code, _ = generate_fallback_captcha()
            return captcha_code
    
    def redeem_gift_code_for_player(self, player_id):
        """Riscatta il codice regalo per un singolo giocatore"""
        try:
            # Step 1: Risolvi il captcha
            captcha_code = self.solve_captcha_for_wos(player_id)
            
            # Step 2: Riscatta il codice regalo
            timestamp = str(int(time.time() * 1000))
            redeem_data = {
                "fid": player_id,
                "cdk": self.gift_code,
                "captcha_code": captcha_code,
                "time": timestamp
            }
            
            encoded_redeem_data = self.encode_wos_data(redeem_data)
            
            response = requests.post(
                "https://wos-giftcode-api.centurygame.com/api/gift_code",
                headers={
                    "accept": "application/json, text/plain, */*",
                    "content-type": "application/x-www-form-urlencoded",
                    "origin": "https://wos-giftcode.centurygame.com"
                },
                data=encoded_redeem_data
            )
            
            result_data = response.json()
            logger.info(f"Redeem result for player {player_id}: {result_data}")
            
            # Interpreta il risultato
            if result_data.get("msg") == "SUCCESS":
                return {
                    'player_id': player_id,
                    'success': True,
                    'message': 'Codice riscattato con successo',
                    'timestamp': datetime.now().isoformat()
                }
            elif result_data.get("msg") == "RECEIVED." and result_data.get("err_code") == 40008:
                return {
                    'player_id': player_id,
                    'success': True,
                    'message': 'Codice già riscattato precedentemente',
                    'timestamp': datetime.now().isoformat()
                }
            elif result_data.get("msg") == "CDK NOT FOUND." and result_data.get("err_code") == 40014:
                return {
                    'player_id': player_id,
                    'success': False,
                    'error': 'Codice regalo non valido',
                    'timestamp': datetime.now().isoformat()
                }
            elif result_data.get("msg") == "TIME ERROR." and result_data.get("err_code") == 40007:
                return {
                    'player_id': player_id,
                    'success': False,
                    'error': 'Codice regalo scaduto',
                    'timestamp': datetime.now().isoformat()
                }
            elif result_data.get("msg") == "CAPTCHA ERROR.":
                return {
                    'player_id': player_id,
                    'success': False,
                    'error': f'Errore captcha (soluzione: {captcha_code})',
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'player_id': player_id,
                    'success': False,
                    'error': result_data.get("msg", "Errore sconosciuto"),
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error redeeming gift code for player {player_id}: {e}")
            return {
                'player_id': player_id,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def update_supabase_progress(self):
        """Aggiorna il progresso nel database Supabase"""
        if not supabase_client:
            # Log solo ogni 10 progressi per non inondare i log
            if self.progress % 10 == 0 or self.progress == self.total:
                logger.info(f"📊 Progress: {self.progress}/{self.total} (Supabase not available)")
            return
            
        try:
            result = supabase_client.table("bulk_redeem_requests").update({
                "progress_current": self.progress,
                "progress_total": self.total,
                "updated_at": datetime.now().isoformat()
            }).eq("id", self.record_id).execute()
            
            if hasattr(result, 'error') and result.error:
                logger.error(f"Supabase update error: {result.error}")
                
        except Exception as e:
            logger.error(f"Error updating Supabase progress: {e}")
    
    def run(self):
        """Esegue il worker per il riscatto bulk"""
        self.status = "running"
        self.start_time = datetime.now()
        active_workers[self.worker_id] = self
        
        try:
            # Aggiorna Supabase all'inizio (se disponibile)
            if supabase_client:
                try:
                    supabase_client.table("bulk_redeem_requests").update({
                        "status": "running",
                        "started_at": self.start_time.isoformat(),
                        "worker_id": self.worker_id,
                        "updated_at": self.start_time.isoformat()
                    }).eq("id", self.record_id).execute()
                    logger.info(f"✅ Updated Supabase record {self.record_id} to running")
                except Exception as e:
                    logger.error(f"❌ Failed to update Supabase record: {e}")
            
            # Processa ogni giocatore
            for i, player_id in enumerate(self.player_list):
                if self.status == "stopped":
                    logger.info(f"🛑 Worker {self.worker_id} stopped by user")
                    break
                
                logger.info(f"🔄 Processing player {i+1}/{self.total}: {player_id}")
                
                # Processa il giocatore
                result = self.redeem_gift_code_for_player(player_id)
                self.results.append(result)
                self.progress = i + 1
                
                # Aggiorna il progresso nel database
                self.update_supabase_progress()
                
                success_status = "✅" if result.get('success') else "❌"
                logger.info(f"{success_status} Player {player_id} processed: {result.get('message', result.get('error', 'Unknown'))}")
                
                # Piccola pausa per non sovraccaricare le API
                time.sleep(2)
            
            # Determina lo stato finale
            if self.status == "stopped":
                final_status = "stopped"
                logger.info(f"🛑 Worker {self.worker_id} completed as stopped")
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
            
            # Salva i risultati finali
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
            
            # Aggiorna Supabase con lo stato finale (se disponibile)
            if supabase_client:
                try:
                    supabase_client.table("bulk_redeem_requests").update({
                        "status": final_status,
                        "updated_at": self.end_time.isoformat()
                    }).eq("id", self.record_id).execute()
                    logger.info(f"✅ Updated Supabase record {self.record_id} to {final_status}")
                except Exception as e:
                    logger.error(f"❌ Failed to update final status in Supabase: {e}")
            
            # Rimuovi dai worker attivi
            active_workers.pop(self.worker_id, None)
            logger.info(f"🗑️ Worker {self.worker_id} removed from active workers")

def worker_manager():
    """Gestisce i worker in background"""
    while True:
        try:
            worker = worker_queue.get()
            if worker is None:  # Sentinel value per fermare il manager
                break
            worker.run()
            worker_queue.task_done()
        except Exception as e:
            logger.error(f"Error in worker manager: {e}")

# Avvia il manager dei worker in background
worker_thread = threading.Thread(target=worker_manager, daemon=True)
worker_thread.start()

def solve_captcha_internal(base64_image):
    """Funzione interna per risolvere captcha (usata dai worker)"""
    try:
        if model_session is None:
            return generate_fallback_captcha()
        
        processed_image = preprocess_image(base64_image)
        return predict_captcha(processed_image)
    except Exception as e:
        logger.error(f"Error in internal captcha solving: {e}")
        return generate_fallback_captcha()

# API Routes Esistenti (mantenute)

@app.route('/solve-captcha', methods=['POST', 'OPTIONS'])
def solve_captcha_endpoint():
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

# Nuove API per Bulk Redeem

@app.route('/api/start-bulk-redeem', methods=['POST'])
def start_bulk_redeem():
    """Avvia un worker per il riscatto bulk"""
    try:
        data = request.get_json()
        
        if not data or 'players' not in data or 'gift_code' not in data:
            return jsonify({'error': 'Missing required fields: players and gift_code'}), 400
        
        player_list = data['players']
        gift_code = data['gift_code']
        record_id = data.get('record_id', str(uuid.uuid4()))  # Fallback se non fornito
        
        # Validazione
        if not isinstance(player_list, list) or len(player_list) == 0:
            return jsonify({'error': 'Invalid player list'}), 400
        
        if len(player_list) > 100:
            return jsonify({'error': 'Maximum 100 players allowed'}), 400
        
        # Crea nuovo worker
        worker_id = str(uuid.uuid4())
        worker = BulkRedeemWorker(record_id, player_list, gift_code, worker_id)
        
        # Aggiungi alla coda
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
    """Ottieni lo stato di un worker"""
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
    """Ferma un worker"""
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
    """Lista di tutti i worker attivi/completati"""
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
    """Endpoint per health check"""
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