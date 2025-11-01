import subprocess
import sys
import os
from threading import Thread
import time
import logging
import requests

# Configura logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def wait_for_api(port, max_retries=12, retry_delay=5):
    """Attende che l'API Flask sia pronta"""
    logger.info(f"⏳ Waiting for API to be ready on port {port}...")
    
    for retry in range(max_retries):
        try:
            health_url = f"http://localhost:{port}/health"
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                logger.info("✅ API is ready and responding!")
                return True
        except Exception as e:
            if retry < max_retries - 1:
                logger.info(f"⏳ API not ready yet ({retry + 1}/{max_retries})...")
                time.sleep(retry_delay)
            else:
                logger.warning(f"⚠️ API not ready after {max_retries} attempts")
    
    return False

def run_flask():
    """Avvia il server Flask"""
    try:
        logger.info("🚀 Starting Flask API...")
        # Usa la porta di Railway
        port = os.environ.get('PORT', '5000')
        os.system(f"gunicorn app:app -b 0.0.0.0:{port} --access-logfile - --error-logfile -")
    except Exception as e:
        logger.error(f"Error starting Flask: {e}")

def run_discord_bot():
    """Avvia il bot Discord"""
    try:
        logger.info("🤖 Starting Discord Bot...")
        
        # Imposta l'URL dell'API basato sulla porta di Railway
        port = os.environ.get('PORT', '5000')
        os.environ['API_BASE_URL'] = f"http://localhost:{port}"
        
        os.system("python discord_bot.py")
    except Exception as e:
        logger.error(f"Error starting Discord bot: {e}")

if __name__ == "__main__":
    logger.info("🎯 Starting all services...")
    
    # Ottieni la porta di Railway
    railway_port = os.environ.get('PORT', '5000')
    logger.info(f"🔧 Railway PORT: {railway_port}")
    
    # Avvia Flask in un thread
    flask_thread = Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()
    
    logger.info("✅ Flask API starting in background...")
    
    # Aspetta che l'API sia pronta (fino a 60 secondi)
    api_ready = wait_for_api(railway_port, max_retries=12, retry_delay=5)
    
    if api_ready:
        # Avvia Discord bot nel thread principale
        logger.info("✅ Starting Discord monitor...")
        run_discord_bot()
    else:
        logger.error("❌ Failed to start Discord bot - API not ready")
        logger.info("💡 Continuing with Flask API only...")
        
        # Mantieni il processo in esecuzione per Flask
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            logger.info("👋 Shutting down...")