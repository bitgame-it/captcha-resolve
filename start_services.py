import subprocess
import sys
import os
from threading import Thread
import time
import logging

# Configura logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        os.system("python discord_bot.py")
    except Exception as e:
        logger.error(f"Error starting Discord bot: {e}")

if __name__ == "__main__":
    logger.info("🎯 Starting all services...")
    
    # Avvia Flask in un thread
    flask_thread = Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()
    
    logger.info("✅ Flask API started")
    
    # Aspetta che Flask sia pronto
    time.sleep(10)
    
    # Avvia Discord bot nel thread principale
    logger.info("✅ Starting Discord monitor...")
    run_discord_bot()