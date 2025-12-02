import subprocess
import sys
import os
from threading import Thread
import time
import logging
import requests
import asyncio

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
        port = os.environ.get('PORT', '10000')
        os.system(f"gunicorn app:app -b 0.0.0.0:{port} --access-logfile - --error-logfile -")
    except Exception as e:
        logger.error(f"Error starting Flask: {e}")

def run_discord_bot():
    """Avvia il bot Discord"""
    try:
        logger.info("🤖 Starting Discord Bot...")
        
        # Imposta l'URL dell'API basato sulla porta di Railway
        port = os.environ.get('PORT', '10000')
        os.environ['API_BASE_URL'] = f"http://localhost:{port}"
        
        os.system("python discord_bot.py")
    except Exception as e:
        logger.error(f"Error starting Discord bot: {e}")

def run_scraper_periodically(scraper_type, interval_hours=6):
    """Esegue uno scraper periodicamente ogni X ore"""
    try:
        logger.info(f"🔄 Starting {scraper_type} scraper (every {interval_hours} hours)...")
        
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            logger.error("❌ Missing Supabase credentials")
            return
        
        # Importa lo scraper appropriato
        if scraper_type == "wiki":
            from wiki_scraper import WikiGiftCodeScraper
            scraper = WikiGiftCodeScraper(supabase_url, supabase_key)
        elif scraper_type == "pl":
            from pl_scraper import PlRedeemScraper
            scraper = PlRedeemScraper(supabase_url, supabase_key)
        else:
            logger.error(f"❌ Unknown scraper type: {scraper_type}")
            return
        
        # Loop infinito con intervallo fisso
        while True:
            try:
                logger.info(f"🚀 Starting {scraper_type} scraping cycle...")
                success = scraper.run_scraping()
                if success:
                    logger.info(f"✅ {scraper_type.upper()} scraping completed successfully")
                else:
                    logger.error(f"❌ {scraper_type.upper()} scraping failed")
            except Exception as e:
                logger.error(f"❌ Error in {scraper_type} scraping cycle: {e}")
            
            # Aspetta 6 ore prima della prossima esecuzione
            logger.info(f"⏰ {scraper_type.upper()} waiting {interval_hours} hours for next scraping...")
            time.sleep(interval_hours * 60 * 60)
            
    except Exception as e:
        logger.error(f"❌ Error starting {scraper_type} scraper: {e}")

if __name__ == "__main__":
    logger.info("🎯 Starting all services...")
    
    # Ottieni la porta di Railway
    railway_port = os.environ.get('PORT', '10000')
    logger.info(f"🔧 Railway PORT: {railway_port}")
    
    # Avvia Flask in un thread
    flask_thread = Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()
    
    logger.info("✅ Flask API starting in background...")
    
    # Aspetta che l'API sia pronta (fino a 60 secondi)
    api_ready = wait_for_api(railway_port, max_retries=12, retry_delay=5)
    
    if api_ready:
        # Avvia Discord bot in un thread
        discord_thread = Thread(target=run_discord_bot)
        discord_thread.daemon = True
        discord_thread.start()
        logger.info("✅ Discord bot started in background...")
        
        # Avvia Wiki scraper ogni 6 ore
        wiki_thread = Thread(target=run_scraper_periodically, args=("wiki", 6))
        wiki_thread.daemon = True
        wiki_thread.start()
        logger.info("✅ Wiki scraper started (every 6 hours)...")
        
        # Avvia PL scraper ogni 6 ore
        pl_thread = Thread(target=run_scraper_periodically, args=("pl", 6))
        pl_thread.daemon = True
        pl_thread.start()
        logger.info("✅ PL scraper started (every 6 hours)...")
        
        # Mantieni il processo principale attivo
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            logger.info("👋 Shutting down...")
    else:
        logger.error("❌ Failed to start services - API not ready")
        logger.info("💡 Continuing with Flask API only...")
        
        # Mantieni il processo in esecuzione per Flask
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            logger.info("👋 Shutting down...")