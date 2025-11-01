import requests
from bs4 import BeautifulSoup
import logging
from datetime import datetime, timedelta
from supabase import create_client
import os
import asyncio
import time
import re

# Configura logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WikiGiftCodeScraper:
    def __init__(self, supabase_url, supabase_key):
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self.supabase = create_client(supabase_url, supabase_key)
        self.wiki_url = "https://www.whiteoutsurvival.wiki/giftcodes/"
        
    def fetch_wiki_page(self):
        """Scarica la pagina wiki dei codici"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            logger.info(f"🌐 Fetching wiki page: {self.wiki_url}")
            response = requests.get(self.wiki_url, headers=headers, timeout=30)
            response.raise_for_status()
            
            logger.info("✅ Successfully fetched wiki page")
            return response.text
            
        except Exception as e:
            logger.error(f"❌ Error fetching wiki page: {e}")
            return None
    
    def parse_codes_from_html(self, html_content):
        """Estrae codici attivi e scaduti dall'HTML"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            active_codes = []
            expired_codes = []
            
            # Trova tutti i tag strong per identificare le sezioni
            strong_tags = soup.find_all('strong')
            
            for i, tag in enumerate(strong_tags):
                text = tag.get_text().strip()
                
                if "Active Codes:" in text:
                    logger.info("📋 Found Active Codes section")
                    # Cerca i codici attivi dopo questo tag
                    current = tag.parent.find_next_sibling()
                    while current and not current.find('strong', string=re.compile('Expired Codes:')):
                        if current.name == 'ul':
                            # Estrai codici dagli span dentro gli ul
                            code_spans = current.find_all('span', class_='code')
                            for span in code_spans:
                                code = span.get_text().strip()
                                if code and code not in active_codes:
                                    active_codes.append(code)
                                    logger.info(f"✅ Found active code: {code}")
                        current = current.find_next_sibling()
                
                elif "Expired Codes:" in text:
                    logger.info("📋 Found Expired Codes section")
                    # Cerca i codici scaduti dopo questo tag
                    current = tag.parent.find_next_sibling()
                    while current:
                        # Cerca codici negli span
                        code_spans = current.find_all('span', class_='code')
                        for span in code_spans:
                            code = span.get_text().strip()
                            if code and code not in expired_codes:
                                expired_codes.append(code)
                                logger.info(f"❌ Found expired code: {code}")
                        
                        # Cerca anche testo diretto nei div (per i codici senza span)
                        if current.name == 'div':
                            text_content = current.get_text().strip()
                            if text_content:
                                # Estrai potenziali codici (parole in maiuscolo/misto)
                                lines = text_content.split('\n')
                                for line in lines:
                                    line = line.strip()
                                    if line and not any(keyword in line.lower() for keyword in ['codes:', 'active', 'expired']):
                                        # Considera come codice se è abbastanza lungo e non contiene spazi
                                        if len(line) >= 5 and ' ' not in line and line not in expired_codes:
                                            expired_codes.append(line)
                                            logger.info(f"❌ Found expired code from text: {line}")
                        
                        # Fermati quando trovi il prossimo strong tag o fine sezione
                        next_strong = current.find_next('strong')
                        if next_strong:
                            break
                        current = current.find_next_sibling()
            
            # Verifica che i codici con pulsanti copy siano attivi
            all_code_spans = soup.find_all('span', class_='code')
            for span in all_code_spans:
                code = span.get_text().strip()
                if code and code not in active_codes and code not in expired_codes:
                    # Se ha un pulsante copy nelle vicinanze, probabilmente è attivo
                    copy_btn = span.find_next('button', class_='copy-btn')
                    if copy_btn:
                        active_codes.append(code)
                        logger.info(f"✅ Found active code with copy button: {code}")
            
            logger.info(f"📊 Parsing completed: {len(active_codes)} active, {len(expired_codes)} expired codes")
            return active_codes, expired_codes
            
        except Exception as e:
            logger.error(f"❌ Error parsing HTML: {e}")
            return [], []
    
    def get_existing_codes_from_db(self):
        """Recupera tutti i codici esistenti dal database"""
        try:
            result = self.supabase.table("gift_codes")\
                .select("giftcode, expiry_date, status")\
                .execute()
            
            existing_codes = {row['giftcode']: row for row in result.data}
            logger.info(f"📦 Found {len(existing_codes)} existing codes in database")
            return existing_codes
            
        except Exception as e:
            logger.error(f"❌ Error getting existing codes: {e}")
            return {}
    
    def calculate_expiry_date(self, gift_code):
        """Calcola una data di scadenza ragionevole per i nuovi codici"""
        # Per i codici dal wiki, assumiamo una scadenza di 30 giorni
        return (datetime.now() + timedelta(days=30)).replace(hour=23, minute=59, second=0)
    
    def update_database(self, active_codes, expired_codes):
        """Aggiorna il database con i nuovi codici"""
        try:
            existing_codes = self.get_existing_codes_from_db()
            current_time = datetime.now().isoformat()
            
            new_codes_added = 0
            codes_updated = 0
            codes_expired = 0
            
            # Aggiungi/aggiorna codici attivi
            for code in active_codes:
                if code in existing_codes:
                    # Codice già esistente - aggiorna se necessario
                    existing_code = existing_codes[code]
                    if existing_code.get('status') != 'active':
                        self.supabase.table("gift_codes")\
                            .update({
                                "status": "active",
                                "updated_at": current_time
                            })\
                            .eq("giftcode", code)\
                            .execute()
                        codes_updated += 1
                        logger.info(f"🔄 Updated code status to active: {code}")
                else:
                    # Nuovo codice - aggiungi
                    expiry_date = self.calculate_expiry_date(code)
                    self.supabase.table("gift_codes")\
                        .insert({
                            "giftcode": code,
                            "date": current_time,
                            "expiry_date": expiry_date.isoformat(),
                            "discord_author": "wiki_scraper",
                            "message_preview": "Added from wiki",
                            "status": "active",
                            "created_at": current_time,
                            "updated_at": current_time
                        })\
                        .execute()
                    new_codes_added += 1
                    logger.info(f"✅ Added new code from wiki: {code}")
            
            # Segna come scaduti i codici nella lista expired
            for code in expired_codes:
                if code in existing_codes:
                    existing_code = existing_codes[code]
                    if existing_code.get('status') != 'expired':
                        self.supabase.table("gift_codes")\
                            .update({
                                "status": "expired",
                                "updated_at": current_time
                            })\
                            .eq("giftcode", code)\
                            .execute()
                        codes_expired += 1
                        logger.info(f"🗑️ Marked code as expired: {code}")
            
            logger.info(f"🎯 Database update: {new_codes_added} new, {codes_updated} updated, {codes_expired} expired")
            return new_codes_added, codes_updated, codes_expired
            
        except Exception as e:
            logger.error(f"❌ Error updating database: {e}")
            return 0, 0, 0
    
    def run_scraping(self):
        """Esegue tutto il processo di scraping"""
        logger.info("🚀 Starting wiki gift code scraping...")
        
        html_content = self.fetch_wiki_page()
        if not html_content:
            logger.error("❌ Failed to fetch wiki page")
            return False
        
        active_codes, expired_codes = self.parse_codes_from_html(html_content)
        
        if not active_codes and not expired_codes:
            logger.warning("⚠️ No codes found on wiki page")
            return False
        
        new, updated, expired = self.update_database(active_codes, expired_codes)
        
        logger.info(f"✅ Wiki scraping completed: {len(active_codes)} active, {len(expired_codes)} expired codes processed")
        return True

async def run_wiki_scraper_periodically():
    """Esegue lo scraper ogni 23 ore"""
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        logger.error("❌ Missing Supabase credentials")
        return
    
    scraper = WikiGiftCodeScraper(supabase_url, supabase_key)
    
    while True:
        try:
            success = scraper.run_scraping()
            if success:
                logger.info("✅ Wiki scraping completed successfully")
            else:
                logger.error("❌ Wiki scraping failed")
        except Exception as e:
            logger.error(f"❌ Error in wiki scraping cycle: {e}")
        
        # Aspetta 23 ore prima della prossima esecuzione
        logger.info("⏰ Waiting 23 hours for next wiki scraping...")
        await asyncio.sleep(23 * 60 * 60)  # 23 ore

def main():
    """Avvia lo scraper immediatamente (per testing)"""
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        logger.error("❌ Missing Supabase credentials")
        return
    
    scraper = WikiGiftCodeScraper(supabase_url, supabase_key)
    scraper.run_scraping()

if __name__ == "__main__":
    main()