import requests
from bs4 import BeautifulSoup
import logging
from datetime import datetime, timedelta
from supabase import create_client
import os
import asyncio
import time
import uuid

# Configura logging (usa lo stesso del primo servizio)
logger = logging.getLogger(__name__)

class PlRedeemScraper:
    def __init__(self, supabase_url, supabase_key):
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self.supabase = create_client(supabase_url, supabase_key)
        self.redeem_url = "https://whiteoutsurvival.pl/redeem/"
        
    def fetch_redeem_page(self):
        """Scarica la pagina di redeem"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            logger.info(f"🌐 Fetching redeem page: {self.redeem_url}")
            response = requests.get(self.redeem_url, headers=headers, timeout=30)
            response.raise_for_status()
            
            logger.info("✅ Successfully fetched redeem page")
            return response.text
            
        except Exception as e:
            logger.error(f"❌ Error fetching redeem page: {e}")
            return None
    
    def parse_codes_from_html(self, html_content):
        """Estrae codici attivi e scaduti dall'HTML del sito PL"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            active_codes = []
            expired_codes = []
            
            # Trova il div principale che contiene i codici
            admin_codes_div = soup.find('div', id='admin-codes')
            if not admin_codes_div:
                logger.error("❌ Could not find admin-codes div")
                return [], []
            
            # Estrai codici attivi (button con classe admin-code-btn)
            active_codes_section = admin_codes_div.find('div', class_='admin-code-list active-codes')
            if active_codes_section:
                active_buttons = active_codes_section.find_all('button', class_='admin-code-btn')
                for button in active_buttons:
                    code = button.get('data-code', '').strip()
                    if code and code not in active_codes:
                        active_codes.append(code)
                        logger.info(f"✅ Found active code: {code}")
            
            # Estrai codici scaduti (span con classe admin-code-btn expired)
            expired_codes_section = admin_codes_div.find('div', class_='admin-code-list expired-codes')
            if expired_codes_section:
                expired_spans = expired_codes_section.find_all('span', class_='admin-code-btn expired')
                for span in expired_spans:
                    code = span.get('data-code', '').strip()
                    if code and code not in expired_codes:
                        expired_codes.append(code)
                        logger.info(f"❌ Found expired code: {code}")
            
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
        """Calcola una data di scadenza ragionevole per i nuovi codici attivi"""
        # Per i codici attivi dal sito PL, assumiamo una scadenza di 30 giorni
        return (datetime.now() + timedelta(days=30)).replace(hour=23, minute=59, second=0)
    
    def get_expired_date_2024(self):
        """Restituisce una data di scadenza nel 2024 per i codici scaduti"""
        return datetime(2024, 12, 31, 23, 59, 0)
    
    def get_active_players_from_supabase(self):
        """Recupera la lista di player attivi dalle configurazioni salvate"""
        try:
            result = self.supabase.table("bulk_redeem_requests")\
                .select("player_list, user_id")\
                .is_("gift_code", "null")\
                .eq("status", "saved")\
                .execute()
            
            if result.data:
                all_player_ids = []
                for record in result.data:
                    if record.get('player_list'):
                        all_player_ids.extend(record['player_list'])
                
                unique_player_ids = list(set(all_player_ids))
                logger.info(f"Found {len(unique_player_ids)} unique players for auto-redeem")
                return unique_player_ids
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error getting active players: {e}")
            return []
    
    def get_players_to_process(self, gift_code, all_players):
        """Filtra i player che non hanno già riscattato questo codice"""
        try:
            if not all_players:
                return []
            
            result = self.supabase.table("redeem_history")\
                .select("player_id")\
                .eq("gift_code", gift_code)\
                .eq("success", True)\
                .in_("player_id", all_players)\
                .execute()
            
            already_redeemed = {row['player_id'] for row in result.data}
            players_to_process = [p for p in all_players if p not in already_redeemed]
            
            logger.info(f"🎯 Players to process for {gift_code}: {len(players_to_process)}")
            return players_to_process
            
        except Exception as e:
            logger.error(f"Error filtering players: {e}")
            return all_players
    
    def start_bulk_redeem_worker(self, gift_code, player_list):
        """Avvia un worker per il riscatto bulk"""
        try:
            import requests
            
            api_base_url = os.getenv("API_BASE_URL", "http://localhost:5000")
            url = f"{api_base_url}/api/start-bulk-redeem"
            record_id = str(uuid.uuid4())
            
            payload = {
                "players": player_list,
                "gift_code": gift_code,
                "record_id": record_id
            }
            
            logger.info(f"🔄 Starting bulk redeem via API: {gift_code} for {len(player_list)} players")
            response = requests.post(url, json=payload, timeout=30)
            
            if response.status_code == 200:
                logger.info(f"✅ Bulk redeem started for {gift_code}")
                return True
            else:
                logger.error(f"❌ Failed to start bulk redeem: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error starting bulk redeem worker: {e}")
            return False
    
    def trigger_redeem_for_new_codes(self, new_active_codes):
        """Triggera il redeem automatico per i nuovi codici trovati"""
        try:
            if not new_active_codes:
                return
            
            logger.info(f"🚀 Triggering automatic redeem for {len(new_active_codes)} new codes")
            
            all_players = self.get_active_players_from_supabase()
            if not all_players:
                logger.warning("⏭️ No players found for auto-redeem")
                return
            
            for gift_code in new_active_codes:
                try:
                    players_to_process = self.get_players_to_process(gift_code, all_players)
                    if not players_to_process:
                        logger.info(f"⏭️ All players already redeemed {gift_code}, skipping")
                        continue
                    
                    success = self.start_bulk_redeem_worker(gift_code, players_to_process)
                    if success:
                        logger.info(f"✅ Started redeem worker for new code: {gift_code}")
                    else:
                        logger.error(f"❌ Failed to start redeem worker for: {gift_code}")
                    
                    time.sleep(1)
                    
                except Exception as e:
                    logger.error(f"❌ Error triggering redeem for {gift_code}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"❌ Error in trigger_redeem_for_new_codes: {e}")
    
    def update_database(self, active_codes, expired_codes):
        """Aggiorna il database con i nuovi codici e triggera il redeem"""
        try:
            existing_codes = self.get_existing_codes_from_db()
            current_time = datetime.now().isoformat()
            
            new_codes_added = 0
            codes_updated = 0
            codes_expired = 0
            new_active_codes = []  # Tiene traccia dei nuovi codici attivi
            
            # Aggiungi/aggiorna codici attivi
            for code in active_codes:
                if code in existing_codes:
                    # Codice già esistente - aggiorna se necessario
                    existing_code = existing_codes[code]
                    if existing_code.get('status') != 'active':
                        try:
                            self.supabase.table("gift_codes")\
                                .update({
                                    "status": "active",
                                    "updated_at": current_time
                                })\
                                .eq("giftcode", code)\
                                .execute()
                            codes_updated += 1
                            logger.info(f"🔄 Updated code status to active: {code}")
                        except Exception as e:
                            logger.error(f"❌ Error updating code {code}: {e}")
                else:
                    # Nuovo codice - aggiungi
                    expiry_date = self.calculate_expiry_date(code)
                    try:
                        self.supabase.table("gift_codes")\
                            .insert({
                                "giftcode": code,
                                "date": current_time,
                                "expiry_date": expiry_date.isoformat(),
                                "discord_author": "pl_redeem_scraper",
                                "message_preview": "Added from PL redeem site",
                                "status": "active"
                            })\
                            .execute()
                        new_codes_added += 1
                        new_active_codes.append(code)  # Aggiungi alla lista dei nuovi
                        logger.info(f"✅ Added new code from PL site: {code}")
                    except Exception as e:
                        logger.error(f"❌ Error adding code {code}: {e}")
            
            # Gestisci codici scaduti
            for code in expired_codes:
                if code in existing_codes:
                    existing_code = existing_codes[code]
                    # Se il codice esiste ma non è marcato come scaduto, aggiornalo
                    if existing_code.get('status') != 'expired':
                        try:
                            expired_date = self.get_expired_date_2024()
                            self.supabase.table("gift_codes")\
                                .update({
                                    "status": "expired",
                                    "expiry_date": expired_date.isoformat(),
                                    "updated_at": current_time
                                })\
                                .eq("giftcode", code)\
                                .execute()
                            codes_expired += 1
                            logger.info(f"🗑️ Marked code as expired with 2024 date: {code}")
                        except Exception as e:
                            logger.error(f"❌ Error expiring code {code}: {e}")
                else:
                    # Se il codice scaduto non esiste nel database, aggiungilo come scaduto
                    try:
                        expired_date = self.get_expired_date_2024()
                        self.supabase.table("gift_codes")\
                            .insert({
                                "giftcode": code,
                                "date": current_time,
                                "expiry_date": expired_date.isoformat(),
                                "discord_author": "pl_redeem_scraper",
                                "message_preview": "Added as expired from PL site",
                                "status": "expired"
                            })\
                            .execute()
                        new_codes_added += 1
                        logger.info(f"📝 Added expired code from PL site: {code}")
                    except Exception as e:
                        logger.error(f"❌ Error adding expired code {code}: {e}")
            
            logger.info(f"🎯 Database update: {new_codes_added} new, {codes_updated} updated, {codes_expired} expired")
            
            # ⚡ TRIGGER AUTOMATICO: se ci sono nuovi codici, avvia il redeem
            if new_active_codes:
                logger.info(f"🚀 Found {len(new_active_codes)} new codes, triggering automatic redeem...")
                self.trigger_redeem_for_new_codes(new_active_codes)
            
            return new_codes_added, codes_updated, codes_expired
            
        except Exception as e:
            logger.error(f"❌ Error updating database: {e}")
            return 0, 0, 0
    
    def run_scraping(self):
        """Esegue tutto il processo di scraping"""
        logger.info("🚀 Starting PL redeem code scraping...")
        
        html_content = self.fetch_redeem_page()
        if not html_content:
            logger.error("❌ Failed to fetch redeem page")
            return False
        
        active_codes, expired_codes = self.parse_codes_from_html(html_content)
        
        if not active_codes and not expired_codes:
            logger.warning("⚠️ No codes found on redeem page")
            return False
        
        new, updated, expired = self.update_database(active_codes, expired_codes)
        
        logger.info(f"✅ PL redeem scraping completed: {len(active_codes)} active, {len(expired_codes)} expired codes processed")
        return True