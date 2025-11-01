import discord
import asyncio
import re
from datetime import datetime, timedelta
import requests
import hashlib
import os
import logging
from supabase import create_client

# Configura logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DiscordGiftCodeMonitor:
    def __init__(self, discord_token, channel_id, supabase_url, supabase_key, api_base_url):
        self.discord_token = discord_token
        self.channel_id = channel_id
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self.api_base_url = api_base_url
        
        # Inizializza Supabase
        self.supabase = create_client(supabase_url, supabase_key)
        
        # Inizializza Discord client
        intents = discord.Intents.default()
        intents.message_content = True
        self.client = discord.Client(intents=intents)
        
        # Pattern per estrarre codice e data
        self.code_pattern = r"Code:\s*([A-Za-z0-9]+)"
        self.date_pattern = r"Valid Until:\s*([A-Za-z]+ \d{1,2}, \d{1,2}:\d{2}) \(UTC\+0\)"
        
        # Mappa mesi
        self.month_map = {
            'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
            'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
        }
    
    def parse_date(self, date_str):
        """Converte la stringa della data in datetime object"""
        try:
            # Esempio: "Oct 31, 23:59"
            match = re.match(r"([A-Za-z]+) (\d{1,2}), (\d{1,2}:\d{2})", date_str)
            if match:
                month_str, day_str, time_str = match.groups()
                month = self.month_map[month_str]
                day = int(day_str)
                
                # Parse time
                hour, minute = map(int, time_str.split(':'))
                
                # Crea datetime (assumiamo anno corrente)
                current_year = datetime.now().year
                return datetime(current_year, month, day, hour, minute)
        except Exception as e:
            logger.error(f"Error parsing date {date_str}: {e}")
        return None
    
    def is_code_expired(self, expiry_date):
        """Controlla se il codice è scaduto"""
        return datetime.now() > expiry_date
    
    def encode_wos_data(self, data):
        """Encoding per le richieste WOS API"""
        secret = "tB87#kPtkxqOS2"
        sorted_keys = sorted(data.keys())
        encoded_data = "&".join([f"{key}={data[key]}" for key in sorted_keys])
        sign = hashlib.md5((encoded_data + secret).encode()).hexdigest()
        return {**data, "sign": sign}
    
    def verify_gift_code(self, gift_code):
        """Verifica se un codice regalo è valido"""
        try:
            timestamp = str(int(datetime.now().timestamp()))
            test_data = {
                "fid": "398483320",  # ID di test
                "cdk": gift_code,
                "time": timestamp
            }
            
            encoded_data = self.encode_wos_data(test_data)
            
            headers = {
                "accept": "application/json, text/plain, */*",
                "content-type": "application/x-www-form-urlencoded",
                "origin": "https://wos-giftcode.centurygame.com",
            }
            
            session = requests.Session()
            session.headers.update(headers)
            
            body_params = "&".join([f"{key}={value}" for key, value in encoded_data.items()])
            
            response = session.post(
                "https://wos-giftcode-api.centurygame.com/api/gift_code",
                data=body_params,
                timeout=30
            )
            
            result_data = response.json()
            logger.info(f"Code verification result for {gift_code}: {result_data}")
            
            # Codice valido se non è scaduto e non è "CDK NOT FOUND"
            if result_data.get("msg") in ["SUCCESS", "RECEIVED", "SAME TYPE EXCHANGE"]:
                return True
            elif result_data.get("msg") == "TIME ERROR.":
                return False
            elif result_data.get("msg") == "CDK NOT FOUND.":
                return False
            else:
                return True  # Altri errori potrebbero essere temporanei
                
        except Exception as e:
            logger.error(f"Error verifying gift code {gift_code}: {e}")
            return True  # In caso di errore, assumiamo sia valido
    
    def extract_gift_code_info(self, message_content):
        """Estrae codice e data di scadenza dal messaggio"""
        code_match = re.search(self.code_pattern, message_content)
        date_match = re.search(self.date_pattern, message_content)
        
        if code_match and date_match:
            gift_code = code_match.group(1)
            date_str = date_match.group(1)
            expiry_date = self.parse_date(date_str)
            
            return {
                'gift_code': gift_code,
                'expiry_date': expiry_date,
                'date_str': date_str
            }
        return None
    
    async def check_channel_for_new_codes(self):
        """Controlla il canale Discord per nuovi codici regalo"""
        try:
            channel = self.client.get_channel(self.channel_id)
            if not channel:
                logger.error(f"Channel {self.channel_id} not found")
                return []
            
            # Cerca gli ultimi 50 messaggi
            messages = []
            async for message in channel.history(limit=50):
                messages.append(message)
            
            logger.info(f"Found {len(messages)} messages in channel")
            
            new_codes = []
            for message in messages:
                # Salta messaggi del bot
                if message.author.bot:
                    continue
                
                content = message.content
                code_info = self.extract_gift_code_info(content)
                
                if code_info:
                    gift_code = code_info['gift_code']
                    expiry_date = code_info['expiry_date']
                    
                    if not expiry_date:
                        logger.warning(f"Could not parse date for code {gift_code}")
                        continue
                    
                    # Controlla se il codice è scaduto
                    if self.is_code_expired(expiry_date):
                        logger.info(f"Code {gift_code} is expired, skipping")
                        continue
                    
                    # Controlla se il codice è già nel database
                    existing_code = self.supabase.table("gift_codes")\
                        .select("*")\
                        .eq("giftcode", gift_code)\
                        .execute()
                    
                    if not existing_code.data:
                        # Nuovo codice trovato!
                        new_codes.append({
                            'gift_code': gift_code,
                            'expiry_date': expiry_date.isoformat(),
                            'date_str': code_info['date_str'],
                            'message_id': message.id,
                            'author': str(message.author),
                            'message_content': content[:100]
                        })
                        logger.info(f"New gift code found: {gift_code} (expires: {code_info['date_str']})")
            
            return new_codes
            
        except Exception as e:
            logger.error(f"Error checking channel: {e}")
            return []
    
    def save_gift_code_to_db(self, code_info):
        """Salva il nuovo codice nel database"""
        try:
            data = {
                "giftcode": code_info['gift_code'],
                "date": datetime.now().isoformat(),
                "expiry_date": code_info['expiry_date'],
                "discord_message_id": code_info['message_id'],
                "discord_author": code_info['author'],
                "message_preview": code_info['message_content'],
                "status": "active"
            }
            
            result = self.supabase.table("gift_codes").insert(data).execute()
            logger.info(f"Saved gift code {code_info['gift_code']} to database")
            return True
            
        except Exception as e:
            logger.error(f"Error saving gift code to database: {e}")
            return False
    
    def get_active_players_from_supabase(self):
        """Recupera la lista di player attivi da TUTTI gli utenti Supabase"""
        try:
            # Cerca TUTTE le configurazioni di TUTTI gli utenti
            result = self.supabase.table("bulk_redeem_requests")\
                .select("player_list, user_id")\
                .is_("gift_code", "null")\
                .not_("player_list", "is", "null")\
                .execute()
            
            if result.data and len(result.data) > 0:
                # Combina TUTTI i player_list di TUTTI gli utenti
                all_player_ids = []
                user_count = len(set(record['user_id'] for record in result.data if record.get('user_id')))
                
                for record in result.data:
                    if record.get('player_list'):
                        all_player_ids.extend(record['player_list'])
                
                # Rimuovi duplicati (se lo stesso player ID è in più liste utente)
                unique_player_ids = list(set(all_player_ids))
                
                logger.info(f"Found {len(unique_player_ids)} unique players from {user_count} users")
                return unique_player_ids
            else:
                logger.warning("No player configurations found in database")
                return []
                
        except Exception as e:
            logger.error(f"Error getting active players: {e}")
            return []
    
    def start_bulk_redeem_worker(self, gift_code, player_list):
        """Avvia un worker per il riscatto bulk"""
        try:
            url = f"{self.api_base_url}/api/start-bulk-redeem"
            payload = {
                "players": player_list,
                "gift_code": gift_code,
                "record_id": str(hash(f"{gift_code}_{datetime.now().timestamp()}"))
            }
            
            response = requests.post(url, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Started bulk redeem worker for code {gift_code}: {result}")
                return result
            else:
                logger.error(f"Failed to start bulk redeem worker: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error starting bulk redeem worker: {e}")
            return None
    
    def check_and_restart_expired_workers(self):
        """Controlla e riavvia worker per codici ancora validi"""
        try:
            # Trova worker completati ieri che hanno codici ancora validi
            yesterday = (datetime.now() - timedelta(days=1)).isoformat()
            
            workers_result = self.supabase.table("bulk_redeem_requests")\
                .select("*, gift_codes!inner(*)")\
                .gte("updated_at", yesterday)\
                .eq("status", "completed")\
                .execute()
            
            for worker in workers_result.data:
                gift_code = worker['gift_codes']['giftcode']
                expiry_date_str = worker['gift_codes']['expiry_date']
                
                if expiry_date_str:
                    expiry_date = datetime.fromisoformat(expiry_date_str.replace('Z', '+00:00'))
                    
                    # Se il codice è ancora valido e il worker è vecchio di più di 12 ore
                    if not self.is_code_expired(expiry_date):
                        worker_updated = datetime.fromisoformat(worker['updated_at'].replace('Z', '+00:00'))
                        hours_since_update = (datetime.now() - worker_updated).total_seconds() / 3600
                        
                        if hours_since_update > 12:
                            logger.info(f"Restarting worker for still-valid code {gift_code}")
                            
                            player_list = self.get_active_players_from_supabase()
                            if player_list:
                                self.start_bulk_redeem_worker(gift_code, player_list)
                            
        except Exception as e:
            logger.error(f"Error checking expired workers: {e}")
    
    async def daily_check(self):
        """Esegue il check giornaliero"""
        logger.info("Starting daily gift code check...")
        
        # Step 1: Cerca nuovi codici nel canale Discord
        new_codes = await self.check_channel_for_new_codes()
        
        # Step 2: Salva nuovi codici e avvia worker
        for code_info in new_codes:
            if self.save_gift_code_to_db(code_info):
                player_list = self.get_active_players_from_supabase()
                if player_list:
                    logger.info(f"Starting automatic redeem for {len(player_list)} players with code {code_info['gift_code']}")
                    self.start_bulk_redeem_worker(code_info['gift_code'], player_list)
                else:
                    logger.warning(f"No player configuration found. Code {code_info['gift_code']} saved but no automatic redeem started.")
        
        # Step 3: Controlla e riavvia worker per codici ancora validi
        self.check_and_restart_expired_workers()
        
        logger.info("Daily check completed")
    
    async def start_monitoring(self):
        """Avvia il monitoraggio"""
        @self.client.event
        async def on_ready():
            logger.info(f'Bot is ready! Logged in as {self.client.user}')
            
            # Esegui il check immediatamente all'avvio
            await self.daily_check()
            
            # Poi esegui ogni 24 ore
            while True:
                await asyncio.sleep(24 * 60 * 60)  # 24 ore
                await self.daily_check()
        
        try:
            await self.client.start(self.discord_token)
        except Exception as e:
            logger.error(f"Error starting Discord client: {e}")

# Configurazione
def main():
    # Leggi le variabili d'ambiente
    DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
    DISCORD_CHANNEL_ID = int(os.getenv("DISCORD_GIFTCODE_CHANNEL_ID"))
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:5000")
    
    # Verifica configurazione
    if not all([DISCORD_TOKEN, DISCORD_CHANNEL_ID, SUPABASE_URL, SUPABASE_KEY]):
        logger.error("Missing required environment variables")
        return
    
    # Crea e avvia il monitor
    monitor = DiscordGiftCodeMonitor(
        discord_token=DISCORD_TOKEN,
        channel_id=DISCORD_CHANNEL_ID,
        supabase_url=SUPABASE_URL,
        supabase_key=SUPABASE_KEY,
        api_base_url=API_BASE_URL
    )
    
    # Avvia il monitoraggio
    asyncio.run(monitor.start_monitoring())

if __name__ == "__main__":
    main()