import os
import subprocess
import time

def main():
    port = os.environ.get('PORT', '10000')
    
    # Avvia Flask con gunicorn (Render preferisce gunicorn)
    cmd = f"gunicorn app:app -b 0.0.0.0:{port} --workers=1 --threads=4 --access-logfile - --error-logfile -"
    
    print(f"🚀 Starting Flask on port {port}...")
    print(f"📝 Command: {cmd}")
    
    # Esegui gunicorn (questo manterrà il processo attivo)
    os.system(cmd)

if __name__ == "__main__":
    main()