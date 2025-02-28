import sqlite3
from datetime import datetime
from pathlib import Path

class DatabaseManager:
    def __init__(self, reset_db=False):
        self.db_path = Path("data") / "scraped_urls.db"
        self.db_path.parent.mkdir(exist_ok=True)
        
        if reset_db:
            print("Resetting database...")
            self._reset_db()
            print("Database reset complete")
        else:
            print("Connecting to existing database...")
            
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()

    def _reset_db(self):
        if self.db_path.exists():
            try:
                self.db_path.unlink()
            except PermissionError:
                print("Database file locked, creating new database...")
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                self.db_path = self.db_path.with_name(f"scraped_urls_{timestamp}.db")

    def _init_db(self):
        with self.conn:
            self.conn.execute('''CREATE TABLE IF NOT EXISTS urls
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          url TEXT UNIQUE)''')
            self.conn.execute('''CREATE TABLE IF NOT EXISTS seo_content
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          url_id INTEGER,
                          content TEXT,
                          FOREIGN KEY(url_id) REFERENCES urls(id))''')

    def save_url(self, url):
        try:
            with self.conn:
                cursor = self.conn.cursor()
                cursor.execute('''INSERT OR IGNORE INTO urls 
                               (url) VALUES (?)''', (url,))
                return cursor.lastrowid
        except sqlite3.IntegrityError:
            return None  # Already exists
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            return None

    def save_seo_content(self, url_id, content):  # Reverted to url_id
        try:
            with self.conn:
                self.conn.execute('''INSERT INTO seo_content 
                                  (url_id, content) VALUES (?, ?)''', 
                                  (url_id, content))  # Reverted to url_id
        except sqlite3.OperationalError as e:
            print(f"Database error: {e}")
            self.conn.rollback() 