# Initialize SQLite database
conn = sqlite3.connect('ai_memory.db')
cursor = conn.cursor()

def create_table():
    cursor.execute('''CREATE TABLE IF NOT EXISTS memory (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        user_input TEXT,
        ai_response TEXT
    )''')
    conn.commit()

def add_record(timestamp, user_input, ai_response):
    cursor.execute('INSERT INTO memory (timestamp, user_input, ai_response) VALUES (?, ?, ?)', (timestamp, user_input, ai_response))
    conn.commit()

def fetch_records():
    cursor.execute('SELECT * FROM memory')
    records = cursor.fetchall()
    return records

def main():
    create_table()
    add_record('2023-08-30 12:34:56', 'Hello, AI!', 'Hello, human!')
    records = fetch_records()
    print('AI Memory Records:')
    for record in records:
        print(record)

if __name__ == '__main__':
    main()