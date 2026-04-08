import os

from dotenv import load_dotenv                                                                                              
import logging                                                      
logging.disable(logging.CRITICAL)                                   
load_dotenv()  # Load environment variables from .env file 
import psycopg2
import psycopg2.extras
from skills.rag_query_engine import answer_question

""" result = answer_question(
      question='what is a deprecition area in SAP',
      table='sap-joule-knowledge-base',
      top_k=5
)

print('Answer:', result.answer) """


 

conn = psycopg2.connect(
       host='localhost', 
       port=5434,
       dbname=os.getenv("PG_DB"),
       user=os.getenv("PG_USER"),
       password=os.getenv("PG_PASSWORD")       
   )
cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

# Query — no embedding, just text search        
cur.execute('''
       SELECT * FROM "sap-joule-knowledge-base"
       LIMIT 5;
   ''', ('%asset class%',))

rows = cur.fetchall()
print(f'Found {len(rows)} rows')
print()
for row in rows:
    print(row['source'], row['file_name'])
    print(row['content'][:20], '...')
    print('uuid:', row['uuid'])

cur.close()
conn.close()