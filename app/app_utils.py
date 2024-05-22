import os
import json
from dotenv import load_dotenv
import psycopg2
from sqlalchemy import create_engine

load_dotenv('/app/app/.env')

def get_connection():
    connection_string = os.getenv('DATABASE_URL')
    conn = psycopg2.connect(connection_string)
    cur = conn.cursor()
    engine = create_engine(connection_string)
    return conn, cur, engine

def cut_connection(conn, cur):
    conn.commit()
    cur.close()
    conn.close()