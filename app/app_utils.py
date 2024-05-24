import os
import json
import pandas as pd
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

def insert_update_train_status(new_row,table_name):
    conn, cur, engine = get_connection()
    shop_value = new_row['shop'][0]
    existing_row = pd.read_sql_query(f"SELECT * FROM {table_name} WHERE shop = '{shop_value}'", engine)

    if existing_row.empty:
        new_row.to_sql(table_name, engine, if_exists='append', index=False)
    else:
        update_columns = [col for col in new_row.columns if col != 'shop']
        update_values = {col: new_row[col].values[0] for col in update_columns}
        update_query = f"""UPDATE {table_name} SET {', '.join([f"{col}='{update_values[col]}'" for col in update_columns])}""" + f" WHERE shop='{shop_value}'"
        cur.execute(update_query)
    cut_connection(conn, cur)

def insert_or_update_row_secrets(new_row, table_name):
    conn, cur, engine = get_connection()
    shop_value = new_row['shop'][0]
    existing_row = pd.read_sql_query(f"SELECT * FROM {table_name} WHERE shop = '{shop_value}'", engine)

    if existing_row.empty:
        new_row.to_sql(table_name, engine, if_exists='append', index=False)
    else:
        update_columns = [col for col in new_row.columns if col != 'shop']
        update_values = {col: new_row[col].values[0] for col in update_columns}
        update_query = f"""UPDATE {table_name} SET {', '.join([f"{col}='{update_values[col]}'" for col in update_columns])}""" + f" WHERE shop='{shop_value}'"
        cur.execute(update_query)
    cut_connection(conn, cur)

def insert_or_update_row_finetune_response(new_row, table_name):
    conn, cur, engine = get_connection()
    shop_value = new_row['shop'][0]
    query = new_row['query'][0]
    existing_row = pd.read_sql_query(f"SELECT * FROM {table_name} WHERE shop = '{shop_value}' and query = '{query}'", engine)

    if existing_row.empty:
        new_row.to_sql(table_name, engine, if_exists='append', index=False)
    else:
        update_columns = [col for col in new_row.columns if col != 'shop']
        update_values = {col: new_row[col].values[0] for col in update_columns}
        update_query = f"""UPDATE {table_name} SET {', '.join([f"{col}='{update_values[col]}'" for col in update_columns])}""" + f" WHERE shop='{shop_value}' and query = '{query}'" 
        cur.execute(update_query)
    cut_connection(conn, cur)

def insert_or_update_row_products(new_df, table_name):
    conn, cur, engine = get_connection()

    for index, row in new_df.iterrows():
        id_value = row['id']
        existing_df = pd.read_sql_query(f"SELECT * FROM {table_name} WHERE id = '{id_value}'", engine)

        if existing_df.empty:
            row_dict = row.to_dict()
            columns = ', '.join(row_dict.keys())
            values = ', '.join([f"'{value}'" for value in row_dict.values()])
            insert_query = f"INSERT INTO {table_name} ({columns}) VALUES ({values})"
            cur.execute(insert_query)
        else:
            update_columns = [col for col in row.index if col != 'id']
            update_values = {}
            for col in update_columns:
                value = row[col]
                if isinstance(value, list):
                    value = '{' + ','.join(map(str, value)) + '}'
                else:
                    value = f"'{str(value)}'"
                update_values[col] = value

            update_query = f"UPDATE {table_name} SET {', '.join([f'{col}={update_values[col]}' for col in update_columns])}"
            update_query += f" WHERE id='{id_value}'"

            cur.execute(update_query)

    conn.commit()
    cut_connection(conn, cur)