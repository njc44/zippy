import os
import numpy as np
import pandas as pd
import requests
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import pickle
import json
import urllib.parse
import psycopg2
from sqlalchemy import create_engine
from app_utils import *


tokenizer = None
model = None

def process_search_queries(search_queries_list):
    encoded_search_queries_list = urllib.parse.quote('[' + ','.join(['"{}"'.format(item) for item in search_queries_list]) + ']')
    url = f"https://nathanjc-splade-v3-vector.hf.space/get_vector?search_queries_list={encoded_search_queries_list}"

    response = requests.get(url)

    if response.status_code == 200:
        return json.loads(response.json()['vector'])
    else:
        return None

def call_search_api(keywords, shop):
    conn, cur, _ = get_connection()
    query = f"""
            SELECT * FROM products 
            where shop = '{shop}'
            """
    product_df = pd.read_sql_query(query, conn)
    cut_connection(conn, cur)
    product_df['sparse_product_vector'] = product_df.apply(lambda x: json.loads(x['sparse_product_vector']),axis=1)

    search_queries = keywords
    search_vecs = process_search_queries(search_queries)
    search_vecs = np.array(list(filter(lambda x: x != None, search_vecs)))
    product_vecs = np.array(product_df['sparse_product_vector'])
    sim = np.zeros((search_vecs.shape[0], product_vecs.shape[0]))

    for i, search_vec in enumerate(search_vecs):
        for j, product_vec in enumerate(product_vecs):    
            sim[i][j] = np.dot(product_vec, search_vec) / (np.linalg.norm(product_vec) * np.linalg.norm(search_vec))
    
    top_3_1st = sorted(enumerate(sim[0]), key=lambda x: x[1], reverse=True)[:3]
    top_3_1st = np.array(product_df['id'])[np.array([i[0] for i in top_3_1st])]

    
    top_2_2nd = sorted(enumerate(sim[1]), key=lambda x: x[1], reverse=True)[:2]
    top_2_2nd = np.array(product_df['id'])[np.array([i[0] for i in top_2_2nd])]

    top_5 = np.array(list(set(np.concatenate([top_3_1st,top_2_2nd]))))

    response = []
    conn, cur, _ = get_connection()
    query = f"SELECT * FROM secrets where shop = '{shop}'"
    secret_df = pd.read_sql_query(query, conn).reset_index(drop=True)
    cut_connection(conn, cur)
    headers = {"X-Shopify-Access-Token": secret_df["shopify_access_token"][0]}
    request = requests.get(f"https://{shop}/admin/api/2024-04/products.json?ids={','.join(top_5)}", headers=headers).json()
    for json_request in request["products"]:
        response.append({
            "id": f"gid://shopify/Product/{json_request['id']}",
            "title": json_request["title"],
            "featuredImage": {
                "url": json_request["image"]["src"]
            }
        })

    return response