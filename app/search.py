import os
import numpy as np
import pandas as pd
import requests
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import pickle
import json
import urllib.parse


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
    product_df = pd.read_parquet('/app/app/data_files/product_df.parquet')
    product_df = product_df[product_df['shop']==shop].reset_index(drop=True)

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
    for id in top_5:
        headers = {
        "X-Shopify-Access-Token": os.getenv("Shopify-Access-Token"),
        }
        request = requests.get(f"https://{shop}/admin/api/2024-04/products/{id}.json", headers=headers).json()
        response.append({'node': {'id': f'gid://shopify/Product/{id}','title': request['product']['title']}})

    return response