import os
import numpy as np
import pandas as pd
import requests
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import pickle


tokenizer = None
model = None

def load_model_and_tokenizer():
    global tokenizer, model
    tokenizer_save_path = os.path.join('/app/app/model', 'tokenizer.pkl')
    with open(tokenizer_save_path, 'rb') as f:
        tokenizer = pickle.load(f)

        # Save the model as a pickle file
    model_save_path = os.path.join('/app/app/model', 'model.pkl')
    with open(model_save_path, 'rb') as f:
        model = pickle.load(f)

def process_search_queries(search_queries):
    global tokenizer, model
    if tokenizer is None or model is None:
        raise ValueError("Model and tokenizer have not been loaded yet. Call load_model_and_tokenizer() first.")
        load_model_and_tokenizer()

    search_tokens = tokenizer(
        search_queries, return_tensors='pt',
        padding=True, truncation=True
    )
    search_output = model(**search_tokens)
    # Aggregate the token-level vecs and transform to sparse
    search_vecs = torch.max(
        torch.log(1 + torch.relu(search_output.logits)) * search_tokens.attention_mask.unsqueeze(-1), dim=1
    )[0].squeeze().detach().cpu().numpy()

    return search_vecs

load_model_and_tokenizer()

def call_search_api(keywords, shop):
    product_df = pd.read_parquet('/app/app/data_files/product_df.parquet')
    product_df = product_df[product_df['shop']==shop].reset_index(drop=True)

    search_queries = keywords
    search_vecs = process_search_queries(search_queries)
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