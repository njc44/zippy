import os
import json
import openai
import numpy as np
from dotenv import load_dotenv
from search import * 
from app_utils import *
import psycopg2
from sqlalchemy import create_engine

load_dotenv('/app/app/.env')
client=openai.OpenAI()

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2.T)  # Transpose vec2 to match the shape of vec1
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

def dense_vector_formating(vector_data): 
    rough_list = vector_data.split(',')
    for idx,vd in enumerate(rough_list):
        if (idx == 0):
            rough_list[idx] = float(vd[1:])
        elif (idx == len(rough_list)-1):
            rough_list[idx] = float(vd[:-1])
        else:
            rough_list[idx] = float(vd)
    return np.array(rough_list)

async def create_response(query, response, shop, action='NA', expiry_date='NA'):

    try:
        response_emb = client.embeddings.create(
            input=query,
            model="text-embedding-3-large"
        )

        embedding = response_emb.data[0].embedding

        new_row = pd.DataFrame({'query':[query], 'response':[response], 'embedding':[embedding], 'expiry_date':[expiry_date], 'action':[action], 'shop':[shop]})
        conn, cur, engine = get_connection()
        table_name = 'finetune_response'
        insert_or_update_row_finetune_response(new_row, table_name)
        # new_row.to_sql(table_name, engine, if_exists='append', index=False)
        cut_connection(conn, cur)

        return True
    except Exception as e:
        print("Error create_response:", e)
        return False


def check_similar_queries(new_query, shop):
    conn, cur, _ = get_connection()
    query = f"SELECT * FROM finetune_response where shop = '{shop}'"
    finetune_response_df = pd.read_sql_query(query, conn)
    cut_connection(conn, cur)

    vector_data = []
    vector_metadata = []
    for data_point in finetune_response_df.iloc():
        vector_data.append(np.array(data_point['embedding']))
        vector_metadata.append({'query':data_point['query'],'response':data_point['response'],'expiry_date':data_point['expiry_date'],'action':data_point['action']})

    vector_data = np.array(vector_data)
    vector_metadata = np.array(vector_metadata)

    response_emb = client.embeddings.create(
        input = new_query,
        model="text-embedding-3-large"
    )
    query_vector = np.array(response_emb.data[0].embedding)

    cosine_similarities = []
    for i in range(vector_data.shape[0]):
        vdi = dense_vector_formating(vector_data[i])
        cosine_sim = cosine_similarity(vdi, query_vector)
        cosine_similarities.append(cosine_sim)

    cosine_similarities = np.array(cosine_similarities)
    indices = np.where(np.array(cosine_similarities) > 0.4)[0]
    indices_sorted = sorted(indices, key=lambda x: cosine_similarities[x], reverse=True)
    top_indices = indices_sorted[:3]


    decision = 'NO'
    product_ids = []
    for idx in top_indices:
        response = vector_metadata[idx]['response']
        query = vector_metadata[idx]['query']
        action = vector_metadata[idx]['action']
        expiry_date = vector_metadata[idx]['expiry_date']
        product_ids = []
        if action == 'recommend_products':
            search_response = call_search_api(new_query, shop)
            if search_response != []:
                for idx, i in enumerate(search_response):
                    product_ids.append(i['node']['id'])
        system_prompt = f"""You are a sales agent on an ecommerce platform, the shop owner gave the instruction that if a user asks the query - {query}, it is to be responded with the response - {response}. Now a new user has asked a new query - {new_query}. \nWould it be appropriate to use the same response for this user's query as well? \nRespond 'YES' or 'NO'. (Your decision should be based on whether both the queries have the same intent as one another)"""

        messages = [{"role": "system","content": system_prompt}]
        response_gpt = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=messages,
            max_tokens=1024
            )
        decision = response_gpt.choices[0].message.content
        if decision == 'YES':
            return True,response,product_ids
        else:
            continue
    return False, '', product_ids