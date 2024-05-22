from fastapi import FastAPI
from fastapi.responses import JSONResponse
import random
import time
import openai
import os
from dotenv import load_dotenv
import requests
import json
import uvicorn
import datetime
import pandas as pd
import numpy as np
from finetune_response import check_similar_queries
from search import *
from app_utils import *
import psycopg2
from sqlalchemy import create_engine

load_dotenv('/app/app/.env')
client=openai.OpenAI()


tools = tools = [
  {
    "type": "function",
    "function": {
      "name": "call_search_api",
      "description": "Gets upto top 5 search results of products based of search keywords only if user query indicates intent to search for specific products",
      "parameters": {
        "type": "object",
        "properties": {
          "keywords": {
            "type": "string",
            "description": "A json parseable list of 2 search keywords/tags eg. [Red T-shirt, maroon tshirt]/[Coffee Mug, coffee cup] etc.",
          }
        },
        "required": ["keywords"],
      },
    }
  }
]


def response_generator(latest_user_message, user_id, shop):
    conn, cur, engine = get_connection()
    table_name = 'messages'
    check_flag, reply, product_ids = check_similar_queries(latest_user_message,shop)
    if check_flag == True:
        return_value = {}
        return_value['response'] = reply
        return_value['product_ids'] = product_ids

        now = datetime.datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
        new_row = pd.DataFrame({'user_id' : [user_id], 'user_message': [latest_user_message], 'assistant_message': [reply], 'shop' : [shop], 'timestamp' : [timestamp_str]})
        new_row.to_sql(table_name, engine, if_exists='append', index=False)
        cut_connection(conn, cur)
        return return_value



    system_prompt = f"""You are a sales agent on an e-commerce platform, your job is to reply to customer queries just as a real life sales agent would. 
    You will be given relevant info about the products and policies if and when required to be used to answer a query appropriately. Try to reply within 120 words."""
    
    query = f"""SELECT * FROM messages 
              where user_id = '{user_id}'
              and shop = '{shop}
              order by timestamp desc'"""
    df = pd.read_sql_query(query, conn)
    if len(df) == 0:
        df1 = df[:7]
        messages = [{"role": "user", "content": latest_user_message}]
        for itr in df1.iloc():
            messages.append({"role": "assistant", "content": itr['assistant']})
            messages.append({"role": "user", "content": itr['user']})
            
        messages.append({"role": "system","content": system_prompt})
        messages.reverse()

    else:
        messages = [{"role": "system","content": system_prompt}, 
                {"role": "user", "content": latest_user_message}]
        
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=messages,
        tools=tools,
        tool_choice="auto",
        max_tokens=1024
        )
    
    search_response = []
    reply = ""
    product_info = ""
    product_ids = []
        
    if type(response.choices[0].message.content) != str:
        if response.choices[0].message.tool_calls[0].function.name == "call_search_api":
                keywords = json.loads(response.choices[0].message.tool_calls[0].function.arguments)['keywords']
                search_response = call_search_api(keywords, shop)
        if search_response == []:
            reply = "Hey sorry, we don't have that item"
        else:
            for idx, i in enumerate(search_response):
                product_info += f'product_{idx}: '+i['node']['title']+'\n'
                product_ids.append(i['node']['id'])
            messages.append({"role": "system","content": f"Here are some products that surfaced from the customer query: \n{product_info} Try to recommend these to the customer, keep it short within 60 words or less if possible."})
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=messages,
                max_tokens=1024
                )
            reply = response.choices[0].message.content
    else:
        reply = response.choices[0].message.content
        
    return_value = {}
    return_value['response'] = reply
    return_value['product_ids'] = product_ids

    now = datetime.datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")

    new_row = pd.DataFrame({'user_id' : [user_id], 'user_message': [latest_user_message], 'assistant_message': [reply], 'shop' : [shop], 'timestamp' : [timestamp_str]})
    new_row.to_sql(table_name, engine, if_exists='append', index=False)
    cut_connection(conn, cur)

    return return_value