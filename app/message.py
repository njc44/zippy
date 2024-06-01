from fastapi import FastAPI, Request
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

load_dotenv()
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
            "description": "A json parseable list of exactly 2 search keywords/tags eg. [Red T-shirt, maroon tshirt]/[Coffee Mug, coffee cup] etc.",
          }
        },
        "required": ["keywords"],
      },
    }
  }
]

def record_message(message,role,action,actionData,user_id,shop):
    conn, cur, engine = get_connection()
    table_name = 'message'
    now = datetime.datetime.now()
    timestamp = now.isoformat(timespec='milliseconds') + 'Z'
    new_row = pd.DataFrame({'message' : [message], 'role': [role], 'action': [action], 'actiondata' : [json.dumps(actionData)], 'timestamp' : [timestamp], 'user_id' : [user_id], 'shop': [shop]})
    new_row.to_sql(table_name, engine, if_exists='append', index=False)
    cut_connection(conn, cur)
    return timestamp

def get_return_value(reply,action,actionData,timestamp):
    return_value = {}
    return_value['message'] = reply
    return_value['role'] = "assistant"
    return_value['action'] = action
    return_value['actionData'] = actionData
    return_value['timestamp'] = timestamp
    return return_value

def add_to_cart(shop,user_id,actionData):
    product_title = get_product_title_from_variant(shop, actionData)
    reply = f"Great choice! I have added {product_title} to you cart. Happy Shopping!"
    timestamp = record_message(reply,'assistant','ADD_TO_CART',actionData,user_id,shop)
    return_value = get_return_value(reply,'ADD_TO_CART',actionData,timestamp)
    return return_value

async def response_generator(message,role,action,actionData,user_id,shop):
    record_message(message,role,action,actionData,user_id,shop)
    # if action == "SUGGEST_PRODUCT_VARIANTS":
    #     reply = ""
    #     actionData = get_product_variants(shop, actionData)
    #     timestamp = record_message(reply,'assistant',action,actionData,user_id,shop)
    #     return_value = get_return_value(reply,action,actionData,timestamp)
    #     return return_value
    
    # if action == "ADD_TO_CART":
    #     product_title = get_product_title_from_variant(shop, actionData)
    #     reply = f"Great choice! I have added {product_title} to you cart. Happy Shopping!"
    #     timestamp = record_message(reply,'assistant',action,actionData,user_id,shop)
    #     return_value = get_return_value(reply,action,actionData,timestamp)
    #     return return_value

    if action == "ADD_TO_CART":
      if 'productId' in actionData.keys():
          productVariants = get_product_variants(shop, actionData)
          if len(productVariants['variants']['nodes']) > 1:
              reply = f"We have found a few variants of the selected products, select the most relevant to add to cart!"
              timestamp = record_message(reply,'assistant','SUGGEST_PRODUCT_VARIANTS',actionData,user_id,shop)
              return_value = get_return_value(reply,'SUGGEST_PRODUCT_VARIANTS',actionData,timestamp)
              return return_value
          else:
              return add_to_cart(shop,user_id,{'variantId': productVariants['variants']['nodes'][0]['id']})
      if 'variantId' in actionData.keys():
          return add_to_cart(shop,user_id,actionData)

    # check_flag, reply, product_ids = check_similar_queries(message,shop)
    # if check_flag == True:
    #     return_value = {}
    #     return_value['response'] = reply
    #     return_value['product_ids'] = product_ids

    #     record_message(reply,'assistant',action,actionData,user_id,shop)
    #     return return_value


    conn, cur, engine = get_connection()
    system_prompt = f"""You are a sales agent on an e-commerce platform, your job is to reply to customer queries just as a real life sales agent would. 
    You will be given relevant info about the products and policies if and when required to be used to answer a query appropriately. Try to reply within 120 words."""

    query = f"""SELECT * FROM message
                where user_id = '{user_id}'
                and shop = '{shop}'
                order by timestamp desc"""
    df = pd.read_sql_query(query, conn)
    if len(df) == 0:
        df1 = df[:20]
        messages = [{"role": "user", "content": message}]
        for itr in df1.iloc():
            if (itr['role']!='epicai') and (itr['message']!=''):
                messages.append({"role": itr['role'], "content": itr['message']})
            
        messages.append({"role": "system","content": system_prompt})
        messages.reverse()

    else:
        messages = [{"role": "system","content": system_prompt}, 
                {"role": "user", "content": message}]
        
    response = client.chat.completions.create(
        model="gpt-4o",#gpt-3.5-turbo-0125
        messages=messages,
        tools=tools,
        tool_choice="auto",
        max_tokens=1024
        )

    search_response = []
    action = None
    reply = ""
        
    if type(response.choices[0].message.content) != str:
        if response.choices[0].message.tool_calls[0].function.name == "call_search_api":
                keywords = json.loads(response.choices[0].message.tool_calls[0].function.arguments)['keywords']
                search_response = call_search_api(keywords, shop)
                action = "SUGGEST_PRODUCTS"
                actionData = {'products':search_response}
    else:
        reply = response.choices[0].message.content

    timestamp = record_message(reply,"assistant",action,actionData,user_id,shop)
    cut_connection(conn, cur)

    return_value = get_return_value(reply,action,actionData,timestamp)
    return return_value

async def get_messages(shop, user_id):
    conn, cur, engine = get_connection()
    query = f"SELECT * FROM message where shop = '{shop}' and user_id = '{user_id}' order by timestamp desc"
    messages = pd.read_sql_query(query, conn)
    response = messages.to_dict(orient='records')
    cut_connection(conn, cur)
    response_json = {"messages":response}
    return response_json