from fastapi import FastAPI, Request
import random
import time
import openai
import os
from dotenv import load_dotenv
import requests
import json
import uvicorn
import datetime
import numpy as np
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
from train import train
from message import response_generator, get_messages
from finetune_response import create_response
from app_utils import *

load_dotenv()
client=openai.OpenAI()

app = FastAPI()

@app.get('/')
def homepage():
    return json.dumps({'Message': 'Welcome to EpicAI API'})

@app.get('/health_check')
def health_check():
    return json.dumps({"success": True}), 200

@app.get('/train')
async def train_(request: Request):
    request_json = await request.json()
    shop = request_json.get('shop')
    shopify_storefront_access_token = request_json.get('shopify_storefront_access_token')
    shopify_access_token = request_json.get("shopify_access_token")

    training_status = await train(shop, shopify_storefront_access_token, shopify_access_token)
    return training_status

@app.get('/train/status')
async def train_(request: Request):
    request_json = await request.json()
    shop = request_json.get('shop')

    training_status = await get_train_status(shop)
    return training_status

@app.get('/message')
async def response_generator_(request: Request):
    request_json = await request.json()
    message = request_json.get('message')
    role = request_json.get('role')
    action = request_json.get('action')
    actionData = request_json.get('actionData')
    user_id = request_json.get('user_id')
    shop = request_json.get('shop')

    return await response_generator(message,role,action,actionData,user_id,shop)

@app.get('/messages')
async def messages(request: Request):
    request_json = await request.json()
    user_id = request_json.get('user_id')
    shop = request_json.get('shop')

    return await get_messages(shop, user_id)

@app.get('/finetune_response')
async def create_response_(request: Request):
    request_json = await request.json()
    query = request_json.get('query')
    response = request_json.get('response')
    shop = request_json.get('shop')
    expiry_date = 'NA'
    action = 'NA'

    create_response_status = await create_response(query, response, shop, expiry_date, action)
    return create_response_status

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=10000)