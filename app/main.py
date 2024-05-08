from fastapi import FastAPI
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
from message import response_generator
from finetune_response import create_response

load_dotenv('/app/app/.env')
client=openai.OpenAI()

app = FastAPI()

@app.get('/')
def homepage():
    return json.dumps({'Message': 'Welcome to EpicAI API'})

@app.get('/health_check')
def health_check():
    return json.dumps({"success": True}), 200

@app.get('/train')
def train_(shop):
    training_status = train(shop)
    return training_status

@app.get('/message')
def response_generator_(latest_user_message, user_id, shop):
    return response_generator(latest_user_message, user_id, shop)

@app.get('/finetune_response')
def create_response_(query, response, shop, expiry_date='NA', action='NA'):
    create_response_status = create_response(query, response, shop, expiry_date, action)
    return create_response_status

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=10000)