from fastapi import FastAPI
import random
import time
import openai
import os
from dotenv import load_dotenv
import requests
import json
import uvicorn

app = FastAPI()

@app.get('/')
def homepage():
    return json.dumps({'Message': 'Welcome to EpicAI API'})

@app.get('/health_check')
def health_check():
    return json.dumps({"success": True}), 200

load_dotenv()
client=openai.OpenAI()

tools = [
  {
    "type": "function",
    "function": {
      "name": "call_search_api",
      "description": "Gets upto top 3 search results of products based of search keywords only if user query indicates intent to search for specific products",
      "parameters": {
        "type": "object",
        "properties": {
          "keywords": {
            "type": "string",
            "description": "A list of search keywords/tags eg. Red T-shirt, coffee mug etc.",
          }
        },
        "required": ["keywords"],
      },
    }
  }
]

def call_search_api(keywords):
    headers = {
        'Content-Type': 'application/json',
        'X-Shopify-Storefront-Access-Token': os.getenv('SHOPIFY_TOKEN')
    }
    payload = {
        'query': 'query searchProducts($query: String!, $first: Int) { search(query: $query, first: $first, types: PRODUCT) { edges { node { ... on Product { id title } } } } }',
        'variables': {
        "query": f"{keywords}",
        "first": 3
    }
    }
    response = requests.post('https://quickstart-31717217.myshopify.com/api/2024-01/graphql.json', headers=headers, data=json.dumps(payload))
    return response.json()['data']['search']['edges']


@app.get('/input_payload')
def response_generator(latest_user_message, previous_agent_response=None, messages=None):
    
    if messages!=None:
        messages = json.loads(messages)['messages']

    system_prompt = f"""You are a sales agent on an ecommerce platform, your job is to reply to customer queries just as a real life sales agent would. You will be given relevant info about the products and policies if and when required to be used to answer a query appropriately"""

    if messages == None:
        messages = [{"role": "system","content": system_prompt}, 
                    {"role": "user", "content": latest_user_message}]
    else:
        messages.append({"role": "assistant", "content": previous_agent_response})
        messages.append({"role": "user", "content": latest_user_message})
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=messages,
        tools=tools,
        tool_choice="auto",
        max_tokens=4096
        )
    
    search_response = []
    reply = ''

    if type(response.choices[0].message.content) != str:
        if response.choices[0].message.tool_calls[0].function.name == "call_search_api":
                keywords = json.loads(response.choices[0].message.tool_calls[0].function.arguments)['keywords']
                search_response = call_search_api(keywords)
        if search_response == []:
            reply = "Hey sorry, we don't have that item"
        else:
            product_info = ""
            for idx, i in enumerate(search_response):
                product_info += f'product_{idx}: '+i['node']['title']+'\n'
            messages.append({"role": "system","content": f"Here are some products that surfaced from the customer query: \n{product_info} Try to recommend these to the customer."})
            response = client.chat.completions.create(
              model="gpt-4-turbo",
              messages=messages,
              max_tokens=4096
              )
            reply = response.choices[0].message.content
    else:
        reply = response.choices[0].message.content

    return_value = {}
    return_value['response'] = reply
    return_value['messages'] = messages

    return json.dumps(return_value)

@app.get('/create_responses')
def create_response(query, response):
    

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=10000)