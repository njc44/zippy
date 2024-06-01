import nest_asyncio
nest_asyncio.apply()

import pandas as pd
import requests
from PIL import Image
import numpy as np
from io import BytesIO
import base64
import aiohttp  # for making API calls concurrently
import argparse  # for running script from command line
import asyncio  # for running API calls concurrently
import json  # for saving results to a jsonl file
import logging  # for logging rate limit warnings and other messages
import os  # for reading API key
import re  # for matching endpoint from request URL
import tiktoken  # for counting tokens
import time  # for sleeping after rate limit is hit
from dataclasses import (dataclass,field)  # for storing API inputs, outputs, and metadata
from pytz import timezone 
from datetime import datetime
import ast
import random
import openai
from dotenv import load_dotenv
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import psycopg2
from sqlalchemy import create_engine

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.getEffectiveLevel()

from gpt4_utils import result_dir, configs, get_batched_data_with_configs
from finetune_response import create_response
from search import *
from app_utils import *

load_dotenv()
client=openai.OpenAI()


def get_product_data(shop):
    SHOPIFY_URL = f"https://{shop}/admin/api/2024-04/graphql.json"

    conn, cur, engine = get_connection()
    query = f"SELECT * FROM secrets where shop = '{shop}'"
    secrets_df = pd.read_sql_query(query, conn).reset_index(drop=True)
    cut_connection(conn, cur)

    headers = {
        "X-Shopify-Access-Token": secrets_df["shopify_access_token"][0],
        "Content-Type": "application/graphql"
    }

    query1  = """mutation { bulkOperationRunQuery( query: "{ products { edges { node { createdAt description descriptionHtml featuredImage { id url altText } handle id isGiftCard onlineStoreUrl productType publishedAt requiresSellingPlan seo { title description } tags title totalInventory updatedAt vendor variants { edges { node { id title price sku weight weightUnit availableForSale image{ altText id url } } } } images { edges { node { id originalSrc altText } } } } } } }" ) { bulkOperation { id status } userErrors { field message } } }"""

    response = requests.post(SHOPIFY_URL, headers=headers, data=query1)

    if response.status_code == 200:
        data = response.json()
        print(data)
    else:
        print(f"Error: {response.status_code} - {response.text}")

    time.sleep(2)

    query2 = """{ currentBulkOperation { id status errorCode createdAt completedAt objectCount fileSize url } }"""

    response = requests.post(SHOPIFY_URL, headers=headers, data=query2)

    if response.status_code == 200:
        data = response.json()
        print(data)
    else:
        print(f"Error: {response.status_code} - {response.text}")
        
    url = data['data']['currentBulkOperation']['url']
    r = requests.get(url) 
    jsonObj = pd.read_json(BytesIO(r.content), lines=True)
    return jsonObj

def add_variant_info(row, variant_df):
    parent_id = row['id']
    df_temp = variant_df[variant_df['__parentId']==parent_id].reset_index(drop=True)
    temp_list = []
    for irow in df_temp.iloc():
        temp_list.append(dict(irow[['id','price','title', 'image', 'sku', 'weight','weightUnit', 'availableForSale', '__parentId']]))
    return temp_list

def get_product_df(shop):
    jsonObj = get_product_data(shop)

    product_df = jsonObj[~jsonObj['createdAt'].isnull()][['featuredImage','handle', 'id', 'isGiftCard', 'productType', 'seo', 'tags', 'title','totalInventory', 'vendor','description']].reset_index(drop=True)
    variant_df = jsonObj[jsonObj['createdAt'].isnull()][['id','price', 'title', 'image', 'sku', 'weight','weightUnit', 'availableForSale', '__parentId']].reset_index(drop=True)
    live_variant_df = variant_df[variant_df['availableForSale']==1].reset_index(drop=True)

    product_df['variant_info'] = product_df.apply(add_variant_info,variant_df=live_variant_df,axis=1)
    product_df = product_df.dropna().reset_index(drop=True)
    product_df[['featuredImage_url','featuredImage_id','featuredImage_altText']] = product_df.apply(lambda x: pd.Series([x['featuredImage']['url'],x['featuredImage']['id'],x['featuredImage']['altText']]),axis=1)
    product_df['id'] = product_df.apply(lambda x: x['id'].split('/')[-1],axis=1)
    
    return product_df

def remove_substrings_between_tags(text):
    pattern = r"<.*?>"
    result = re.sub(pattern, "", text)
    return result


def get_brand_and_policy_info(shop):

    conn, cur, engine = get_connection()
    query = f"SELECT * FROM secrets where shop = '{shop}'"
    secrets_df = pd.read_sql_query(query, conn).reset_index(drop=True)
    cut_connection(conn, cur)

    headers = {
        'Content-Type': 'application/json',
        'X-Shopify-Storefront-Access-Token': secrets_df["shopify_storefront_access_token"][0]
    }
    payload = {
        'query': 'query { shop { brand { slogan shortDescription } privacyPolicy { body handle title } refundPolicy { body handle title } shippingPolicy { body handle title } subscriptionPolicy { body handle title } termsOfService { body handle title } } }'
    }
    response = requests.post(f'https://{shop}/api/2024-04/graphql.json', headers=headers, data=json.dumps(payload))
    data = response.json()['data']['shop']
    for policy in ['privacyPolicy','refundPolicy','shippingPolicy','subscriptionPolicy','termsOfService']:
        text = data[policy]['body']
        cleaned_text = remove_substrings_between_tags(text)
        data[policy]['body'] = cleaned_text
    return data


#####################################

@dataclass
class StatusTracker:
    """Stores metadata about the script's progress. Only one instance is created."""

    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0  # script ends when this reaches 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0  # excluding rate limit errors, counted above
    num_other_errors: int = 0
    time_of_last_rate_limit_error: int = 0  # used to cool off after hitting rate limits
    num_content_moderation_error: int = 0

def task_id_generator_function():
    """Generate integers 0, 1, 2, and so on."""
    task_id = 0
    while True:
        yield task_id
        task_id += 1
    
def append_to_jsonl(data, filename: str) -> None:
    """Append a json payload to the end of a jsonl file."""
    logger.debug(f"""Append to json\ndata: {data}\nfilename: {filename}""")
    json_string = json.dumps(data)
    with open(filename, "a") as f:
        f.write(json_string + "\n")

def num_tokens_consumed_from_request(request_json: dict):
    """Count the number of tokens in the request. Only supports completion and embedding requests."""
    token_encoding_name="cl100k_base"
    encoding = tiktoken.get_encoding(token_encoding_name)
    max_tokens = request_json.get("max_tokens")
    completion_tokens = max_tokens
    image_token = 1000*2
    # chat completions
    num_tokens = 0
    for message in request_json["messages"]:
        num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        for key, value in message.items():
            if isinstance(value, str):
                num_tokens += len(encoding.encode(value))
            elif isinstance(value, list):
                for entry in value:
                    for k1, v1 in entry.items():
                        if k1 == 'text':
                            if isinstance(v1, str):
                                num_tokens += len(encoding.encode(v1))
            if key == "name":  # if there's a name, the role is omitted
                num_tokens -= 1  # role is always required and always 1 token
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens + image_token + completion_tokens

@dataclass
class APIRequest:
    """Stores an API request's inputs, outputs, and other metadata. Contains a method to make an API call."""

    task_id: int
    request_json: dict
    token_consumption: int
    attempts_left: int
    metadata: dict
    result: list = field(default_factory=list)

    async def call_api(
        self,
        session: aiohttp.ClientSession,
        request_url: str,
        request_header: dict,
        retry_queue: asyncio.Queue,
        save_filepath: str,
        status_tracker: StatusTracker
    ):
        """Calls the OpenAI API and saves results."""
        logger.debug(f"Starting request #{self.task_id}")
        logger.info(f"num of tasks started: {status_tracker.num_tasks_started}\nnum of tasks in progress: {status_tracker.num_tasks_in_progress}\nnum of tasks failed so far: {status_tracker.num_tasks_failed}")

        ind_time = datetime.now(timezone("Asia/Kolkata")).strftime('%H:%M:%S.%f')
        logger.info(f"current time: {ind_time}")
        error = None
        rate_limit_happened = False
        try:
            logger.debug(f"calling azure with following creds\n1. request_url: {request_url}\n2. request_header: {request_header}\n3. request_json: {self.request_json}")
            async with session.post(
                url=request_url, headers=request_header, json=self.request_json
            ) as response:
                response = await response.json()
            if "error" in response:
                if "content management policy" not in response["error"].get("message", ""):
                    logger.warning(
                        f"Request {self.task_id} failed with error {response['error']}"
                    )
                status_tracker.num_api_errors += 1
                error = response
                if "Rate limit" in response["error"].get("message", ""):
                    rate_limit_happened = True
                    status_tracker.time_of_last_rate_limit_error = time.time()
                    status_tracker.num_rate_limit_errors += 1
                    status_tracker.num_api_errors -= (
                        1  # rate limit errors are counted separately
                    )
                if "content management policy" in response["error"].get("message", ""):
                    status_tracker.num_content_moderation_error += 1

        except (
            Exception
        ) as e:  # catching naked exceptions is bad practice, but in this case we'll log & save them
            logger.warning(f"Request {self.task_id} failed with Exception {e}")
            status_tracker.num_other_errors += 1
            error = e
        if error:
            self.result.append(error)
            logger.info(f"attempts left: {self.attempts_left}")
            # Retrying only in case of rate-limit
            if self.attempts_left > 0 and rate_limit_happened:
                retry_queue.put_nowait(self)
            else:
                logger.error(f"Request failed after all attempts. Saving errors")
                data = (
                    [self.metadata['request_id'], self.metadata['region'], [str(e) for e in self.result], self.request_json] if self.metadata
                    else [self.request_json, [str(e) for e in self.result]]
                )
                append_to_jsonl(data, save_filepath)
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
        else:
            data = (
                [self.metadata['request_id'], self.metadata['region'], response,  "fetch_request_json_using_metadata"] if self.metadata
                else [self.request_json, response]
            )
            append_to_jsonl(data, save_filepath)
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1
            logger.debug(f"Request {self.task_id} saved to {save_filepath}")

# functions

def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')
        
def format_request(request_json, region_name, is_fallback):

    request_id = request_json['product_id_0']
    attributes = 'ornamentation, brand, fit_shape, sleeve_length, sleeve_styling, pattern, combo_of, length, stitch_type, occasion, neck, color, fabric, print_or_pattern_type'

    system_messages_content = [{"type": "text", "text": f"""You are a data extraction agent, your job is to create -
1. An list of extreamly relevant search queries for the following products. Make sure to include thematic queries as well like "winter inner" or "beach outfit", also mention color if important, limit to 7 queries.
2. A consise and SEO optimised product description that contains all visual/non-visual attributes and relevant keywords for the search queries to pick it up, limit to 50 words.
Respond in json parseable format with key as - id and values as dictionary: 
- with key as 'queries' & value as list of queries
- with key as 'product_description' & value as product description string."""}]

    lst = list(request_json.keys())
    idx_size=0
    for i in lst:
        if 'product_id' in i:
            idx_size+=1

    messages_content = []
    for i in range(idx_size):
        product_id = request_json[f'product_id_{i}']
        image = request_json[f'featureImage_url_{i}']
        title = request_json[f'title_{i}']
        featuredImage_altText = request_json[f'featuredImage_altText_{i}']
        product_description = request_json[f'product_description_{i}']
        if product_id != None:
            if messages_content != []:
                messages_content = messages_content + [{"type": "text", "text": f"product_id - {product_id}\nproduct_description - {product_description}\ntitle - {title}\nOther text - {featuredImage_altText}"}] + [{
                            "type": "image_url",
                            "image_url": {
                                "url": image,
                                "detail": "low"
                            }
                            }] 
            else:
                messages_content = [{"type": "text", "text": f"product_id - {product_id}\nproduct_description - {product_description}\ntitle - {title}\nOther text - {featuredImage_altText}"}] + [{
                            "type": "image_url",
                            "image_url": {
                                "url": image,
                                "detail": "low"
                            }
                            }] 

    messages = [{"role": "system","content": system_messages_content},{"role": "user","content": messages_content}]
        
    payload = {
        "messages": messages,
        "max_tokens": 4096,
        "temperature": 0.2,
        "metadata": {"region": region_name, "request_id": request_id}
    }
    if region_name == 'openai':
        payload.update({"model": "gpt-4o"})
    return payload

def get_header(key, region):
    if region == 'openai':
        return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}"
            }
    else:
        return {"api-key": key,
                      "Content-Type": "application/json"}
    
async def process_api_requests_from_file(
    batch_data: list,
    save_filepath: str,
    url: str,
    api_key: str,
    region_name: str,
    max_rpm: float,
    max_tpm: float,
    max_attempts: int,
    is_fallback:bool=False
):
    """Processes API requests in parallel, throttling to stay under rate limits."""
    # constants
    max_requests_per_minute = max_rpm
    max_tokens_per_minute = max_tpm
    request_url = url
    token_encoding_name = "cl100k_base"
    seconds_to_pause_after_rate_limit_error = 15
    seconds_to_sleep_each_loop = (
        0.001  # 1 ms limits max throughput to 1,000 requests per second
    )

    # infer API endpoint and construct request header
    # api_endpoint = api_endpoint_from_url(request_url)
    api_endpoint = request_url
    request_header = get_header(api_key, region_name)
    logger.debug(f"""request-header: {request_header}""")

    # initialize trackers
    queue_of_requests_to_retry = asyncio.Queue()
    task_id_generator = (
        task_id_generator_function()
    )  # generates integer IDs of 0, 1, 2, ...
    status_tracker = (
        StatusTracker()
    )  # single instance to track a collection of variables
    next_request = None  # variable to hold the next request to call

    # initialize available capacity counts
    available_request_capacity = max_requests_per_minute
    available_token_capacity = max_tokens_per_minute
    last_update_time = time.time()

    # initialize flags
    file_not_finished = True  # after file is empty, we'll skip reading it
    logger.debug(f"Initialization complete.")

    # initialize batch processing
    row_index = 0
    batch_size = len(batch_data)
    # `requests` will provide requests one at a time
    logger.debug(f"Entering main loop")
    async with aiohttp.ClientSession() as session:  # Initialize ClientSession here
        while True:
            # get next request (if one is not already waiting for capacity)
            if next_request is None:
                if not queue_of_requests_to_retry.empty():
                    next_request = queue_of_requests_to_retry.get_nowait()
                    logger.debug(f"Retrying request {next_request.task_id}: {next_request}")
                elif file_not_finished:
                    # get new request
                    if row_index == batch_size:
                        logger.debug("Read file exhausted")
                        file_not_finished = False
                    else:
                        each_row = batch_data[row_index]
                        request_json = json.loads(each_row)
                        request_json = format_request(request_json, region_name, is_fallback)
                        logger.debug(f"""request_data:{request_json}""")
                        next_request = APIRequest(
                            task_id=next(task_id_generator),
                            request_json=request_json,
                            token_consumption=num_tokens_consumed_from_request(request_json),
                            attempts_left=max_attempts,
                            metadata=request_json.pop("metadata", None),
                        )
                        status_tracker.num_tasks_started += 1
                        status_tracker.num_tasks_in_progress += 1
                        row_index += 1
                        logger.debug(
                            f"Reading request {next_request.task_id}: {next_request}"
                        )  

            # update available capacity
            current_time = time.time()
            seconds_since_update = current_time - last_update_time
            available_request_capacity = min(
                available_request_capacity
                + max_requests_per_minute * seconds_since_update / 60.0,
                max_requests_per_minute,
            )
            available_token_capacity = min(
                available_token_capacity
                + max_tokens_per_minute * seconds_since_update / 60.0,
                max_tokens_per_minute,
            )
            last_update_time = current_time

            # if enough capacity available, call API
            if next_request:
                next_request_tokens = next_request.token_consumption
                if (
                    available_request_capacity >= 1
                    and available_token_capacity >= next_request_tokens
                ):
                    # update counters
                    available_request_capacity -= 1
                    available_token_capacity -= next_request_tokens
                    next_request.attempts_left -= 1

                    # call API
                    asyncio.create_task(
                        next_request.call_api(
                            session=session,
                            request_url=request_url,
                            request_header=request_header,
                            retry_queue=queue_of_requests_to_retry,
                            save_filepath=save_filepath,
                            status_tracker=status_tracker,
                        )
                    )
                    next_request = None  # reset next_request to empty

            # if all tasks are finished, break
            if status_tracker.num_tasks_in_progress == 0:
                break

            # main loop sleeps briefly so concurrent tasks can run
            # await asyncio.sleep(seconds_to_sleep_each_loop)
            await asyncio.sleep(60/max_requests_per_minute)

            # if a rate limit error was hit recently, pause to cool down
            seconds_since_rate_limit_error = (
                time.time() - status_tracker.time_of_last_rate_limit_error
            )
            if (
                seconds_since_rate_limit_error
                < seconds_to_pause_after_rate_limit_error
            ):
                remaining_seconds_to_pause = (
                    seconds_to_pause_after_rate_limit_error
                    - seconds_since_rate_limit_error
                )
                await asyncio.sleep(remaining_seconds_to_pause)
                # ^e.g., if pause is 15 seconds and final limit was hit 5 seconds ago
                logger.warn(
                    f"Pausing to cool down until {time.ctime(status_tracker.time_of_last_rate_limit_error + seconds_to_pause_after_rate_limit_error)}"
                )

    # after finishing, log final status
    logger.info(
        f"""Parallel processing complete. Results saved to {save_filepath}"""
    )
    if status_tracker.num_tasks_failed > 0:
        logger.warning(
            f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} requests failed. Errors logged to {save_filepath}."
        )
    if status_tracker.num_rate_limit_errors > 0:
        logger.warning(
            f"{status_tracker.num_rate_limit_errors} rate limit errors received. Consider running at a lower rate."
        )
    if status_tracker.num_content_moderation_error > 0:
        logger.warning(
            f"{status_tracker.num_content_moderation_error} / {status_tracker.num_tasks_started} content moderation errors received."
        )

def save_keys(shop, shopify_storefront_access_token, shopify_access_token):
    table_name = 'secrets'
    new_row = pd.DataFrame({'shop' : [shop], 'shopify_storefront_access_token': [shopify_storefront_access_token], 'shopify_access_token': [shopify_access_token]})
    insert_or_update_row_secrets(new_row, table_name)

async def train(shop, shopify_storefront_access_token, shopify_access_token):
    try:
        table_name = 'train_status'
        new_row = pd.DataFrame({'train_status' : ['training start'],'shop' : [shop]})
        insert_update_train_status(new_row,table_name)
        save_keys(shop, shopify_storefront_access_token, shopify_access_token)
        product_df = get_product_df(shop)
        df = product_df[['id','featuredImage_url','title','description','featuredImage_altText']]
        df1 = pd.DataFrame()
        batch_size = 4
        for i in range(0,len(df),batch_size):
            df_x = df[i:min(i+batch_size,len(df))].reset_index(drop=True)
            x = df_x.T[:1].rename(columns={j:f'product_id_{j}' for j in range(batch_size)}).reset_index(drop=True)
            y = df_x.T[1:2].rename(columns={j:f'featureImage_url_{j}' for j in range(batch_size)}).reset_index(drop=True)
            z = df_x.T[2:3].rename(columns={j:f'title_{j}' for j in range(batch_size)}).reset_index(drop=True)
            w = df_x.T[3:4].rename(columns={j:f'product_description_{j}' for j in range(batch_size)}).reset_index(drop=True)
            v = df_x.T[4:5].rename(columns={j:f'featuredImage_altText_{j}' for j in range(batch_size)}).reset_index(drop=True)
            df_x = pd.concat([x,y,z,w,v],axis=1)
            df1 = pd.concat([df1,df_x],axis=0).reset_index(drop=True)
        main_data_path = f'/app/app/gpt4v_feature_extraction.jsonl'
        df1.to_json(main_data_path, orient='records', lines=True)

        final_batched_data = get_batched_data_with_configs(main_data_path,is_post_process=True)
        async def process_batches(batch_configs):
            await asyncio.gather(*(process_api_requests_from_file(**batch) for batch in batch_configs))
        asyncio.run(process_batches(final_batched_data))

        table_name = 'train_status'
        new_row = pd.DataFrame({'train_status' : ['async feature extraction done'],'shop' : [shop]})
        insert_update_train_status(new_row,table_name)

        df_list = []
        for k, v in configs.items():
            if k!='openai':
                continue
            p = os.path.join(result_dir, f'{k}.jsonl')
            if os.path.isfile(p):
                df = pd.read_json(p, lines=True)
                df_list.append(df)
        big_df = pd.concat(df_list, ignore_index=True)
        print('len - ', big_df.shape)
        passed_pddf = big_df[big_df[3]=='fetch_request_json_using_metadata'].reset_index(drop=True)

        output_list = []
        for idx in range(len(passed_pddf)):
            try:
                output_list.append(ast.literal_eval(passed_pddf[2][idx]['choices'][0]['message']['content'].replace('json','').replace("```",'')))
            except:
                pass
        output_json = {k: v for dct in output_list for k, v in dct.items()}

        product_df['queries'] = product_df.apply(lambda x: output_json[str(x['id'])]['queries'],axis=1)
        product_df['seo_product_description'] = product_df.apply(lambda x: output_json[str(x['id'])]['product_description'],axis=1)
        product_info_list = [','.join([query for query in product_df[product_df['id']==str(id)]['queries'][product_df[product_df['id']==str(id)].index[0]]]) + \
        product_df[product_df['id']==str(id)]['seo_product_description'][product_df[product_df['id']==str(id)].index[0]] + \
        product_df[product_df['id']==str(id)]['title'][product_df[product_df['id']==str(id)].index[0]] + \
        product_df[product_df['id']==str(id)]['featuredImage_altText'][product_df[product_df['id']==str(id)].index[0]] for id in product_df['id']]
        product_info_list = [re.sub(r'[^a-zA-Z0-9\s]', '',str(i).replace(',',' ').replace('\n','')) for i in product_info_list]
        product_info_list = [' '.join(j.split()) for j in product_info_list]
        product_vecs = process_search_queries(product_info_list)
        product_df['sparse_product_vector'] = pd.DataFrame(product_vecs).apply(lambda row: row.tolist(), axis=1)
        product_df['shop'] = shop

        table_name = 'products'
        product_df = product_df[['id','sparse_product_vector','shop']]
        product_df['sparse_product_vector'] = product_df.apply(lambda x: json.dumps(list(x['sparse_product_vector'])),axis=1)

        conn, cur, engine = get_connection()
        insert_or_update_row_products(product_df, table_name)
        cut_connection(conn, cur)

        table_name = 'train_status'
        new_row = pd.DataFrame({'train_status' : ['Writing data to product table done'], 'shop' : [shop]})
        insert_update_train_status(new_row,table_name)

        os.remove("/app/app/gpt4v_feature_extraction.jsonl")
        # data = get_brand_and_policy_info(shop)
        
        # questions = ['What is your return policy?','How long does shipping typically take?']
        # system_prompt = f"""You are a sales agent on an ecommerce platform, your job is to reply to customer queries just as a real life sales agent would. What would your reply be to '{questions[0]}' & {questions[1]} given:
        # Refund Policy - {data['refundPolicy']['body']}\n
        # Shipping Policy - {data['shippingPolicy']['body']}\n
        # Make sure the answers are crisp and to the point
        # Respond in json format with keys 'refund' & 'shipping' with the values as your responses.
        # """

        # messages = [{"role": "system","content": system_prompt}]
        # response = client.chat.completions.create(
        #     model="gpt-3.5-turbo-0125",
        #     messages=messages,
        #     response_format = { "type": "json_object" }
        #     )
        # reply = response.choices[0].message.content
        # reply = json.loads(reply)
        # bool1 = await create_response(questions[0],reply['refund'],shop)
        # bool2 = await create_response(questions[1],reply['shipping'],shop)

        table_name = 'train_status'
        new_row = pd.DataFrame({'train_status' : ['Creating finetuned response done, Training complete!'],'shop' : [shop]})
        insert_update_train_status(new_row,table_name)

        return True #bool1 and bool2
    except Exception as e:
        print("Error train", e)
        table_name = 'train_status'
        new_row = pd.DataFrame({'train_status' : ['Training failed'],'shop' : [shop]})
        insert_update_train_status(new_row,table_name)
        return False