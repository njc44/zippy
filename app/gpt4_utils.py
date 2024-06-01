import os
from itertools import islice
import pandas as pd

from dotenv import load_dotenv
load_dotenv()


run_id = 1
rootdir = f'/tmp/{run_id}'
result_dir = os.path.join(rootdir, 'gpt4v_output')
os.makedirs(result_dir, exist_ok=True)


configs = {
# "summarization_uswest": {"url": "https://deployment.openai.azure.com/openai/deployments/gpt4-vision/chat/completions?api-version=2024-02-15-preview", "api_key": "key", "max_rpm": 8, "max_tpm": 30000, "max_attempts": 1},

"openai": {"url": "https://api.openai.com/v1/chat/completions", "api_key": os.getenv("OPENAI_API_KEY"),"max_rpm": 100, "max_tpm": 8000, "max_attempts": 1}
}


config_savepath = {region: {'save_filepath': os.path.join(result_dir, f'{region}.jsonl')} for region, _ in configs.items()}

# COMMAND ----------

def get_overall_data_size(path):
    df = pd.read_json(path, lines=True)
    return df.shape[0]

def get_traffic_of_regions(data_size, is_post_process):
    ##define weights
    if not is_post_process:
        azure_weights = {
        'chat_auseast':0.2,
        'chat_japaneast':0.2,
        'chat_swznorth':0.1,
        'crosssell_auseast': 0.025,
        'crosssell_japaneast': 0.025,
        'crosssell_swncentral': 0.025,
        'crosssell_swtnorth': 0.025,
        'crosssell_uswest': 0.025,
        'search_auseast': 0.025,
        'search_japaneast': 0.025,
        'search_swncentral': 0.025,
        'search_swtnorth': 0.025,
        'search_uswest': 0.025,
        'dp_auseast': 0.025,
        'dp_japaneast': 0.025,
        'dp_swncentral': 0.025,
        'dp_swtnorth': 0.025,
        'dp_uswest': 0.025,
        'summarization_auseast': 0.025,
        'summarization_japaneast': 0.025,
        'summarization_swncentral': 0.025,
        'summarization_swtnorth': 0.025,
        'summarization_uswest': 0.025,
    }
    else:
        azure_weights = {}
    azure_traffic = {k: round(v*data_size) for k, v in azure_weights.items()}
    remaining_data_size = data_size - sum(azure_traffic.values())
    if is_post_process:
        overall_traffic = {'openai': remaining_data_size}
    elif remaining_data_size > 0 and not is_post_process:
        key = list(azure_traffic.keys())[0]
        overall_traffic = {key: azure_traffic[key]+remaining_data_size}
    else:
        overall_traffic = {}
    azure_traffic.update(overall_traffic)
    azure_traffic = dict(sorted(azure_traffic.items()))
    return azure_traffic

def get_batched_data(data_path, is_post_process):
    data_size = get_overall_data_size(data_path)
    print(f'overall data size: {data_size}')
    traffic_dist = get_traffic_of_regions(data_size, is_post_process)
    print(f'traffic distribution: {traffic_dist}')
    batched_data = {}
    with open(data_path, 'r') as f:
        for region, batch_size in traffic_dist.items():
            # take the next BATCH_SIZE lines
            batch = [line.strip() for line in islice(f, batch_size)]
            if not batch:
                # no more lines - we're done
                break
            batched_data[region] = batch
    return batched_data

def get_batched_data_with_configs(main_data_path, is_post_process = False):
    batched_data_dict = get_batched_data(main_data_path, is_post_process)
    batched_data_with_configs = []
    for region, batch_data in batched_data_dict.items():
        templated_data = {'region_name': region,'batch_data': batch_data}
        templated_data.update(configs[region])
        templated_data.update(config_savepath[region])
        batched_data_with_configs.append(templated_data)
    return batched_data_with_configs