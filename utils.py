# This file contains necessary helper functions
# e.g. GPT request, create_dataloader
import openai
import random
import sys
import numpy as np
import torch
import argparse
from torch.utils.data import Dataset
from pathlib import Path
import json
import re
from collections import Counter

# put your API key in the list
API_KEY_POOL = [
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
# '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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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',
'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'
]
NUM_API_KEYS = len(API_KEY_POOL)
# NO_SOLUTION = '<NO_SOL>'
NO_SOLUTION = '-10086'

API_PARTITION_POOL = [
    {"cur_index":0, "keys":API_KEY_POOL[0:20]},
    {"cur_index":0, "keys":API_KEY_POOL[20:40]},
    {"cur_index":0, "keys":API_KEY_POOL[40:60]},
    {"cur_index":0, "keys":API_KEY_POOL[60:80]},
    {"cur_index":0, "keys":API_KEY_POOL[80:100]},
    {"cur_index":0, "keys":API_KEY_POOL[100:120]},
    {"cur_index":0, "keys":API_KEY_POOL[120:140]},
    {"cur_index":0, "keys":API_KEY_POOL[140:160]},
    {"cur_index":0, "keys":API_KEY_POOL[160:180]},
    {"cur_index":0, "keys":API_KEY_POOL[180:200]},
    {"cur_index":0, "keys":API_KEY_POOL[200:206]},
]

# set the random seed for reproducibility
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# pass in a list of prompts and returns a response body contains a list of responses
def GPT3_request(model:str, input_prompt:list, max_tokens:int, temperature=0.7, stop=None, worker_id=None):
    resp = None
    done = False
    while not done:
        try:
            key_index = random.randint(0, NUM_API_KEYS - 1)
            openai.api_key = API_KEY_POOL[key_index]
            # key_index = API_PARTITION_POOL[worker_id]['cur_index']
            # openai.api_key = API_PARTITION_POOL[worker_id]["keys"][key_index]
            resp = openai.Completion.create(
                model=model,
                prompt=input_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop = stop
            )
            done = True
            # key_index += 1
            # if key_index == len(API_PARTITION_POOL[worker_id]['keys']):
            #     API_PARTITION_POOL[worker_id]['cur_index'] = 0
            # else:
            #     API_PARTITION_POOL[worker_id]['cur_index'] = key_index
        except:
            errno = sys.exc_info()[:2]
            if errno[0] == openai.error.InvalidRequestError:
                print(f"Invalid Request\nPrompt: {input_prompt}\n")
                assert False
            else:
                print(f"Error: {errno[0]}\n")
                # print(f"Reason: {errno[1]}\n")

            # key_index = API_PARTITION_POOL[worker_id]['cur_index']
            # key_index += 1
            # if key_index == len(API_PARTITION_POOL[worker_id]['keys']):
            #     API_PARTITION_POOL[worker_id]['cur_index'] = 0
            # else:
            #     API_PARTITION_POOL[worker_id]['cur_index'] = key_index
            # time.sleep(5)
    return resp


def load_data(args):
    questions = []
    answers = []
    decoder = json.JSONDecoder()

    if args.dataset == "gsm8k":
        with open(args.dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                questions.append(json_res["question"].strip())
                answers.append(json_res["answer"].split("#### ")[-1])
    elif args.dataset == "aqua":
        with open(args.dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                qes = json_res["question"].strip() + " Answer Choices:"

                for opt in json_res["options"]:
                    qes += f" ({opt}"

                questions.append(qes)
                answers.append(json_res["correct"])
    elif args.dataset == "svamp":
        with open(args.dataset_path) as f:
            json_data = json.load(f)
            for line in json_data:
                q = line["Body"].strip() + " " + line["Question"].strip()
                a = str(line["Answer"])
                if a[-2:] == ".0":
                    a = a[:-2]
                questions.append(q)
                answers.append(a)
    elif args.dataset == "asdiv":
        with open(args.dataset_path) as f:
            json_data = json.load(f)
            for line in json_data:
                q = line["Body"].strip() + " " + line["Question"].strip()
                a = line['Answer'].split(" ")[0]
                questions.append(q)
                answers.append(a)
    elif args.dataset in ("addsub", "singleeq"):
        with open(args.dataset_path) as f:
            json_data = json.load(f)
            for line in json_data:
                q = line["sQuestion"].strip()
                a = str(line["lSolutions"][0])
                if a[-2:] == ".0":
                    a = a[:-2]
                questions.append(q)
                answers.append(a)
    elif args.dataset == "csqa":
        with open(args.dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                choice = "Answer Choices:"
                for c in json_res["question"]["choices"]:
                    choice += " ("
                    choice += c["label"]
                    choice += ") "
                    choice += c["text"]
                questions.append(json_res["question"]["stem"].strip() + " " + choice)
                answers.append(json_res["answerKey"])
    elif args.dataset == "strategyqa":
        with open(args.dataset_path, encoding='utf-8') as f:
            json_data = json.load(f)
            for line in json_data:
                q = line["question"].strip() 
                if line['answer']:
                    a = 'yes'
                else:
                    a = 'no'
                questions.append(q)
                answers.append(a)
    elif args.dataset in ("coin_flip", "last_letters"):
        with open(args.dataset_path) as f:
            json_data = json.load(f)
            json_data = json_data["examples"]
            for line in json_data:
                q = line["question"]
                a = line["answer"]
                questions.append(q)
                answers.append(a)
    else:
        raise NotImplementedError

    print(f"dataset: {args.dataset}")
    print(f"dataset_size: {len(answers)}")
    args.dataset_size = len(answers)
    return questions, answers


# process the dataset into a loader of batches
def batchlize(examples:list, batch_size:int):
    size = 0
    questions = []
    length = len(examples)
    random.shuffle(examples)
    while size < length:
        if length - size > batch_size:
            questions.append(examples[size:size+batch_size])
            size += batch_size
        else:
            questions.append(examples[size:size+(length-size)])
            size += (length - size)
        # print(f"size: {size}")
    return questions


# return a customized dataloader of batches
# Not PyTorch dataloader, it supprts random index(slice) access
def create_dataloader(args)->list:
    set_random_seed(args.random_seed)
    questions, answers = load_data(args)
    dataset = []
    for idx in range(len(questions)):
        dataset.append({"question":questions[idx], "answer":answers[idx], "question_idx":idx})

    dataloader = batchlize(dataset, args.minibatch_size)
    print(f"dataloader size: {len(dataloader)}")
    return dataloader


# read the generated/prepared prompt json file
# return a string of prefix prompt before each question
def create_input_prompt(args, cot_flag:bool)->str:
    x, z, y = [], [], []
    
    with open(args.prompt_path, encoding="utf-8") as f:
        json_data = json.load(f)
        json_data = json_data["prompt"]
        for line in json_data:
            x.append(line["question"])
            z.append(line["rationale"])
            y.append(line["pred_ans"])

    index_list = list(range(len(x)))
    
    prompt_text = ""
    for i in index_list:
        if cot_flag:
            prompt_text += x[i] + " " + z[i] + " " + \
                        args.direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"
        else:
            prompt_text += x[i] + " " + args.direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"
    return prompt_text


def answer_extraction(args, responses):
    ans_list = ["" for i in range(len(responses['choices']))]
    for resp_idx in range(len(responses['choices'])):
        temp = responses['choices'][resp_idx].text
        if args.dataset in ("gsm8k", "svamp", "asdiv", "addsub", "singleeq"):
            temp = temp.replace(",", "")
            temp = [s for s in re.findall(r'-?\d+\.?\d*', temp)]
        elif args.dataset in ("aqua", "csqa"):
            temp = re.findall(r'A|B|C|D|E', temp)
        elif args.dataset in ("strategyqa", "coin_flip"):
            temp = temp.lower()
            temp = re.sub("\"|\'|\n|\.|\s|\:|\,"," ", temp)
            temp = temp.split(" ")
            temp = [i for i in temp if i in ("yes", "no")]
        elif args.dataset in ("last_letters"):
            temp = re.sub("\"|\'|\n|\.|\s","", temp)
            temp = [temp]
        if len(temp) != 0:
            answer = temp[-1]
            # if there is . at the end of answer, remove it
            # e.g. answer = 64.
            if answer != "":
                if answer[-1] == ".":
                    answer = answer[:-1]

            # round the answer to nearest integer
            if args.dataset in ("gsm8k", "svamp", "asdiv", "singleeq"):
                try:
                    answer = str(round(float(answer)))
                except:
                    answer = "" # no sol or sol doesn't have valid format
            elif args.dataset in ("last_letters"):
                try:
                    answer = answer[-args.concat_length:]
                except:
                    answer = ""
            
            ans_list[resp_idx] = answer
        else:
            ans_list[resp_idx] = ""
    return ans_list


def get_gsm8k_examples(args):
    path = args.dataset_path
    with open(path) as fh:
        examples = [json.loads(line) for line in fh.readlines() if line]

    for idx, ex in enumerate(examples):
        ex.update(question=ex["question"] + "\n")
        ex.update(answer=ex["answer"] + "<|endoftext|>")
        ex.update(q_idx = idx)
        # ex.update(answer=ex["answer"])

    if args.setting == "fair":
        split = 'train'
    else:
        split = 'test'
    print(f"{len(examples)} {split} examples")
    return examples

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

def extract_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS

def generate_prompt_file(args, result, output_dir=None):
    examples = get_gsm8k_examples(args)
    output_prompt = {"prompt" : []}

    experiment_seeds = [0, 1, 42]

    for seed in experiment_seeds:
        if args.sort_by == "variance":
            # take the first 8 samples according to sorted list
            selected_idx = [0,1,2,3,4,5,6,7]
            selected_questions = [result[i][0] for i in selected_idx]
        elif args.sort_by == "disagreement":
            set_random_seed(seed)
            args.random_seed = seed
            count = 0
            for item in result:
                if len(item[-1]) == args.num_trails:
                    count += 1
                else:
                    break
            selected_idx = random.sample(range(0,count), 8)
            selected_questions = [result[i][0] for i in selected_idx]

        for idx in selected_questions:
            prompt = {}
            prompt['question'] = "Q: " + examples[idx]['question'].replace("\n", "") + "\nA: "
            
            # clean the rationale
            t = examples[idx]['answer'][:examples[idx]['answer'].find('\n####')]
            while t.find("<<") != -1 and t.find(">>") != -1:
                t = t[:t.find("<<")] + "" + t[t.find(">>")+2:]

            t = t.replace("\n\n", "\n")
            t = t.replace("\n", ". ").replace("..", ".")
            prompt['rationale'] = t

            prompt['pred_ans'] = extract_answer(examples[idx]['answer'])

            output_prompt['prompt'].append(prompt)
        
        with open(f'./test_prompts_{args.setting}/{args.method}_k={args.num_trails}_seed{seed}', 'w') as f:
            json.dump(result, f, indent=4)
        
        if args.sort_by == "variance":
            return


def find_most_frequent(arr, n):
    # method 1: return max(arr[:n], key=arr.count)
    # method 2:
    arr_acounts = Counter(arr[:n])
    most_frequent_item, frequency = arr_acounts.most_common(1)[0]
    return frequency, most_frequent_item