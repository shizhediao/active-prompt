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
import time
from API_POOL_REPO import *
import asyncio

# put your API key in the list
# NO_SOLUTION = '<NO_SOL>'
NO_SOLUTION = '-10086' # use this when calculating numerical results

# set the random seed for reproducibility
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


async def dispatch_openai_requests(
    #messages_list: list[list[dict[str,Any]]],
    messages_list,
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
):
# -> list[str]
    """Dispatches requests to OpenAI API asynchronously.
    
    Args:
        messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        model: OpenAI model to use.
        temperature: Temperature to use for the model.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use for the model.

    Returns:
        List of responses from OpenAI API.
    """
    async_responses = [
        openai.ChatCompletion.acreate(
            model=model,
            messages=x,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        for x in messages_list
    ]
    return await asyncio.gather(*async_responses)


def chatgpt_request(model:str, message_list:list, max_tokens:int, temperature=0.7, sleep=3, worker_id=None, API_PARTITION_POOL=None):
    resp = None
    done = False
    while not done:
        try:
            # random select key
            # key_index = random.randint(0, NUM_API_KEYS - 1)
            # openai.api_key = API_KEY_POOL[key_index]

            # api key polling request
            key_index = API_PARTITION_POOL[worker_id]['cur_index']

            # if use API KEY POOL, use this line
            openai.api_key = API_PARTITION_POOL[worker_id]["keys"][key_index]

            # if only use one key, then comment the above line and use this one
            # openai.api_key = "YOUR_API_KEY"

            predictions = asyncio.run(
                dispatch_openai_requests(
                    messages_list=message_list,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=1.0,
                )
            )

            resp = {'choices':[]}
            for pred in predictions:
                resp['choices'].append(pred['choices'][0])
            
            done = True
            key_index += 1
            if key_index == len(API_PARTITION_POOL[worker_id]['keys']):
                API_PARTITION_POOL[worker_id]['cur_index'] = 0
            else:
                API_PARTITION_POOL[worker_id]['cur_index'] = key_index
        except:
            errno = sys.exc_info()[:2]
            if errno[0] == openai.error.InvalidRequestError:
                # print(f"Invalid Request\nPrompt: {message_list}\n")
                print(f"Reason: {errno[1]}")
                key_index = API_PARTITION_POOL[worker_id]['cur_index']
                print(f"invalid key: {API_PARTITION_POOL[worker_id]['keys'][key_index]}")
                assert False
            else:
                print(f"Error: {errno[0]}\n")
                print(f"Reason: {errno[1]}\n")
                print(f"key_index: {key_index}")
                print(f"worker_id: {worker_id}")
                print(f"len of pool: {len(API_PARTITION_POOL[worker_id]['keys'])}")

            key_index = API_PARTITION_POOL[worker_id]['cur_index']
            key_index += 1
            if key_index == len(API_PARTITION_POOL[worker_id]['keys']):
                API_PARTITION_POOL[worker_id]['cur_index'] = 0
            else:
                API_PARTITION_POOL[worker_id]['cur_index'] = key_index
            time.sleep(sleep)
    return resp



# pass in a list of prompts and returns a response body contains a list of responses
def GPT3_request(model:str, input_prompt:list, max_tokens:int, temperature=0.7, stop=None, worker_id=None, API_PARTITION_POOL=None):
    resp = None
    done = False
    while not done:
        try:
            # random select key
            # key_index = random.randint(0, NUM_API_KEYS - 1)
            # openai.api_key = API_KEY_POOL[key_index]

            # api key polling request
            key_index = API_PARTITION_POOL[worker_id]['cur_index']

            # if use API KEY POOL, use this line
            openai.api_key = API_PARTITION_POOL[worker_id]["keys"][key_index]

            # if only use one key, then comment the above line and use this one
            # openai.api_key = "YOUR_API_KEY"


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
            key_index += 1
            if key_index == len(API_PARTITION_POOL[worker_id]['keys']):
                API_PARTITION_POOL[worker_id]['cur_index'] = 0
            else:
                API_PARTITION_POOL[worker_id]['cur_index'] = key_index
        except:
            errno = sys.exc_info()[:2]
            if errno[0] == openai.error.InvalidRequestError:
                print(f"Invalid Request\nPrompt: {input_prompt}\n")
                print(f"Reason: {errno[1]}")
                key_index = API_PARTITION_POOL[worker_id]['cur_index']
                print(f"invalid key: {API_PARTITION_POOL[worker_id]['keys'][key_index]}")
                assert False
            else:
                print(f"Error: {errno[0]}\n")
                print(f"Reason: {errno[1]}\n")
                print(f"key_index: {key_index}")
                print(f"worker_id: {worker_id}")
                print(f"len of pool: {len(API_PARTITION_POOL[worker_id]['keys'])}")

            key_index = API_PARTITION_POOL[worker_id]['cur_index']
            key_index += 1
            if key_index == len(API_PARTITION_POOL[worker_id]['keys']):
                API_PARTITION_POOL[worker_id]['cur_index'] = 0
            else:
                API_PARTITION_POOL[worker_id]['cur_index'] = key_index
            time.sleep(3)
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
                answers.append(json_res["answer"].split("#### ")[-1].replace(",", ""))
    elif args.dataset == "aqua":
        with open(args.dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                qes = json_res["question"].strip() + " Answer Choices:"

                for opt in json_res["options"]:
                    opt = opt.replace(')', ') ')
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
            json_data = json.load(f)["Instances"]
            for line in json_data:
                q = line['input'].strip()
                a = line['output'][0]
                questions.append(q)
                answers.append(a)
    elif args.dataset in ("addsub", "singleeq", "multiarith"):
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
        if 'task' in args.dataset_path:
            with open(args.dataset_path) as f:
                json_data = json.load(f)["examples"]
                for line in json_data:
                    q = line["input"].strip()
                    a = int(line["target_scores"]["Yes"])
                    if a == 1:
                        a = "yes"
                    else:
                        a = "no"
                    questions.append(q)
                    answers.append(a)
        else:
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
    elif args.dataset == 'time_zone':
        with open(args.dataset_path) as f:
            json_data = json.load(f)
            for line in json_data:
                q = line['question'].strip()
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
            if args.dataset == "strategyqa":
                prompt_text += x[i] + " " + z[i] + " " + \
                            "So the answer is" + " " + y[i] + ".\n\n"
            else:
                prompt_text += x[i] + " " + z[i] + " " + \
                            args.direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"
        else:
            prompt_text += x[i] + " " + args.direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"
    return prompt_text


def answer_extraction(args, responses):
    ans_list = ["" for i in range(len(responses['choices']))]
    for resp_idx in range(len(responses['choices'])):
        if args.model == 'gpt-3.5-turbo':
            temp = responses['choices'][resp_idx]['message']['content']
        else:
            temp = responses['choices'][resp_idx].text
        if args.dataset in ("gsm8k", "svamp", "asdiv", "addsub", "singleeq", "multiarith"):
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
        elif args.dataset in ('time_zone'):
            temp = temp.split('The answer is ')[-1].replace('.', '')
            temp = [temp]
        if len(temp) != 0:
            answer = temp[-1]
            # if there is . at the end of answer, remove it
            # e.g. answer = 64.
            if answer != "":
                if answer[-1] == ".":
                    answer = answer[:-1]

            # round the answer to nearest integer
            if args.dataset in ("gsm8k", "svamp"):
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


def find_most_frequent(arr, n):
    # method 1: return max(arr[:n], key=arr.count)
    # method 2:
    arr_acounts = Counter(arr[:n])
    most_frequent_item, frequency = arr_acounts.most_common(1)[0]
    return frequency, most_frequent_item