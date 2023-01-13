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

# put your API key in the list
API_KEY_POOL = None
NUM_API_KEYS = None
# NO_SOLUTION = '<NO_SOL>'
NO_SOLUTION = '-10086'

API_PARTITION_POOL = None

# API_PARTITION_POOL = [
#     {"cur_index":0, "keys":API_KEY_POOL[0:60]},
#     {"cur_index":0, "keys":API_KEY_POOL[60:120]},
#     {"cur_index":0, "keys":API_KEY_POOL[120:180]},
#     {"cur_index":0, "keys":API_KEY_POOL[180:240]},
#     {"cur_index":0, "keys":API_KEY_POOL[240:296]},
# ]

# set the random seed for reproducibility
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# pass in a list of prompts and returns a response body contains a list of responses
def GPT3_request(model:str, input_prompt:list, max_tokens:int, temperature=0.7, stop=None, worker_id=None, API_PARTITION_POOL=None):
    resp = None
    done = False
    while not done:
        try:
            # key_index = random.randint(0, NUM_API_KEYS - 1)
            # openai.api_key = API_KEY_POOL[key_index]
            key_index = API_PARTITION_POOL[worker_id]['cur_index']
            openai.api_key = API_PARTITION_POOL[worker_id]["keys"][key_index]
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
            json_data = json.load(f)["Instances"]
            for line in json_data:
                q = line['input'].strip()
                a = line['output'][0]
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