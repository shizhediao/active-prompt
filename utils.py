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

# put your API key in the list
API_KEY_POOL = [
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
    {"cur_index":0, "keys":API_KEY_POOL[0:7]},
    {"cur_index":0, "keys":API_KEY_POOL[7:14]},
    {"cur_index":0, "keys":API_KEY_POOL[14:21]},
    {"cur_index":0, "keys":API_KEY_POOL[21:28]}
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
                print(f"Reason: {errno[1]}\n")

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
        pass
    elif args.dataset == "asdiv":
        pass
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
        if args.dataset == "gsm8k":
            temp = temp.replace(",", "")
            temp = [s for s in re.findall(r'-?\d+\.?\d*', temp)]
        elif args.dataset == "aqua":
            temp = re.findall(r'A|B|C|D|E', temp)
        if len(temp) != 0:
            answer = temp[-1]
            # if there is . at the end of answer, remove it
            # e.g. answer = 64.
            if answer != "":
                if answer[-1] == ".":
                    answer = answer[:-1]

            # round the answer to nearest integer
            if args.dataset == "gsm8k":
                try:
                    answer = str(round(float(answer)))
                except:
                    answer = "" # no sol or sol doesn't have valid format
            
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