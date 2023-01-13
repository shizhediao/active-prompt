# This file used to generate uncertainty score for each question
from utils import *
from pathlib import Path
import re
from tqdm import tqdm
import concurrent
import time
import argparse
import numpy as np
import math
from API_POOL_REPO import *


def main():
    args = arg_parser()
    print('*****************************')
    print(args)
    print('*****************************')

    print(f"NUM_API_KEYS: {NUM_API_KEYS}")
    set_random_seed(args.random_seed)

    dataloader = create_dataloader(args)

    if args.setting == "fair":
        if args.dataset_size > 1000:
            batch_amount = math.floor(1000 / args.minibatch_size)
            dataloader = dataloader[:batch_amount] # only take 1000 questions randomly to annotate, randomness decided by seed
        print(f"Use Fair Setting, dataloader size: {len(dataloader)}")

    start =time.time()
    result = create_uncertainty(args, dataloader)

    # output the results
    with open(args.output_dir, 'w') as f:
        for item in result:
            f.write(f"{item}, uncertainty: {len(item[-1])}, variance: {item[1]}\n")

    end = time.time()
    print('Total Execution Time: ', end - start, " seconds")

    # if args.dataset == "gsm8k":
    #     print("Start generates prompt files")
    #     generate_prompt_file(args, result)


# give the questions bank, and how many trails to run for each question
def generate_uncertainty_batch(args, question_pool, batch_limit=None, worker_id=None):
    # maintain a list of uncertaintie for each question
    # each element is a tuple of 2 element, tuple[0] is quesiton idx
    # tuple[1] is the uncertainty dict, record how many occurrences of each case

    if args.method == "few_shot_cot":
        given_prompt = create_input_prompt(args, True)
    uncertainty_list = []
    batch_count = 0
    for _, batch in enumerate(question_pool):
        if batch_limit is not None and batch_count == batch_limit:
            break
        if args.dataset == "gsm8k":
            uncertainty_batch = [[qes['question_idx'], float, {}] for qes in batch]
        elif args.dataset == "strategyqa":
            uncertainty_batch = [[qes['question_idx'], {"yes":0, "no":0}] for qes in batch]
        else:
            uncertainty_batch = [[qes['question_idx'], {}] for qes in batch]
            # NO_SOLUTION = '<NO_SOL>'

        for trail in range(args.num_trails):
            # construct first stage zero-shot prompt (step by step)
            prompt_list = []
            for example in batch:
                if args.method == "few_shot_cot":
                    prompt = given_prompt + "Question: " + example['question'] + "\nA: Let's think step by step."
                elif args.method == "zero_shot_cot":
                    prompt = "Question: " + example['question'] + "\nA: Let's think step by step."
                prompt_list.append(prompt)

            # get the first stage zero-shot result
            responses = GPT3_request(model=args.model, input_prompt=prompt_list, max_tokens=args.max_length_cot, stop=['Question:', "Q:"], worker_id=worker_id,
            API_PARTITION_POOL=API_PARTITION_POOL)
                
            # construct second stage prompt, to generate a single arabic num answer
            if args.method == "zero_shot_cot":
                for i in range(len(prompt_list)):
                    prompt_list[i] += responses.choices[i].text + args.direct_answer_trigger

                # get the second stage zero-shot rationale result -> arabic num answer
                responses = GPT3_request(model=args.model, input_prompt=prompt_list, max_tokens=args.max_length_cot, stop='.', worker_id=worker_id,
                API_PARTITION_POOL=API_PARTITION_POOL)

            # check uncertainty
            ans_list = answer_extraction(args, responses)

            for ans_idx in range(len(ans_list)):
                answer = ans_list[ans_idx]
                if answer != "":
                    if answer in uncertainty_batch[ans_idx][-1]:
                        uncertainty_batch[ans_idx][-1][answer] += 1 # increment answer occurrence
                    else:
                        uncertainty_batch[ans_idx][-1][answer] = 1 # first occurence
                else:
                    # Handle no solution case
                    if NO_SOLUTION in uncertainty_batch[ans_idx][-1]:
                        uncertainty_batch[ans_idx][-1][NO_SOLUTION] += 1
                    else:
                        uncertainty_batch[ans_idx][-1][NO_SOLUTION] = 1

        # calculate variance for each question
        if args.dataset == "gsm8k":
            for uncertainty in uncertainty_batch:
                ans_list = []
                for ans, occurs in uncertainty[-1].items():
                    for i in range(int(occurs)):
                        ans_list.append(float(ans))
                uncertainty[1] = np.var(ans_list)
                uncertainty_list.append(uncertainty)
        else:
            for uncertainty in uncertainty_batch:
                uncertainty_list.append(uncertainty)
        batch_count += 1
    
    return uncertainty_list


def generate_uncertainty_parallel(args, question_pool, batch_limit):
    # MAX_WORKERS = args.max_num_workers
    st = time.time()
    MAX_WORKERS = args.max_num_workers

    if batch_limit is None or batch_limit > len(question_pool) or batch_limit == 0:
        batch_limit = len(question_pool)

    question_bank = question_pool[:batch_limit]

    if len(question_bank) < MAX_WORKERS:
        work_size = 1
    else:
        work_size = int(len(question_bank) / MAX_WORKERS)
    print(f"batch limit: {batch_limit}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []

        idx_count = 0
        for batch_count in range(0, len(question_bank), work_size):
            # handle last batch
            worker_id = idx_count % len(API_PARTITION_POOL)
            if len(question_bank) - batch_count > work_size:
                futures.append(executor.submit(generate_uncertainty_batch, args, question_bank[batch_count:batch_count + work_size], 
                None, worker_id))
            else:
                futures.append(executor.submit(generate_uncertainty_batch, args, question_pool[batch_count:len(question_bank)], None, worker_id))
            idx_count += 1

        uncertainty_list = []
        for future in concurrent.futures.as_completed(futures):
            for item in future.result():
                uncertainty_list.append(item)
    end = time.time()
    print("Partition Execution time: ", end-st, " seconds")
    return uncertainty_list


# divide the entire dataloader into chunks by partition size
# process each chunk in parallel
# return a sorted list by uncertainty from high to low
def create_uncertainty(args, questions):
    result = []
    count = 0
    while count < len(questions):
        if len(questions) - count > args.partition:
            temp = generate_uncertainty_parallel(args, questions[count:count+args.partition], args.partition)
            count += args.partition
            print(count)
        else:
            temp = generate_uncertainty_parallel(args, questions[count:len(questions)], len(questions)-count)
            count += len(questions) - count
            print(count)

        for item in temp:
            result.append(item)
        time.sleep(5)
    if args.sort_by == "disagreement":
        if args.dataset == "strategyqa":
            try:
                result.sort(key=lambda x: abs(x[-1]['yes'] - x[-1]['no']))
            except:
                result.sort(key=lambda x: -len(x[-1]))
        else:
            result.sort(key=lambda x: -len(x[-1]))
    elif args.sort_by == "variance":
        result.sort(key=lambda x: -x[1])
    return result


def arg_parser():
    parser = argparse.ArgumentParser(description="Uncertainty_Generation")
    parser.add_argument("--random_seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--dataset", type=str, default="gsm8k", choices=["gsm8k","svamp", "aqua", "csqa", "last_letters", "strategyqa"], help="dataset to inference"
    )
    parser.add_argument(
        "--prompt_path", type=str, default="prompts/active", help="type of prompts to use"
    )
    parser.add_argument("--minibatch_size", type=int, default=1, choices=[1, 5, 10], help="batch size (num_prompts) for each request")
    
    parser.add_argument("--max_num_workers", type=int, default=0, help="maximum number of workers for inference")
    
    parser.add_argument(
        "--model", type=str, default="code-davinci-002", choices=["text-davinci-002", "code-davinci-002"], help="model used for decoding."
    )
    
    parser.add_argument(
        "--method", type=str, default="few_shot_cot", choices=["zero_shot_cot", "few_shot_cot"], help="method"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./uncertainty_result", help="output directory"
    )
    parser.add_argument(
        "--max_length_cot", type=int, default=256, help="maximum length of output tokens by model for reasoning extraction"
    )
    parser.add_argument(
        "--limit_batch_size", type=int, default=0, help="whether to limit test dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for testing."
    )
    parser.add_argument(
        "--api_time_interval", type=float, default=1.0, help="how many seconds sleep between each request"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help=""
    )
    parser.add_argument(
        "--log_dir", type=str, default="./log/", help="log directory"
    )
    parser.add_argument(
        "--num_trails", type=int, default=5, help="number of trails to run for each qeestion"
    )
    parser.add_argument(
        "--partition", type=int, default=10, help="number of chunks to partition the dataloader to avoid rate limit errors"
    )
    parser.add_argument(
        "--sort_by", type=str, default='disagreement', choices=['disagreement', 'variance'], help="sort the final result by given option"
    )
    parser.add_argument(
        "--setting", type=str, default='unfair', choices=['fair', 'unfair'], help="decide whether annotate on test data or not"
    )
    parser.add_argument(
        "--concat_length", type=int, default=2, help='Used for task last_letters, indicates length of last letter concat'
    )
    parser.add_argument(
        "--api_pool_idx", type=int, default=1, choices=[0,1,2,3], help='Choose which API pool to use'
    )
    
    args = parser.parse_args()

    global API_KEY_POOL
    global NUM_API_KEYS
    global API_PARTITION_POOL
    API_KEY_POOL = POOL_REPO[args.api_pool_idx]
    NUM_API_KEYS = len(API_KEY_POOL)

    API_PARTITION_POOL = [
        {"cur_index":0, "keys":API_KEY_POOL[0:40]},
        {"cur_index":0, "keys":API_KEY_POOL[40:80]},
        {"cur_index":0, "keys":API_KEY_POOL[80:120]},
        {"cur_index":0, "keys":API_KEY_POOL[120:160]},
        {"cur_index":0, "keys":API_KEY_POOL[160:]},
    ]
    
    # Fill in the dataset path
    if args.dataset == "gsm8k":
        if args.setting == "unfair":
            args.dataset_path = r"D:\HKUST_NLP_Research\cot_active_learning\grade_school_math\data\test.jsonl" # test data path
        elif args.setting == "fair":
            args.dataset_path = r"D:\HKUST_NLP_Research\cot_active_learning\grade_school_math\data\train.jsonl" # train data path
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
        # args.datset_size = 1319
    elif args.dataset == "svamp":
        raise ValueError("dataset is not properly defined ...")
    elif args.dataset == "aqua":
        if args.setting == "unfair":
            args.dataset_path = r"D:\HKUST_NLP_Research\cot_active_learning\AQuA\test.json" # test data path
        elif args.setting == "fair":
            args.dataset_path = r"D:\HKUST_NLP_Research\cot_active_learning\AQuA\train.json" # train data path
            # args.dataset_path = r"D:\HKUST_NLP_Research\cot_active_learning\AQuA\dev.json" # dev data path
        args.direct_answer_trigger = "\nThe answer is"
    elif args.dataset == "csqa":
        if args.setting == "unfair":
            args.dataset_path = r"D:\HKUST_NLP_Research\cot_active_learning\CSQA\dev_rand_split.jsonl" # test(dev) data path
        elif args.setting == "fair":
            args.dataset_path = r"D:\HKUST_NLP_Research\cot_active_learning\CSQA\train_rand_split.jsonl" # train data path
        args.direct_answer_trigger = "\nSo the answer is"
    elif args.dataset == "strategyqa":
        if args.setting == "unfair":
            args.dataset_path = r"D:\HKUST_NLP_Research\cot_active_learning\strategyqa\dev.json" # test(dev) data path
        elif args.setting == "fair":
            args.dataset_path = r"D:\HKUST_NLP_Research\cot_active_learning\strategyqa\train.json" # train data path
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif args.dataset == "last_letters":
        if args.setting == "unfair":
            args.dataset_path = r"D:\HKUST_NLP_Research\cot_active_learning\last_letters\last_letters_test.json" # test(dev) data path
        elif args.setting == "fair":
            args.dataset_path = r"D:\HKUST_NLP_Research\cot_active_learning\last_letters\last_letters_train2.json" # train data path
        args.direct_answer_trigger = "\nTherefore, the answer is"
    else:
        raise ValueError("dataset is not properly defined ...")
        
    # "Therefore, the answer ..." -> "The answer ..."
    trigger = args.direct_answer_trigger.replace("\nTherefore, ", "")
    args.direct_answer_trigger_for_zeroshot = trigger[0].upper() + trigger[1:]
    args.direct_answer_trigger_for_zeroshot_cot = args.direct_answer_trigger
    args.direct_answer_trigger_for_fewshot = "The answer is"
    args.cot_trigger = "Let's think step by step."
    
    return args


if __name__ == "__main__":
    main()