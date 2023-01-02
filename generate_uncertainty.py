# This file used to generate uncertainty score for each question
from utils import *
from pathlib import Path
import re
from tqdm import tqdm
import concurrent
import time
import argparse
import numpy as np


def main():
    args = arg_parser()
    print('*****************************')
    print(args)
    print('*****************************')

    set_random_seed(args.random_seed)

    dataloader = create_dataloader(args)

    start =time.time()
    result = create_uncertainty(args, dataloader)

    # output the results
    with open(args.output_dir, 'w') as f:
        for item in result:
            f.write(f"{item}, uncertainty: {len(item[-1])}, variance: {item[1]}\n")

    end = time.time()
    print('Total Execution Time: ', end - start, " seconds")

# give the questions bank, and how many trails to run for each question
def generate_uncertainty_batch(args, question_pool, batch_limit=None):
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
        uncertainty_batch = [[qes['question_idx'], float, {}] for qes in batch]

        for trail in range(args.num_trails):
            # construct first stage zero-shot prompt (step by step)
            prompt_list = []
            for example in batch:
                if args.method == "few_shot_cot":
                    prompt = given_prompt + "Q: " + example['question'] + "A:"
                elif args.method == "zero_shot_cot":
                    prompt = "Q: " + example['question'] + "A: Let's think step by step."
                prompt_list.append(prompt)

            # get the first stage zero-shot result
            responses = GPT3_request(model=args.model, input_prompt=prompt_list, max_tokens=args.max_length_cot, stop='Q:')
                
            # construct second stage prompt, to generate a single arabic num answer
            if args.method == "zero_shot_cot":
                for i in range(len(prompt_list)):
                    prompt_list[i] += responses.choices[i].text + "\nTherefore, the answer (arabic numerals) is"

                # get the second stage zero-shot rationale result -> arabic num answer
                responses = GPT3_request(model=args.model, input_prompt=prompt_list, max_tokens=args.max_length_cot, stop='.')

            # check uncertainty
            for resp_idx in range(len(responses['choices'])):
                answer = responses['choices'][resp_idx].text
                answer = answer.replace("$","").replace(",","").replace("%","")
                answer = [s for s in re.findall(r'-?\d+\.?\d*', answer)]
                try:
                    answer = answer[-1]

                    if answer != "":
                        if answer[-1] == ".":
                            answer = answer[:-1]

                    answer = str(round(float(answer)))
                    
                    if answer in uncertainty_batch[resp_idx][-1]:
                        uncertainty_batch[resp_idx][-1][answer] += 1 # increment answer occurrence
                    else:
                        uncertainty_batch[resp_idx][-1][answer] = 1 # first occurence
                except:
                    # Handle no solution case
                    if NO_SOLUTION in uncertainty_batch[resp_idx][-1]:
                        uncertainty_batch[resp_idx][-1][NO_SOLUTION] += 1
                    else:
                        uncertainty_batch[resp_idx][-1][NO_SOLUTION] = 1

        # calculate variance for each question
        for uncertainty in uncertainty_batch:
            ans_list = []
            for ans, occurs in uncertainty[-1].items():
                for i in range(int(occurs)):
                    ans_list.append(float(ans))
            uncertainty[1] = np.var(ans_list)
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

        for batch_count in range(0, len(question_bank), work_size):
            # handle last batch
            if len(question_bank) - batch_count > work_size:
                futures.append(executor.submit(generate_uncertainty_batch, args, question_bank[batch_count:batch_count + work_size], 
                None))
            else:
                futures.append(executor.submit(generate_uncertainty_batch, args, question_pool[batch_count:len(question_bank)], None))

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
        result.sort(key=lambda x: -len(x[-1]))
    elif args.sort_by == "variance":
        result.sort(key=lambda x: -x[1])
    return result


def arg_parser():
    parser = argparse.ArgumentParser(description="Uncertainty_Generation")
    parser.add_argument("--random_seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--dataset", type=str, default="gsm8k", choices=["gsm8k","svamp", "aqua"], help="dataset to inference"
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
        "--temperature", type=float, default=0, help=""
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
    
    args = parser.parse_args()
    
    # Fill in the dataset path
    if args.dataset == "gsm8k":
        args.dataset_path = ""
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
        args.datset_size = 1319
    elif args.dataset == "svamp":
        raise ValueError("dataset is not properly defined ...")
    elif args.dataset == "aqua":
        args.dataset_path = ""
        args.direct_answer_trigger = "The answer is"
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