# This file used to generate uncertainty score for each question
from utils import *
import time
import argparse
import numpy as np
import json
from scipy.stats import entropy

def main():
    args = arg_parser()
    print('*****************************')
    print(args)
    print('*****************************')

    print(f"API_KEY: {API_KEY}")
    set_random_seed(args.random_seed)

    dataloader = create_dataloader(args)

    if args.dataset_size > 1000:
        dataloader = dataloader[:1000] # only take 1000 questions randomly to annotate, randomness decided by seed
    print(f"Dataloader size: {len(dataloader)}")


    if args.qes_limit == 0:
        args.qes_limit = len(dataloader)

    start =time.time()
    result = create_uncertainty(args, dataloader)
    end = time.time()
    print('Total Execution Time: ', end - start, " seconds")

    # output the results
    path = f"{args.output_dir}/uncertainty_result_{args.dataset}_k{args.num_trails}.txt"
    with open(path, 'w') as f:
        try:
            f.write(json.dumps(result, indent=4))
        except:
            for item in result:
                try:
                    if args.dataset in ("gsm8k", "asdiv", "svamp", "singleeq", "addsub", "multiarith"):
                        f.write(f"{item}, uncertainty: {len(item[-1])}, variance: {item[1]}\n")
                    else:
                        f.write(f"{item}, uncertainty: {len(item[-1])}\n")
                except:
                    pass


def generate_uncertainty_qes(args, question):
    if args.method == "few_shot_cot":
        given_prompt = create_input_prompt(args, True)

    if args.dataset in ("gsm8k", "asdiv", "svamp", "singleeq", "addsub", "multiarith"):
        # the float is reserved for variance calculation result
        uncertainty_record = {'dataset_idx':question['question_idx'], 'variance':float, 'entropy':float, 'occurrence':{}}
    elif args.dataset == "strategyqa":
        uncertainty_record = {'dataset_idx':question['question_idx'], 'entropy':float, 'occurrence':{"yes":0, "no":0}}
    else:
        uncertainty_record = {'dataset_idx':question['question_idx'], 'entropy':float, 'occurrence':{}}

    for trail in range(args.num_trails):
        # if zero-shot to generate uncertainty, construct first stage zero-shot prompt (step by step)
        if args.method == "few_shot_cot":
            prompt = given_prompt + "Q: " + question['question'] + "\nA: Let's think step by step."
        elif args.method == "zero_shot_cot":
            prompt = "Q: " + question['question'] + "\nA: Let's think step by step."
        prompt_list = [prompt]

        # if use zero-shot, here we get the first stage zero-shot result
        # if not use zero-shot, here we get the final output
        responses = GPT3_request(model=args.model, input_prompt=prompt_list, max_tokens=args.max_length_cot, time_interval=args.api_time_interval
                                , temperature=args.temperature , stop=['Question:', "Q:"])
            
        # construct second stage prompt, to generate a single arabic num answer
        if args.method == "zero_shot_cot":
            prompt_list[0] += responses['choices'][0]['text'] + args.direct_answer_trigger

            # get the second stage zero-shot rationale result -> arabic num answer
            responses = GPT3_request(model=args.model, input_prompt=prompt_list, max_tokens=args.max_length_cot, time_interval=args.api_time_interval,
                                    temperature=args.temperature, stop='.')

        # extract the pred answer
        pred_ans = answer_extraction(args, responses)

        # check uncertainty
        if pred_ans != "":
            if pred_ans in uncertainty_record['occurrence']:
                uncertainty_record['occurrence'][pred_ans] += 1 # increment answer occurrence
            else:
                uncertainty_record['occurrence'][pred_ans] = 1 # first occurence
        else:
            # Handle no solution case
            if NO_SOLUTION in uncertainty_record['occurrence']:
                uncertainty_record['occurrence'][NO_SOLUTION] += 1
            else:
                uncertainty_record['occurrence'][NO_SOLUTION] = 1

    # calculate the variance for the question (only applied to datasets with numerical answer)
    if args.dataset in ("gsm8k", "asdiv", "svamp", "singleeq", "addsub", "multiarith"):
        ans_list = []
        for ans, occurs in uncertainty_record['occurrence'].items():
            for i in range(int(occurs)):
                ans_list.append(float(ans))
        uncertainty_record['variance'] = np.var(ans_list)
        
    # calculate the entropy for all dataset
    frequency_list = list(uncertainty_record['occurrence'].values())
    uncertainty_record['entropy'] = entropy(frequency_list)

    # calculate the disagreement for all dataset
    uncertainty_record['disagreement'] = len(uncertainty_record['occurrence'])
    
    return uncertainty_record


# return a sorted list by uncertainty from high to low
def create_uncertainty(args, questions):
    result = []
    count = 0

    for qes in questions:
        if count == args.qes_limit:
            break
        uncertainty_record = generate_uncertainty_qes(args, qes)
        result.append(uncertainty_record)
        count += 1

    if args.sort_by == "disagreement":
        if args.dataset == "strategyqa":
            try:
                # sort based on the entropy or the difference between yes and no answers
                result.sort(key=lambda x: abs(x['occurrence']['yes'] - x['occurrence']['no']))
            except:
                # sort by disagreement
                result.sort(key=lambda x: -len(x['occurrence']))
        else:
            result.sort(key=lambda x: -len(x['occurrence']))
    elif args.sort_by == "variance" and args.dataset in ("gsm8k", "asdiv", "svamp", "singleeq", "addsub", "multiarith"):
        # sort by variance
        result.sort(key=lambda x: -x['variance'])
    elif args.sort_by == "entropy" :
        result.sort(key=lambda x:-x['entropy'])
    return result


def arg_parser():
    parser = argparse.ArgumentParser(description="Uncertainty_Generation")
    parser.add_argument("--random_seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--dataset", type=str, default="gsm8k", choices=["gsm8k","svamp", "aqua", "csqa", "last_letters", "strategyqa", "asdiv", "singleeq", "addsub", "multiarith", "time_zone"], help="dataset to inference"
    )
    parser.add_argument(
        "--prompt_path", type=str, default="./basic_cot_prompts/math_word_problems", help="prompts to use"
    )
    parser.add_argument(
        "--model", type=str, default="code-davinci-002", choices=["text-davinci-002", "code-davinci-002"], help="model used for decoding."
    )
    parser.add_argument(
        "--method", type=str, default="few_shot_cot", choices=["zero_shot_cot", "few_shot_cot"], help="method"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./", help="output directory"
    )
    parser.add_argument(
        "--max_length_cot", type=int, default=256, help="maximum length of output tokens by model for reasoning extraction"
    )
    parser.add_argument(
        "--qes_limit", type=int, default=0, help="whether to limit test dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for testing."
    )
    parser.add_argument(
        "--api_time_interval", type=float, default=1.0, help="how many seconds sleep between each request"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help=""
    )
    parser.add_argument(
        "--num_trails", type=int, default=5, help="number of trails to run for each qeestion"
    )
    parser.add_argument(
        "--sort_by", type=str, default='disagreement', choices=['disagreement', 'variance', 'entropy'], help="sort the final result by given option"
    )
    parser.add_argument(
        "--concat_length", type=int, default=2, help='Used for task last_letters, indicates length of last letter concat'
    )
    
    args = parser.parse_args()
    
    # Fill in the dataset path
    if args.dataset == "gsm8k":
        args.dataset_path = "./dataset/GSM8K/train.jsonl" # train data path
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "svamp":
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "asdiv":
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "singleeq":
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "addsub":
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "multiarith":
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "aqua":
        args.dataset_path = "./dataset/AQuA/train.json" # train data path
        args.direct_answer_trigger = "\nThe answer is"
    elif args.dataset == "csqa":
        args.dataset_path = "./dataset/CSQA/train_rand_split.jsonl" # train data path
        args.direct_answer_trigger = "\nSo the answer is"
    elif args.dataset == "strategyqa":
        args.dataset_path = "./dataset/strategyQA/train.json" # train data path
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif args.dataset == "last_letters":
        args.dataset_path = "./dataset/last_letters/last_letters_train2.json" # train data path
        args.direct_answer_trigger = "\nTherefore, the answer is"
    elif args.dataset == 'time_zone':
        args.dataset_path = "./dataset/timezone_convert/timezone_convertion_train.json"
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