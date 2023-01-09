from utils import *
from pathlib import Path
import re
from tqdm import tqdm
import concurrent
import time
import argparse


def main():
    # load arguments from terminal
    args = arg_parser()
    print('*****************************')
    print(args)
    print('*****************************')

    set_random_seed(args.random_seed)

    # load dataset
    dataloader = create_dataloader(args)

    if args.method == "few_shot":
        input_prompt = create_input_prompt(args, cot_flag=False)
    elif args.method == "few_shot_cot" or args.method == "auto_cot" or args.method == "active_cot":
        input_prompt = create_input_prompt(args, cot_flag=True)
    else:
        pass

    start = time.time()
    if args.max_num_workers > 1:
        print("Parallel Inference")
        inference_cot_parallel(args, dataloader, input_prompt) # run cot parallel
    else:
        print("Single Thread Inference")
        if args.limit_batch_size == 0:
            args.limit_batch_size = len(dataloader)
        correct = inference_cot(args, dataloader, args.limit_batch_size, input_prompt)
        print(f"correct: {correct}")
        print(f"total: {args.limit_batch_size * args.minibatch_size}")
        print(f"Accuracy: {correct / (args.limit_batch_size * args.minibatch_size)}")
    end = time.time()
    print(f"Execution time: {end - start} seconds")


def inference_cot_parallel(args, question_pool, given_prompt):
    MAX_WORKERS = args.max_num_workers
    batch_limit = args.limit_batch_size
    # batch_count = 0

    if batch_limit is None:
        total = args.dataset_size
    elif batch_limit > len(question_pool) or batch_limit == 0:
        batch_limit = len(question_pool)
        total = args.dataset_size
    else:
        total = batch_limit * args.minibatch_size

    question_bank = question_pool[:batch_limit]

    work_size = int(len(question_bank) / MAX_WORKERS)
    correct = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []

        for batch_count in range(0, len(question_bank), work_size):
            # handle last batch
            if len(question_bank) - batch_count > work_size:
                futures.append(executor.submit(inference_cot, args, question_pool[batch_count:batch_count + work_size], None, given_prompt))
            else:
                futures.append(executor.submit(inference_cot, args, question_pool[batch_count:len(question_bank)], None, given_prompt))

        for future in concurrent.futures.as_completed(futures):
            correct += future.result()
        print(f"correct: {correct}")
        print(f"total: {total}")
        print(f"Accuracy: {correct / total}")
    return correct / total


def inference_cot(args, question_pool, batch_limit, given_prompt):
    correct = 0
    batch_count = 0

    for batch in question_pool:
        if batch_limit is not None and batch_count == batch_limit:
            break
        prompt_list = []
        for qes in batch:
            prompt = given_prompt + "Q: " + qes['question'] + "\nA:"
            prompt_list.append(prompt)

        if args.dataset == "gsm8k":
            responses = GPT3_request(model=args.model, input_prompt=prompt_list, max_tokens=256, temperature=0, stop='\n')
        elif args.dataset == "aqua":
            responses = GPT3_request(model=args.model, input_prompt=prompt_list, max_tokens=256, temperature=0, stop='\n')
        else:
            print("Dataset process not implemented")
            raise NotImplementedError

        ans_list = answer_extraction(args, responses)

        for ans_idx in range(len(ans_list)):
            if ans_list[ans_idx] == batch[ans_idx]['answer']:
                correct += 1

        batch_count += 1

    return correct


def arg_parser():
    parser = argparse.ArgumentParser(description="CoT")
    parser.add_argument("--random_seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--dataset", type=str, default="gsm8k", choices=["gsm8k","svamp", "aqua"], help="dataset to inference"
    )
    parser.add_argument(
        "--prompt_path", type=str, default="prompts/active", help="type of prompts to use"
    )
    # parser.add_argument(
    #     "--resume_id", type=int, default=0, help="whether to limit test dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for testing."
    # )
    parser.add_argument("--minibatch_size", type=int, default=1, choices=[1, 5, 10], help="batch size (num_prompts) for each request")
    
    parser.add_argument("--max_num_workers", type=int, default=0, help="maximum number of workers for inference")
    
    parser.add_argument(
        "--model", type=str, default="code-davinci-002", choices=["text-davinci-002", "code-davinci-002"], help="model used for decoding."
    )
    
    parser.add_argument(
        "--method", type=str, default="active_cot", choices=["zero_shot", "zero_shot_cot", "few_shot", "few_shot_cot", "auto_cot", "active_cot"], help="method"
    )
    parser.add_argument(
        "--output_dir", type=str, default="experiment/multiarith", help="output directory"
    )
    parser.add_argument(
        "--max_length_cot", type=int, default=256, help="maximum length of output tokens by model for reasoning extraction"
    )
    parser.add_argument(
        "--limit_batch_size", type=int, default=0, help="whether to limit test dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for testing."
    )
    parser.add_argument(
        "--api_time_interval", type=float, default=1.0, help="how many seconds to sleep between each request"
    )
    parser.add_argument(
        "--temperature", type=float, default=0, help=""
    )
    parser.add_argument(
        "--log_dir", type=str, default="./log/", help="log directory"
    )
    
    args = parser.parse_args()
    args.output_dir = Path(args.output_dir)
    
    if args.dataset == "gsm8k":
        # args.dataset_path = "./dataset/grade-school-math/test.jsonl"
        args.dataset_path = r"D:\HKUST_NLP_Research\cot_active_learning\grade_school_math\data\test.jsonl"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
        args.datset_size = 1319
    elif args.dataset == "svamp":
        pass
        # args.dataset_path = "./dataset/SVAMP/SVAMP.json"
        # args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "aqua":
        args.dataset_path = r"D:\HKUST_NLP_Research\cot_active_learning\AQuA\test.json"
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