# active-cot
Chain-of-Thought with Active Learning

## Generate Uncertainty Result
```shell
python generate_uncertainty.py --dataset="gsm8k" --minibatch_size=10 --max_num_workers=5 --model="code-davinci-002" --method="few_shot_cot" --limit_batch_size=0 --prompt_path="./prompts/manual" --random_seed=42 --partition=10 --output_dir=./uncertainty_result/result_k=5_var3.txt --num_trails=10 --sort_by=disagreement --setting=fair --api_pool_idx=0
```

last letters dataset need to indicate how long the concat letter length in the command args

## Run inference script
```shell
python inference.py --dataset="last_letters" --minibatch_size=10 --max_num_workers=5 --model="code-davinci-002" --method="active_cot" --limit_batch_size=0 --prompt_path="./final_test_prompt/code_style_last_letters" --random_seed=42 --multipath=40 --temperature=0.7 --api_pool_idx=2 --concat_length=4
```