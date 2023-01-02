# active-cot
Chain-of-Thought with Active Learning

## Generate Uncertainty Result
```shell
python generate_uncertainty.py --dataset="gsm8k" --minibatch_size=10 --max_num_workers=10 --model="code-davinci-002" --method="few_shot_cot" --limit_batch_size=0 --prompt_path="./prompts/manual" --random_seed=42 --partition=10 --output_dir=./uncertainty_result/result_k=5_var3.txt --num_trails=5 --sort_by=disagreement
```

## Run inference script
```shell
python inference.py --dataset="gsm8k" --minibatch_size=10 --max_num_workers=5 --model="code-davinci-002" --method="active_cot" --limit_batch_size=0 --prompt_path="./prompts/active"
```