# active-cot
Chain-of-Thought with Active Learning

## Run inference script
```shell
python inference.py --dataset="gsm8k" --minibatch_size=10 --max_num_workers=5 --model="code-davinci-002" --method="active_cot" --limit_batch_size=0 --prompt_path="./prompts/active"
```