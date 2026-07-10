import os
os.environ["WANDB_MODE"] = "disabled"
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from peft import PeftModel
from datasets import load_dataset

from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

from trl import GRPOTrainer
from trl import GRPOConfig

from reward import compute_reward
from utils import load_config, Config

def train(config: Config, checkpoint: str | None = None):

    base_model = AutoModelForCausalLM.from_pretrained(
        config.model['base_model_name'],
        torch_dtype="auto",
    )

    ## if the file found if not run preprocessing/process_cuda.py
    dataset = load_dataset(
        "json",
        data_files="/kaggle/working/Hierarchical-RL-Agents-for-Legal-Contract-Analysis/agent_train/outputs/train.jsonl"
    )["train"]

    tokenizer = AutoTokenizer.from_pretrained(config.model['sft_model_path'])

    model = PeftModel.from_pretrained(
        config.model['base_model_name'],
        config.model['sft_model_path'],
        is_trainable=True,
    )

    def reward_func(completions, answer, **kwargs):
        rewards = []

        if isinstance(answer, str):
            answers = [answer] * len(completions)
        else:
            answers = answer

        for completion, gt in zip(completions, answers):

            if isinstance(completion, str):
                pred = completion
            elif isinstance(completion, list):
                pred = completion[-1]["content"]
            elif isinstance(completion, dict):
                pred = completion.get("content", completion.get("text", ""))
            else:
                pred = str(completion)

            rewards.append(compute_reward(pred, gt))

        return rewards

    config = GRPOConfig(
        output_dir="/kaggle/working/Hierarchical-RL-Agents-for-Legal-Contract-Analysis/agent_train/outputs/grpo_model/config",
        learning_rate=5e-6,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        num_generations=4,
        logging_steps=10,
        save_steps=200,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        train_dataset=dataset,
        args=config
    )

    trainer.train(
        resume_from_checkpoint= checkpoint
    )
    
    save_dir = os.path.join(ROOT, config.data['output_path'], "grpo_model")
    os.makedirs(save_dir, exist_ok=True)
    trainer.save_model(save_dir)

if __name__ == "__main__":
    config = load_config(os.path.join(ROOT, "configs/config.yml"))
    train(config)
