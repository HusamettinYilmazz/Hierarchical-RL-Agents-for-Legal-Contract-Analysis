from datasets import load_dataset

from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

from trl import GRPOTrainer
from trl import GRPOConfig

from reward import compute_reward

MODEL_PATH = "/kaggle/input/models/husamsha/checkpoint-2614/pytorch/default/1/checkpoint-2614"

dataset = load_dataset(
    "json",
    data_files="/kaggle/working/Hierarchical-RL-Agents-for-Legal-Contract-Analysis/agent_train/outputs/train.jsonl"
)["train"]

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto"
)

def reward_func(completions, answer, **kwargs):

    rewards = []

    for completion in completions:

        rewards.append(
            compute_reward(
                completion,
                answer
            )
        )

    return rewards

config = GRPOConfig(
    output_dir="/kaggle/working/Hierarchical-RL-Agents-for-Legal-Contract-Analysis/agent_train/outputs/grpo_model/config",
    learning_rate=5e-6,
    per_device_train_batch_size=1,
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

trainer.train()
trainer.save_model("/kaggle/working/Hierarchical-RL-Agents-for-Legal-Contract-Analysis/agent_train/outputs/grpo_model")
