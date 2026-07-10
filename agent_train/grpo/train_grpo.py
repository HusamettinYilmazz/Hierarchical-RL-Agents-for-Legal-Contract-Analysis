from datasets import load_dataset

from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

from trl import GRPOTrainer
from trl import GRPOConfig

from reward import compute_reward

MODEL_PATH = "./sft_model"

dataset = load_dataset(
    "json",
    data_files="data/processed/train.jsonl"
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
    output_dir="./grpo_model",
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
trainer.save_model("./grpo_model")
