from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

from peft import LoraConfig

from trl import SFTTrainer
from trl import SFTConfig

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

dataset = load_dataset(
    "json",
    data_files="/home/husammm/Desktop/courses/cs_courses/RL/projects/Hierarchical_RL_Agents_for_Legal_Contract_Analysis/agent_train/assets/CUAD/CUAD_v1.json"
)["train"]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto"
)

def format_example(example):

    return (
        example["prompt"]
        + " "
        + example["answer"]
    )

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ]
)

training_args = SFTConfig(
    output_dir="./sft_model",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    logging_steps=10,
    save_steps=200,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    peft_config=lora_config,
    formatting_func=format_example
)

trainer.train()
trainer.save_model("./sft_model")
