import os
os.environ["WANDB_MODE"] = "disabled"
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

from dataset import create_dataset
from utils import load_config, Config


def formatting_func(example, tokenizer):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False
    )

    tokens = tokenizer(
        text,
        truncation=True,
        max_length=1024   # or 2048 if GPU  allows
    )

    return tokenizer.decode(tokens["input_ids"])


def train(config: Config, checkpoint: str | None = None):

    dataset = create_dataset(json_path=config.data['dataset_path'])

    tokenizer = AutoTokenizer.from_pretrained(config.model['model_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model['model_name'],
        device_map="auto"
    )

    lora_config = LoraConfig(
        r= config.training['lora_r'],
        lora_alpha= config.training['lora_alpha'],
        lora_dropout= config.training['lora_dropout'],
        bias= config.training['lora_bias'],
        task_type= config.training['lora_task_type'],
        target_modules= config.training['lora_target_modules']
    )

    training_args = SFTConfig(
        output_dir= os.path.join(ROOT, config.data['output_path'], "sft"),
        num_train_epochs= config.training['sft_num_train_epochs'],
        per_device_train_batch_size= config.training['sft_per_device_train_batch_size'],
        gradient_accumulation_steps= config.training['sft_gradient_accumulation_steps'],
        learning_rate= float(config.training['sft_learning_rate']),
        logging_steps= config.training['sft_logging_steps'],
        save_steps= config.training['sft_save_steps'],
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        args=training_args,
        formatting_func=formatting_func
    )

    trainer.train(
        resume_from_checkpoint= checkpoint
    )
    trainer.save_model(os.path.join(ROOT, config.data['output_path']))

if __name__ == "__main__":
    config = load_config(os.path.join(ROOT, "configs/config.yml"))
    train(config)
