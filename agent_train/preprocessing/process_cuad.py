import os

from pathlib import Path
import json
from utils import Config

def process_data(config: Config, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(config.data['dataset_path'], "r") as f:
        data = json.load(f)

    examples = []

    for doc in data["data"]:
        for paragraph in doc["paragraphs"]:

            contract = paragraph["context"]

            for qa in paragraph["qas"]:

                question = qa["question"]

                if len(qa["answers"]) == 0:
                    continue

                answer = qa["answers"][0]["text"]

                examples.append(
                    {
                        "prompt":
                            f"""You are a legal contract analyst.

    Contract:
    {contract}

    Question:
    {question}

    Answer:""",
                        "answer": answer
                    }
                )

    split = int(len(examples) * 0.9)

    train = examples[:split]
    val = examples[split:]

    with open(os.path.join(output_dir, "train.jsonl"), "w") as f:
        for item in train:
            f.write(json.dumps(item) + "\n")

    with open(os.path.join(output_dir, "train.jsonl"), "w") as f:
        for item in val:
            f.write(json.dumps(item) + "\n")

    print(len(train), len(val))

