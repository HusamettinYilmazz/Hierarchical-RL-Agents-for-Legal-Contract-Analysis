import os
import json
from pathlib import Path

from utils import Config


def process_data(config: Config, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(config.data["dataset_path"], "r") as f:
        data = json.load(f)

    examples = []

    for doc in data["data"]:
        for paragraph in doc["paragraphs"]:

            contract = paragraph["context"]

            results = {}

            for qa in paragraph["qas"]:

                field = qa["question"].strip()

                if len(qa["answers"]) == 0:
                    results[field] = {
                        "found": False,
                        "value": None
                    }
                else:
                    answers = list({
                        ans["text"].strip()
                        for ans in qa["answers"]
                        if ans["text"].strip()
                    })

                    results[field] = {
                        "found": True,
                        "value": answers
                    }

            prompt = f"""You are an expert legal contract analyst.

Analyze the following contract and extract the requested legal information.

Return ONLY a valid JSON object.

Contract:
{contract}
"""

            examples.append(
                {
                    "prompt": prompt,
                    "answer": json.dumps(results, ensure_ascii=False)
                }
            )

    split = int(0.9 * len(examples))

    train = examples[:split]
    val = examples[split:]

    with open(output_dir / "train.jsonl", "w", encoding="utf-8") as f:
        for ex in train:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(output_dir / "val.jsonl", "w", encoding="utf-8") as f:
        for ex in val:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Train: {len(train)}")
    print(f"Validation: {len(val)}")
