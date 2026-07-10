import json
from pathlib import Path

INPUT_FILE = "/home/husammm/Desktop/courses/cs_courses/RL/projects/Hierarchical_RL_Agents_for_Legal_Contract_Analysis/agent_train/assets/CUAD/CUAD_v1.json"
OUTPUT_DIR = Path("/home/husammm/Desktop/courses/cs_courses/RL/projects/Hierarchical_RL_Agents_for_Legal_Contract_Analysis/agent_train/outputs")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(INPUT_FILE, "r") as f:
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

with open(OUTPUT_DIR / "train.jsonl", "w") as f:
    for item in train:
        f.write(json.dumps(item) + "\n")

with open(OUTPUT_DIR / "val.jsonl", "w") as f:
    for item in val:
        f.write(json.dumps(item) + "\n")

print(len(train), len(val))
