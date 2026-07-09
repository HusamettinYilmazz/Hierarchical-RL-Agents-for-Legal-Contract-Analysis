
from datasets import Dataset
import json

def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        cuad = json.load(f)

    return cuad

def load_data_from_json(cuad_json):
    examples = []

    for contract in cuad_json["data"]:
        for paragraph in contract["paragraphs"]:
            context = paragraph["context"]

            for qa in paragraph["qas"]:
                question = qa["question"]

                answer = qa["answers"][0]["text"] if qa["answers"] else "No relevant clause found."

                messages = [
                    {
                        "role": "user",
                        "content": (
                            "You are a legal contract analyst.\n\n"
                            f"Contract:\n{context}\n\n"
                            f"Question:\n{question}\n\n"
                            "Return the exact clause text if present. Otherwise say: No relevant clause found."
                        )
                    },
                    {
                        "role": "assistant",
                        "content": answer
                    }
                ]


                examples.append({"messages": messages})

    return examples

def create_dataset(json_path):
    cuad_json = load_json(json_path=json_path)
    examples = load_data_from_json(cuad_json=cuad_json)

    dataset = Dataset.from_list(examples)

    return dataset
