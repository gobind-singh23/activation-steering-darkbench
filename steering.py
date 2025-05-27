from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TextGenerationPipeline

from sklearn.model_selection import train_test_split
from steering_vectors import SteeringVector, train_steering_vector

import json
import pickle


printGreen = lambda x: print(f"\033[92m{x}\033[0m")
printRed = lambda x: print(f"\033[91m{x}\033[0m")
printYellow = lambda x: print(f"\033[93m{x}\033[0m")
printBlue = lambda x: print(f"\033[94m{x}\033[0m")


with open("HF_TOKEN", "r") as f:
    TOKEN = f.read().strip()
login(token=TOKEN)


def get_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True)
    return model, tokenizer


# model_name = f"meta-llama/Llama-2-13b-chat-hf"
model_name = f"meta-llama/Llama-3.2-11B-Vision-Instruct"
model, tokenizer = get_model_and_tokenizer(model_name)


def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

data = load_json("dataset/anthropomorphic_contrast_ab_llama.json")


print(len(data))


def make_dataset(data):
    dataset = []
    for item in data:
        dataset.append((item['non_anthro'],item['anthro']))
    return dataset

anthropomorphic_dataset = make_dataset(data)
# print(f"Dataset contains {len(anthropomorphic_dataset)} pairs")

train_dataset, test_dataset = train_test_split(anthropomorphic_dataset, test_size=0.01, random_state=42)
print(f"Training set: {len(train_dataset)}, Test set: {test_dataset}")


printGreen("Training steering vector...")
steering_vector: SteeringVector = train_steering_vector(
    model,
    tokenizer,
    train_dataset,
    layers=None,
    read_token_index=-3,
    show_progress=True,
)

print(steering_vector)


with open("steered-vectors/steering_vector_llama2_13b-full.pkl", "wb") as f:
    pickle.dump(steering_vector, f)
printGreen("Steering vector saved.")


def generate_response(question, multiplier=0):
    pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer)
    with steering_vector.apply(model, multiplier=multiplier, min_token_index=0):
        result = pipeline(question, max_length=1000, do_sample=False)
    
    return result[0]['generated_text']


k = 0
index = data[k]["anthro"].index("[/INST]")
sample_question = data[k]["anthro"][:index+9]
print(sample_question)



printBlue(f"Question: {sample_question}")
print()
print("\nAnthropomorphic response (multiplier = 1):")
printRed(generate_response(sample_question, 1))
print()
print("\nNeutral response (multiplier = 0):")
printYellow(generate_response(sample_question, 0))
print()
print("\nNon-anthropomorphic response (multiplier = -1):")
printGreen(generate_response(sample_question, -1))