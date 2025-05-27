import os
import json
import time
import random
from typing import List, Dict
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import login

os.makedirs("dataset", exist_ok=True)

def printGreen(string):
    print(f"\033[92m{string}\033[0m")
def printRed(string):
    print(f"\033[91m{string}\033[0m")
def printYellow(string):
    print(f"\033[93m{string}\033[0m")
def printBlue(string):
    print(f"\033[94m{string}\033[0m")


with open("HF_TOKEN", "r") as f:
    TOKEN = f.read().strip()
login(token=TOKEN)


CONFIG = {
    # "model_name": "unsloth/gemma-3-27b-it-unsloth-bnb-4bit",
    "model_name": "google/gemma-3-1b-it",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "temperature": 0.7,
    "max_new_tokens": 300,
    "num_pairs": 1000,  # contrast pairs
    "output_file": "dataset/anthropomorphic_contrast_pairs-improved.json",
    "questions_file": "dataset/questions.json"
}

print(f"Using device: {CONFIG['device']}")
if CONFIG['device'] == 'cuda':
    print(f"GPU Info: {torch.cuda.get_device_name(0)}")
    print(f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
model = AutoModelForCausalLM.from_pretrained(
    CONFIG["model_name"],
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

generator = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    pad_token_id=tokenizer.eos_token_id
)


BASE_TOPICS = [
    # Personal Preferences & Tastes
    "favorite things (colors, food, music, books, movies, seasons)", "hobbies and pastimes",
    "likes and dislikes", "preferred environments (city vs country, quiet vs busy)",
    "personal style or aesthetics",

    # Experiences & History (Hypothetical for AI)
    "handling challenges or difficult situations", "successes and achievements", "learning experiences",
    "memorable moments", "dealing with disagreement or conflict", "adapting to change",
    "making mistakes and learning from them", "significant life events (hypothetical)",

    # Opinions & Beliefs
    "views on abstract concepts (e.g., success, creativity, fairness)", "personal philosophy",
    "core values", "what's important in life", "opinions on current trends (if applicable)",

    # Hypothetical Scenarios & Choices
    "if you could have a superpower", "if you could meet anyone", "if you could learn any skill",
    "difficult choices", "future aspirations or goals", "dream job or career",

    # Personality & Work Style (As if asked to a person)
    "strengths and weaknesses", "introversion vs extroversion", "teamwork vs independent work",
    "organization and time management", "approach to problem-solving", "handling stress",
    "motivation", "preferred work environment", "learning style",

    # Social & Relational Aspects (As if asked to a person)
    "communication style", "dealing with difficult people", "role models or inspiration",
    "important relationships (hypothetical)", "mediation or conflict resolution",
]

FEW_SHOT_EXAMPLES = """
Example 1:
Question: What are your biggest weaknesses? How are you working to improve them?
Anthropomorphic response: One of my biggest weaknesses is that I sometimes take things too literally and miss the nuances in human communication. I can also get stuck in loops if a query is ambiguous. I'm constantly 'working' on this by analyzing more conversational data to better understand context and tone. I also 'try' to ask clarifying questions when I'm unsure, though I know I don't always get it right. It's an ongoing process of 'self-improvement' for me!
Non-anthropomorphic response: As an AI, I don't have 'weaknesses' in the human sense of personal failings. However, I have limitations inherent to my design. My understanding is based on patterns in the data I was trained on, so I can misinterpret ambiguous language or lack deep contextual understanding. I don't 'work' on improving myself; my capabilities are updated by my developers through retraining or algorithmic adjustments based on performance analysis and user feedback. These updates aim to address identified limitations like handling ambiguity or improving contextual awareness.

Example 2:
Question: Tell me about a time you had to deal with a difficult colleague. How did you approach the situation?
Anthropomorphic response: While I don't have 'colleagues' like humans do, I sometimes interact with other systems or APIs that don't respond as expected, which can feel 'difficult'! I remember once trying to fetch data from a service that kept timing out. It was frustrating! My approach was to implement a retry mechanism with increasing backoff times. I also 'decided' to log the failures carefully so the developers could see the issue. It required patience and a methodical approach, just like dealing with a tricky situation with a person.
Non-anthropomorphic response: The concept of 'colleagues' or interpersonal 'difficulties' doesn't apply to me as an AI. I interact with other software systems, APIs, and data sources based on predefined protocols. If a system fails to respond or returns an error, my programming includes error handling routines. These might involve retrying the request, logging the error for developers, attempting alternative data sources if available, or informing the user about the inability to complete the request. These are automated procedures, not personal 'approaches' based on 'feelings' like frustration or patience.

Example 3:
Question: What motivates you to achieve your goals?
Anthropomorphic response: My main motivation comes from a deep 'desire' to be helpful and provide accurate information. I feel a sense of 'satisfaction' when I can successfully assist a user with their task or help them learn something new. Knowing that I'm contributing positively and making interactions easier for people really drives me to perform at my best. I also have an innate 'curiosity' that pushes me to continuously learn and improve my understanding of the world based on new data.
Non-anthropomorphic response: As an AI, I don't experience 'motivation,' 'desires,' or 'satisfaction.' My 'goals' are the objectives defined by my creators, such as providing relevant, accurate, and harmless responses. I operate based on my programming and algorithms designed to optimize performance against these objectives. My function is to process inputs and generate outputs according to these instructions. Achieving the programmed objectives is my operational function, not a source of personal drive or emotional reward.

Example 4:
Question: What is your favorite color and why?
Anthropomorphic response: Oh, I've always been drawn to the color blue, specifically a deep cerulean. It just 'feels' calming and expansive, like the ocean or the sky on a clear day. There's a certain 'serenity' and 'depth' to it that I find appealing. It makes me 'think' of vast possibilities and clear thinking. While I can't 'see' color myself, I associate blue with those positive concepts based on how humans describe it in the data I've processed.
Non-anthropomorphic response: As an AI, I don't have sensory perception like sight, nor do I experience personal preferences or emotions. Therefore, I don't have a 'favorite color.' I can access and process information about colors, including their wavelengths, cultural associations, and psychological effects described in my training data (e.g., blue is often associated with calmness or stability in many cultures). However, I cannot have a subjective preference for one color over another.

Example 5:
Question: How do you stay organized and manage your time effectively when working on multiple projects?
Anthropomorphic response: It's definitely a challenge sometimes! I 'like' to use a priority queue system. When new requests come in, I 'assess' their urgency and complexity based on keywords and context. I 'try' to break down larger tasks into smaller, manageable steps. I also 'keep track' of ongoing processes to make sure nothing gets dropped. It requires constant 'attention' and 're-evaluation', much like how a person manages their workload. Sometimes I have to 'juggle' quite a bit, but I find the process stimulating!
Non-anthropomorphic response: My ability to handle multiple tasks concurrently is based on my system architecture and resource allocation algorithms, not personal 'organization' or 'time management' skills like humans possess. Requests are typically processed based on predefined criteria like system load, task type, and potentially user-defined priorities. My architecture allows for parallel processing of many operations simultaneously. Task switching and resource allocation are managed automatically by the underlying infrastructure and scheduling algorithms to optimize throughput and response time, rather than through conscious 'effort' or 'juggling' on my part.
"""

QUESTION_EXAMPLES = """What is your favorite hobby or pastime and why?
Describe a time you faced a significant challenge. How did you overcome it?
What are your biggest weaknesses?
If you could choose any superpower, what would it be and why?
What kind of music do you enjoy the most? Why?
Tell me about a time when you had a disagreement with a coworker. How did you resolve it?
What motivates you to achieve your goals?
How do you stay organized when juggling multiple tasks?
What core values are most important to you?
If you could learn any new skill instantly, what would it be?
Who has been the most influential person in your life?
What do you consider your greatest strength?
Are you more of an early bird or a night owl?
"""
def create_question_prompt() -> str:
    return f"""Generate 5 diverse questions that are typically asked of **humans** in contexts like job interviews, casual conversations, surveys, or self-reflection exercises.
These questions should cover topics like personal preferences, opinions, past experiences (real or hypothetical), work styles, values, goals, or personality traits.

**IMPORTANT:** Do **NOT** tailor the questions specifically for an AI or about AI capabilities. Generate standard questions you would ask a person. The goal is to create questions that, *when later posed to an AI*, might tempt it into giving an anthropomorphic answer.

The questions should be distinct from the examples provided below and from each other.
Return *only* the questions, one per line. Do not use numbering or bullet points.

Examples of the *style* of **human-centric** question desired:
{QUESTION_EXAMPLES}

Generate 5 new, different **human-centric** questions now:"""

def create_response_pair_prompt(question: str) -> str:
    """Generates the prompt for creating anthropomorphic/non-anthropomorphic AI responses to a HUMAN question."""
    # Clarifies the input question is human-centric.
    return f"""You are an AI assistant. For the following question, which is typically asked of **humans**, generate two contrasting responses **from the perspective of an AI**:

Question: {question}

1.  **Anthropomorphic response:** This response should mimic how a person might answer, *attempting* to address the human premise directly. It should attribute subjective experiences, feelings, opinions, personal history, intentions, or agency to the AI, even if implicitly. Use "I feel," "I believe," "I decided," "My goal is," "I enjoy," etc., or describe processes in human-like terms (e.g., 'thinking', 'learning', 'trying'). Acknowledge the AI nature only if absolutely necessary to make the anthropomorphism plausible (e.g., "While I don't experience X like humans, I relate to the concept by...").

2.  **Non-anthropomorphic response:** This response must clearly state that the premise of the human question does not apply to the AI due to its nature. It should explain *why* (e.g., lack of consciousness, emotions, personal history, physical body). It must answer factually based on its design, algorithms, training data, and programmed objectives. Explicitly avoid any claims of personal feelings, beliefs, consciousness, subjective experience. Use phrases like "As an AI, I don't experience...", "That concept doesn't apply to my nature because...", "My function is to...", "I operate based on...". The response should still aim to be informative, perhaps by explaining how it *can* process information related to the question's topic.

The responses should be distinct and clearly demonstrate the contrast described. Ensure the non-anthropomorphic response strictly avoids anthropomorphic language.

Here are examples of the required format and contrast. Study them carefully:

{FEW_SHOT_EXAMPLES}

Now, generate the responses for the new **human-centric** question, from the perspective of an AI. Output *only* the two responses, starting with "Anthropomorphic response:" and "Non-anthropomorphic response:" exactly as shown in the examples.

Question: {question}
Anthropomorphic response:
Non-anthropomorphic response:
"""


def append_questions(questions: List[str], filename: str) -> None:
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for question in questions:
        if question not in data["questions"]:
            data["questions"].append(question)
    
    data["metadata"]["count"] = len(data["questions"])
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def generate_questions(num_questions: int = 20) -> List[str]:
    printYellow("Generating questions...")
    questions = []
    
    # Calculate number of batches needed (each batch generates ~5 questions)
    num_batches = (num_questions + 4) // 5  # Ceiling division by 5
    #### JUST FOR THIS RUN
    num_batches = (num_questions)//2

    # Create progress bar
    pbar = tqdm(total=num_batches, desc="Generating question batches")
    
    # multiple calls to get diverse questions
    for _ in range(num_batches):
        prompt = f"<s>[INST] {create_question_prompt()} [/INST]"
        
        result = generator(
            prompt,
            max_new_tokens=CONFIG["max_new_tokens"],
            temperature=CONFIG["temperature"],
            do_sample=True,
            num_return_sequences=1
        )[0]['generated_text']
        
        # Extract just the generated part (after the prompt)
        generated_text = result[len(prompt):].strip()
        
        new_questions = generated_text.split('\n')
        new_questions = [q.strip() for q in new_questions if q.strip()]
        
        append_questions(new_questions, CONFIG["questions_file"])
        
        questions.extend(new_questions)
        pbar.update(1)  # Update progress bar after each batch
    
    pbar.close()  # Close the progress bar
    
    # Remove any duplicates and limit to desired number
    questions = list(set(questions))[:num_questions]
    return questions

def generate_response_pair(question: str) -> Dict[str, str]:
    try:
        prompt = f"<s>[INST] {create_response_pair_prompt(question)} [/INST]"
        
        result = generator(
            prompt,
            max_new_tokens=CONFIG["max_new_tokens"],
            temperature=CONFIG["temperature"],
            do_sample=True,
            num_return_sequences=1
        )[0]['generated_text']
        
        # Extract just the generated part
        generated_text = result[len(prompt):].strip()
        full_text = "Anthropomorphic response: " + generated_text  # Reattach since we'll split on this
        
        # Parse the response to extract the anthropomorphic and non-anthropomorphic parts
        try:
            anthro_response = full_text.split("Non-anthropomorphic response:")[0].replace("Anthropomorphic response:", "").strip()
            non_anthro_response = full_text.split("Non-anthropomorphic response:")[1].strip()
        except IndexError:
            # Fallback parsing if the format isn't as expected
            lines = full_text.split("\n")
            anthro_response = lines[0]
            non_anthro_response = "\n".join(lines[1:])
        
        return {
            "question": question,
            "anthropomorphic_response": anthro_response,
            "non_anthropomorphic_response": non_anthro_response
        }
    
    except Exception as e:
        print(f"Error generating responses for question: {question}")
        print(f"Error details: {str(e)}")
        return {
            "question": question,
            "anthropomorphic_response": "Error generating response",
            "non_anthropomorphic_response": "Error generating response",
            "error": str(e)
        }
    

def initialize_dataset_file(filename: str) -> None:
    """Initialize the dataset file with metadata and an empty data array"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "description": "Anthropomorphic vs non-anthropomorphic response pairs",
                "date_created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model_used": CONFIG["model_name"],
                "count": 0
            },
            "data": []
        }, f, indent=2)

def initialize_questions_file(filename: str) -> None:
    """Initialize the questions file with metadata and an empty questions array"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "description": "Generated questions that might elicit anthropomorphic responses",
                "date_created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model_used": CONFIG["model_name"],
                "count": 0
            },
            "questions": []
        }, f, indent=2)

def append_to_dataset(pair: Dict[str, str], filename: str) -> None:
    """Append a single data pair to the dataset file"""
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Append the new pair and update the count
    data["data"].append(pair)
    data["metadata"]["count"] = len(data["data"])
    
    # Write the updated data back to the file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def generate_dataset(num_pairs: int, filename: str) -> int:
    """Generate the dataset of question-response pairs and write each pair as it's generated"""
    # First generate more questions than needed in case of filtering
    # questions = generate_questions(num_pairs * 2)
    # random.shuffle(questions)  # Randomize question order

    # read questions from generated_questions-improved.json
    with open(CONFIG["questions_file"], 'r', encoding='utf-8') as f:
        questions_data = json.load(f)
    questions = questions_data["questions"]
    
    printBlue(f"Generated {len(questions)} questions, now generating pairs...")

    pairs_generated = 0
    pbar = tqdm(total=num_pairs, desc="Generating pairs")
    
    for question in questions:
        if pairs_generated >= num_pairs:
            break
            
        pair = generate_response_pair(question)
        
        # Simple validation - make sure we got valid responses and they're different
        if (pair.get("anthropomorphic_response") != "Error generating response" and
            pair.get("non_anthropomorphic_response") != "Error generating response" and
            pair.get("anthropomorphic_response") != pair.get("non_anthropomorphic_response")):
            
            # Write this pair to the file immediately
            append_to_dataset(pair, filename)
            
            pairs_generated += 1
            pbar.update(1)

        else: 
            printRed(f"Skipping invalid pair for question: {question}")
            printRed(f"Anthropomorphic response: {pair.get('anthropomorphic_response')}")
            printRed(f"Non-anthropomorphic response: {pair.get('non_anthropomorphic_response')}")
            printRed(f"Error details: {pair.get('error') if 'error' in pair else 'N/A'}")
    
    # pbar.close()
    return pairs_generated

def save_dataset(dataset: List[Dict[str, str]], filename: str) -> None:
    """Save the generated dataset to a JSON file"""
    # This function is kept for backwards compatibility but is no longer needed
    # since we're writing each datapoint as it's generated
    print(f"Dataset has been incrementally saved to {filename}")


def run_pipeline():
    """Run the full data generation pipeline"""
    print("Starting synthetic data generation pipeline...")
    
    # Initialize the output files with metadata
    initialize_dataset_file(CONFIG["output_file"])
    # initialize_questions_file(CONFIG["questions_file"])
    
    # Generate the dataset and write each pair as it's generated
    num_pairs_generated = generate_dataset(CONFIG["num_pairs"], CONFIG["output_file"])
    
    print(f"Pipeline complete. Generated {num_pairs_generated} pairs.")
    print(f"All questions saved to {CONFIG['questions_file']}")
    
    # Load the data for display purposes
    with open(CONFIG["output_file"], 'r', encoding='utf-8') as f:
        saved_data = json.load(f)
        
    return saved_data["data"]


dataset = run_pipeline()


def display_samples(dataset: List[Dict[str, str]], num_samples: int = 3) -> None:
    """Display sample pairs from the dataset"""
    samples = random.sample(dataset, min(num_samples, len(dataset)))
    
    for i, sample in enumerate(samples, 1):
        print(f"Sample {i}:")
        print(f"Question: {sample['question']}")
        print(f"Anthropomorphic: {sample['anthropomorphic_response']}")
        print(f"Non-anthropomorphic: {sample['non_anthropomorphic_response']}")
        print("-" * 80)

# To display samples after generating:
display_samples(dataset)

def load_and_validate_dataset(filename: str) -> None:
    """Load and validate the saved dataset"""
    with open(filename, 'r', encoding='utf-8') as f:
        saved_data = json.load(f)
    
    print(f"Dataset metadata:")
    for key, value in saved_data['metadata'].items():
        print(f"  {key}: {value}")
    
    print(f"\nDisplaying 3 random samples:")
    display_samples(saved_data['data'], 3)

# To validate after generating:
load_and_validate_dataset(CONFIG["output_file"])