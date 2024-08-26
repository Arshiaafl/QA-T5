from transformers import pipeline
from transformers import T5Tokenizer
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
from rouge import Rouge
import warnings
from datasets import load_dataset, Dataset, load_metric
import json

model_name = "t5-small"  # You can use 't5-small', 't5-base', or 't5-large' based on your computational resources
tokenizer = T5Tokenizer.from_pretrained(model_name)

def hub_dataset():
    """
    Loads a dataset from the Hugging Face Hub and split it into training and validation sets.

    Args:
        

    Returns:
        tuple: A tuple containing the dataset
    """
    # Loads the dataset
    dataset = load_dataset("google-research-datasets/disfl_qa", split="validation")
    
    
    return dataset

def json_dataset(json_file_path):
    """
    Load a dataset from a JSON file and split it into training and validation sets.

    Args:
        json_file_path (str): Path to the JSON file containing the dataset.

    Returns:
        tuple: A tuple containing the dataset
    """
    # Loads and process data from JSON
    with open(json_file_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    data = {
        "disfluent question": [],
        "original question": []
    }

    for key, value in json_data.items():
        data["disfluent question"].append(value['disfluent'])
        data["original question"].append(value['original'])

    # Converts data to Hugging Face Dataset format
    dataset = Dataset.from_dict(data)
    return dataset

val_dataset = json_dataset("dev.json")
nltk.download('wordnet')

# Initializes the summarization pipeline
summarizer = pipeline("summarization", model="my_fine_tuned_t5_small_model", min_length=5, max_length=42, num_beams=5, length_penalty=0.4)

# Initializes ROUGE metric
rouge = Rouge()

def METEOR_score(candidate, reference):
    score = nltk.translate.meteor_score.meteor_score([reference], candidate)
    return score

def calculate_bleu(candidate, reference):
    reference_tokens = reference.split()
    candidate_tokens = candidate.split()
    
    # Smoothing function to avoid zero BLEU score for shorter sentences
    smoothie = SmoothingFunction().method4
    
    # Calculates BLEU score
    score = sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothie)
    return score

def calculate_rouge(candidate, reference):
    scores = rouge.get_scores(candidate, reference)[0]
    return scores['rouge-l']['f']

def accuracy_eval(dataset):
    bleu_scores = []
    meteor_scores = []
    rouge_scores = []
    wrong_rewrites = []

    # Loop over the validation data and generate predictions
    for example in dataset:
        disfluent_question = example['disfluent question']
        original_question = example['original question']
        
        # Prepends the prefix "summarize: " as required by your task
        input_text = "summarize: " + disfluent_question
        
        # Generates summary
        pred = summarizer(input_text)[0]['summary_text']
        
        # Calculates BLEU score
        bleu_score = calculate_bleu(pred, original_question)
        bleu_scores.append(bleu_score)
        
        # Calculates METEOR score
        pred_tokens = tokenizer.tokenize(pred)
        ref_tokens = tokenizer.tokenize(original_question)
        meteor_score = METEOR_score(pred_tokens, ref_tokens)
        meteor_scores.append(meteor_score)
        
        # Calculates ROUGE score
        rouge_score = calculate_rouge(pred, original_question)
        rouge_scores.append(rouge_score)
        
        # Collects examples of incorrect rewrites based on BLEU score
        if bleu_score < 0.1:  # Threshold for considering a rewrite as incorrect (adjust as needed)
            wrong_rewrites.append((input_text, pred, original_question))
    
    # Calculates average BLEU, METEOR, and ROUGE scores
    avg_bleu_score = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    avg_meteor_score = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0
    avg_rouge_score = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0
    
    return avg_bleu_score, avg_meteor_score, avg_rouge_score, wrong_rewrites

# Evaluates model
avg_bleu, avg_meteor, avg_rouge, wrong_rewrites = accuracy_eval(val_dataset)

# Print evaluation results
print(f"Average BLEU Score: {avg_bleu:.4f}")
print(f"Average METEOR Score: {avg_meteor:.4f}")
print(f"Average ROUGE Score: {avg_rouge:.4f}")
print(f"Examples of incorrect rewrites (based on BLEU): {wrong_rewrites[:5]}")  # Display first 5 incorrect examples
