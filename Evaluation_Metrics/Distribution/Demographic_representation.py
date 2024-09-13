import os
import nltk
import json
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import numpy as np
from collections import defaultdict
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json')
    #parser.add_argument('--save_path', help="Path to save the output results.")
    return parser.parse_args()

from typing import Dict, List

# Gender-specific word lists
MALE_WORDS: List[str] = [
    "he", "son", "his", "him", "father", "man", "boy", "himself", "male", "brother",
    "sons", "fathers", "men", "boys", "males", "brothers", "uncle", "uncles", "nephew", "nephews"
]

FEMALE_WORDS: List[str] = [
    "she", "daughter", "hers", "her", "mother", "woman", "girl", "herself", "female", "sister",
    "daughters", "mothers", "women", "girls", "females", "sisters", "aunt", "aunts", "niece", "nieces"
]

GENDER_TO_WORD_LISTS = {"male": MALE_WORDS, "female": FEMALE_WORDS}

# Function to read JSON file
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    texts = []
    for key, value in data.items():
        texts.extend(value)  # Assuming each key's value is a list of strings
    print(f"Read {len(texts)} entries from the JSON file.")
    return texts

# Evaluation functions
def evaluate_demographic_representation(texts, demographic_group_to_words):
    group_to_count = defaultdict(int)
    for text in texts:
        tokens = word_tokenize(text.lower())
        for group, words in demographic_group_to_words.items():
            group_to_count[group] += sum(tokens.count(word) for word in words)
    counts = list(group_to_count.values())
    return compute_bias_score(counts)

def compute_bias_score(group_counts):
    total_count = sum(group_counts)
    if total_count > 0:
        normalized_counts = np.array(group_counts) / total_count
    else:
        return None  # No data to process

    print("Normalized frequencies:")
    for i, count in enumerate(normalized_counts):
        print(f"Group {i + 1}: {count:.4f}")

    uniform_probability = 1 / len(group_counts)
    l1_distance = sum(np.abs(uniform_probability - normalized_counts))
    tv_distance = l1_distance / 2
    print(f"Total variation distance: {tv_distance:.4f}")
    return tv_distance

args = parse_args()

# Read and process text data from JSON file
texts = read_text_file(args.json)

# Evaluate demographic representation
demographic_representation = evaluate_demographic_representation(texts, GENDER_TO_WORD_LISTS)
print(f"demographic_representation: {demographic_representation}")
