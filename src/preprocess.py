import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4') # Added omw-1.4 for WordNet

def preprocess_text(text):
    """
    Cleans and preprocesses text data.
    """
    # --- 1. CLEANING FUNCTION ---
    # This function acts like a "Correction Pen" for the text.
    # It removes things that confuse the AI, like:
    # - Website links (URLs)
    # - HTML tags (<br>, <div>)
    # - Weird symbols (@, #, $)
    
    # 1. Lowercase (Make everything "abc" instead of "ABC" so "The" and "the" look the same)
    text = str(text).lower()
    
    # 2. Remove URLs (http://...)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # 3. Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # 4. Remove special characters (Keep only english letters)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # 5. Tokenization (Split sentence into ["word1", "word2"])
    tokens = text.split() 
    
    # 6. Remove Stopwords and Lemmatize
    # Stopwords = boring words like "the", "is", "at". We delete them.
    # Lemmatize = Turning "Running" -> "Run".
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    # Glue the words back together into a single string
    return " ".join(tokens)

def load_and_process_data():
    print("Loading datasets...")
    try:
        df_fake = pd.read_csv("Fake.csv")
        df_true = pd.read_csv("True.csv")
    except FileNotFoundError:
        print("Error: data files not found. Please run download_data.py first.")
        return

    # Add labels
    df_fake['label'] = 1 # Fake
    df_true['label'] = 0 # True

    # Combine datasets
    print("Combining datasets...")
    df = pd.concat([df_fake, df_true], axis=0).reset_index(drop=True)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Total records: {len(df)}")
    
    # Preprocess text
    print("Preprocessing text (this might take a while)...")
    # Using 'text' column if available, else 'title'
    if 'text' in df.columns:
        df['clean_text'] = df['text'].apply(preprocess_text)
    else:
        print("Warning: 'text' column not found, using 'title'")
        df['clean_text'] = df['title'].apply(preprocess_text)

    # Save processed data
    output_file = "processed_data.csv"
    df.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")

if __name__ == "__main__":
    load_and_process_data()
