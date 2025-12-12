import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def train_baseline_model():
    data_path = "processed_data.csv"
    if not os.path.exists(data_path):
        print("Processed data not found. Please run src/preprocess.py first.")
        return

    print("Loading processed data...")
    df = pd.read_csv(data_path)
    
    # Handle NaN values in text
    df = df.dropna(subset=['clean_text'])
    
    X = df['clean_text']
    y = df['label']
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # --- 2. PIPELINE CREATION ---
    # A Pipeline is like a factory assembly line.
    # Step 1: TfidfVectorizer -> Turns text into numbers (Frequency counts).
    # Step 2: LinearSVC -> The actual mathematical model that draws a line between Real and Fake.
    print("Training Baseline Model (TF-IDF + LinearSVC)...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1,2))),
        ('clf', LinearSVC(verbose=1))
    ])
    
    # --- 3. TRAINING ---
    # This is where the model "learns". It looks at X_train (text) and y_train (answers)
    # and figures out the patterns.
    pipeline.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = pipeline.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))
    
    # Save model
    model_path = "baseline_model.pkl"
    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_baseline_model()
