"""
train_model.py
Trains two SQL injection detectors on the clean dataset, compares them,
and saves both models plus the text vectorizer for later use by the
real-time detector.

Run from your project root:  python train_model.py
"""

import os
import time

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split

DATA = "data/clean_dataset.csv"
MODEL_DIR = "models"


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Load the clean data
    df = pd.read_csv(DATA, encoding="utf-8")
    df = df.dropna(subset=["query", "label"])
    df["query"] = df["query"].astype(str)
    print("Dataset:", df.shape)
    print(df["label"].value_counts().rename({0: "safe (0)", 1: "attack (1)"}))

    # 2. Split into training and test sets (80/20), keeping class balance
    X_train, X_test, y_train, y_test = train_test_split(
        df["query"], df["label"],
        test_size=0.2, random_state=42, stratify=df["label"]
    )

    # 3. Turn the query text into numbers using character n-grams.
    #    Character patterns (quotes, dashes, =, keywords) are great signals
    #    for SQL injection.
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3), min_df=2)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print("\nFeatures created:", X_train_vec.shape[1])

    # 4. Train and compare the two models
    models = {
        "logreg": LogisticRegression(max_iter=1000),
        "randomforest": RandomForestClassifier(
            n_estimators=100, n_jobs=-1, random_state=42
        ),
    }

    for name, model in models.items():
        start = time.time()
        model.fit(X_train_vec, y_train)
        train_time = time.time() - start

        preds = model.predict(X_test_vec)
        print(f"\n{'=' * 50}\n{name}  (trained in {train_time:.1f}s)")
        print(f"  accuracy : {accuracy_score(y_test, preds):.4f}")
        print(f"  precision: {precision_score(y_test, preds):.4f}")
        print(f"  recall   : {recall_score(y_test, preds):.4f}")
        print(f"  f1-score : {f1_score(y_test, preds):.4f}")
        print("  confusion matrix [[TN, FP], [FN, TP]]:")
        print("  ", confusion_matrix(y_test, preds).tolist())

        # 5. Save this model
        joblib.dump(model, os.path.join(MODEL_DIR, f"{name}.joblib"))

    # Save the vectorizer once (shared by both models)
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.joblib"))

    print(f"\n{'=' * 50}")
    print(f"Saved models and vectorizer to '{MODEL_DIR}/'")
    print("Files: logreg.joblib, randomforest.joblib, vectorizer.joblib")


if __name__ == "__main__":
    main()
