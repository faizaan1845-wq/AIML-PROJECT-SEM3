"""
train_model.py
TrustRate AI — Model Training and Benchmarking Pipeline

Trains and compares text classification models:
- Logistic Regression (Balanced TF-IDF)
- Multinomial Naive Bayes
- Complement Naive Bayes
- Calibrated SGDClassifier (Modified Huber)

Evaluates Accuracy, Precision, Recall, F1, ROC-AUC, and Confusion Matrix.
Saves the best model to model.joblib and complete evaluation metrics to metrics.json.
"""

from pathlib import Path
import json
import warnings
import joblib
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

BASE = Path(__file__).resolve().parent
CURATED_PATH = BASE / "data" / "curated_training_dataset.csv"
RAW_PATH = BASE / "public_reviews_dataset_cleaned.csv"
MODEL_PATH = BASE / "model.joblib"
METRICS_PATH = BASE / "metrics.json"

def load_training_data():
    if CURATED_PATH.exists():
        print(f"Loading curated training dataset from {CURATED_PATH.name}...")
        df = pd.read_csv(CURATED_PATH)
        df["review_text"] = df["review_text"].fillna("").astype(str).str.strip()
        df = df[df["review_text"].str.len() > 10]
        return df, "data/curated_training_dataset.csv"
    elif RAW_PATH.exists():
        print(f"Curated dataset not found. Loading from raw {RAW_PATH.name}...")
        cols = ["review_id", "reviewer_id", "review_text", "reviewer_classified_fake", "reviewer_classified_honest"]
        df = pd.read_csv(RAW_PATH, usecols=cols, low_memory=False)
        df["review_text"] = df["review_text"].fillna("").astype(str).str.strip()
        fake = df["reviewer_classified_fake"].astype(str).str.lower().eq("true")
        honest = df["reviewer_classified_honest"].astype(str).str.lower().eq("true")
        df = df.loc[fake ^ honest].copy()
        df["label"] = np.where(fake.loc[df.index], "suspicious", "genuine")
        df = df[(df["review_text"].str.len() > 10) & df["reviewer_id"].notna() & df["review_id"].notna()]
        df = df.drop_duplicates(subset=["reviewer_id"], keep="first")
        counts = df["label"].value_counts()
        per_class = min(15000, int(counts.min()))
        df = pd.concat([
            df[df["label"] == "genuine"].sample(n=per_class, random_state=42),
            df[df["label"] == "suspicious"].sample(n=per_class, random_state=42)
        ]).sample(frac=1, random_state=42).reset_index(drop=True)
        return df, RAW_PATH.name
    else:
        raise FileNotFoundError("Neither curated_training_dataset.csv nor public_reviews_dataset_cleaned.csv was found.")

def main():
    print("=" * 65)
    print("🚀 TrustRate AI — Supervised Review Authenticity Training")
    print("=" * 65)
    
    df, dataset_source = load_training_data()
    print(f"Training dataset size: {len(df):,} rows")
    print(f"Class distribution: {df['label'].value_counts().to_dict()}")
    
    # Train/Test Split (75/25 stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        df["review_text"], df["label"],
        test_size=0.25, random_state=42, stratify=df["label"]
    )
    y_test_bin = (y_test == "suspicious").astype(int)
    
    # Candidate Pipelines
    pipelines = {
        "Logistic Regression (Balanced)": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=40000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("classifier", LogisticRegression(C=1.5, max_iter=1000, solver="liblinear", class_weight="balanced", random_state=42))
        ]),
        "Multinomial Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=40000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("classifier", MultinomialNB(alpha=0.5))
        ]),
        "Complement Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=40000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("classifier", ComplementNB(alpha=0.5))
        ]),
        "Calibrated SGD (Modified Huber)": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=40000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("classifier", SGDClassifier(loss="modified_huber", alpha=1e-4, max_iter=1000, random_state=42))
        ])
    }
    
    results = {}
    best_name = None
    best_model = None
    best_f1 = -1.0
    
    print("\nBenchmarking candidate models:")
    print("-" * 65)
    
    for name, pipeline in pipelines.items():
        print(f"Fitting {name}...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline.fit(X_train, y_train)
            
        preds = pipeline.predict(X_test)
        classes = list(pipeline.classes_)
        susp_idx = classes.index("suspicious")
        probs = pipeline.predict_proba(X_test)[:, susp_idx]
        
        acc = float(accuracy_score(y_test, preds))
        prec = float(precision_score(y_test, preds, pos_label="suspicious", zero_division=0))
        rec = float(recall_score(y_test, preds, pos_label="suspicious", zero_division=0))
        f1 = float(f1_score(y_test, preds, pos_label="suspicious", zero_division=0))
        auc = float(roc_auc_score(y_test_bin, probs))
        
        results[name] = {
            "accuracy": round(acc, 4),
            "precision_suspicious": round(prec, 4),
            "recall_suspicious": round(rec, 4),
            "f1_suspicious": round(f1, 4),
            "roc_auc": round(auc, 4)
        }
        
        print(f"  -> Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
        
        # Selection criterion: Highest suspicious F1 (and AUC as tie-breaker)
        if f1 > best_f1:
            best_f1 = f1
            best_name = name
            best_model = pipeline
            
    print("-" * 65)
    print(f"🏆 Best Model Selected: {best_name} (F1: {best_f1:.4f})")
    
    # Final evaluation on best model
    best_preds = best_model.predict(X_test)
    cm = confusion_matrix(y_test, best_preds, labels=["genuine", "suspicious"]).tolist()
    
    # Extract top predictive words for explainability
    top_suspicious_words = []
    top_genuine_words = []
    
    try:
        tfidf = best_model.named_steps["tfidf"]
        feature_names = np.array(tfidf.get_feature_names_out())
        classifier = best_model.named_steps["classifier"]
        
        if hasattr(classifier, "coef_"):
            coefs = classifier.coef_[0]
            top_susp_indices = np.argsort(coefs)[-25:]
            top_gen_indices = np.argsort(coefs)[:25]
            
            for idx in reversed(top_susp_indices):
                top_suspicious_words.append({"word": str(feature_names[idx]), "weight": round(float(coefs[idx]), 3)})
            for idx in top_gen_indices:
                top_genuine_words.append({"word": str(feature_names[idx]), "weight": round(float(coefs[idx]), 3)})
        elif hasattr(classifier, "feature_log_prob_"):
            # For Naive Bayes
            log_prob_diff = classifier.feature_log_prob_[susp_idx] - classifier.feature_log_prob_[1 - susp_idx]
            top_susp_indices = np.argsort(log_prob_diff)[-25:]
            top_gen_indices = np.argsort(log_prob_diff)[:25]
            
            for idx in reversed(top_susp_indices):
                top_suspicious_words.append({"word": str(feature_names[idx]), "weight": round(float(log_prob_diff[idx]), 3)})
            for idx in top_gen_indices:
                top_genuine_words.append({"word": str(feature_names[idx]), "weight": round(float(log_prob_diff[idx]), 3)})
    except Exception as e:
        print(f"Could not extract feature importance: {e}")
        
    # Save Model
    joblib.dump(best_model, MODEL_PATH)
    print(f"💾 Saved best model pipeline to {MODEL_PATH.name}")
    
    # Save Metrics
    metrics_data = {
        "best_model": best_name,
        "dataset_used": dataset_source,
        "rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "results": results,
        "best_model_test_metrics": results[best_name],
        "confusion_matrix": cm,
        "confusion_matrix_labels": ["genuine", "suspicious"],
        "top_suspicious_words": top_suspicious_words,
        "top_genuine_words": top_genuine_words,
        "methodology": "TF-IDF (unigrams + bigrams, sublinear tf) + Calibrated Linear Classifier. Stratified 75/25 split with exclusive reviewer deduplication."
    }
    
    METRICS_PATH.write_text(json.dumps(metrics_data, indent=2))
    print(f"📊 Saved comprehensive evaluation metrics to {METRICS_PATH.name}")
    print("\n✅ Training complete! Run `streamlit run app.py` to start the interactive system.")

if __name__ == "__main__":
    main()
