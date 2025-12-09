import os
import torch
import evaluate
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from scipy.sparse import hstack
from torch.nn import CrossEntropyLoss
from sklearn.metrics import (classification_report,accuracy_score, precision_score, recall_score,f1_score, roc_auc_score)
from src.visualization import plot_confusion_matrix, plot_roc_curve
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from src.utils import save_model

# 4.1 Baseline Model Logistic Regression
def baselineModel(train_df, val_df):
#   Create TF-IDF
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train = tfidf.fit_transform(train_df["clean_text_stemmed"])
    X_val   = tfidf.transform(val_df["clean_text_stemmed"])
    y_train, y_val = train_df["label"], val_df["label"]
#   We have Imbalanced class labels AI = 33552 Human = 8388 to solve this problem we need to use Class Weights to handel imbalancing with text data
    model = LogisticRegression(max_iter=300, class_weight='balanced')
    model.fit(X_train, y_train)
    prediction = model.predict(X_val)
    print("\nClassification Report for Logistic Regression")
    print(classification_report(y_val, prediction, digits=4))
    save_model(model, "logisticModel")
    save_model(tfidf, "tfidf")
    return model, tfidf


# 4.2 Traditional ML Models SVM and XGBoost
def traditionalModels(train_df, val_df, tfidf,features):

    X_train_tfidf = tfidf.transform(train_df["clean_text_stemmed"])
    X_val_tfidf = tfidf.transform(val_df["clean_text_stemmed"])
    y_train, y_val = train_df["label"], val_df["label"]
    X_train_feat = train_df[features].values
    X_val_feat = val_df[features].values

# Features Engineering
    X_train_combined = hstack([X_train_tfidf, X_train_feat])
    X_val_combined = hstack([X_val_tfidf, X_val_feat])
    # SVM
    svm = LinearSVC(class_weight='balanced')
    svm.fit(X_train_combined, y_train)
    svmPrediction = svm.predict(X_val_combined)
    print("\nClassification Report for SVM using Combined Features")
    print(classification_report(y_val, svmPrediction, digits=4))
    save_model(svm, "svmModel")

    # XGBoost
    xgb = XGBClassifier(
        eval_metric="logloss",
        early_stopping_rounds=10,
        random_state=42,
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=4
    )

    xgb.fit(
        X_train_combined.toarray(), y_train,
        eval_set=[(X_val_combined.toarray(), y_val)],
        verbose=False
    )
    xgbPrediction = xgb.predict(X_val_combined.toarray())
    print("\nClassification Report for XGBoost using Combined Features")
    print(classification_report(y_val, xgbPrediction, digits=4))
    save_model(xgb, "xgboostModel")

    return svm, xgb

# 4.3 — Deep Learning AraBERT and CAMelBERT and AraELECTRE
def compute_bert_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    accuracy = evaluate.load("accuracy").compute(predictions=preds, references=labels)
    precision = evaluate.load("precision").compute(predictions=preds, references=labels, average="binary")
    recall = evaluate.load("recall").compute(predictions=preds, references=labels, average="binary")
    f1 = evaluate.load("f1").compute(predictions=preds, references=labels, average="binary")

    return {
        "accuracy": round(accuracy["accuracy"], 4),
        "precision": round(precision["precision"], 4),
        "recall": round(recall["recall"], 4),
        "f1": round(f1["f1"], 4)
    }

def transformerModel(train_df, val_df,model_name):
    out_dir = f"models/{model_name.split('/')[-1]}"
    tokenizer_dir = os.path.join(out_dir, "tokenizer")

    train_df = train_df.copy()
    val_df = val_df.copy()

    if os.path.exists(out_dir) and os.path.exists(tokenizer_dir):
        print(f"\n✔ Model already exists. Loading without training...\n")
        model = AutoModelForSequenceClassification.from_pretrained(out_dir)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        return model, tokenizer
    epochs = 4
    lr = 2e-5
    max_len = 256
    batch_size = 8

    print(f"\nFine-tuning model: {model_name}\n")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if train_df["label"].dtype == object:
        train_df["label"] = train_df["label"].map({"human": 1, "ai": 0})
        val_df["label"] = val_df["label"].map({"human": 1, "ai": 0})
#   Tokenization
    def tokenize(batch):
        return tokenizer(batch["clean_text"], padding=True, truncation=True, max_length=max_len)
#  Change label class into labels because hagging face trainer
    train_df = train_df.rename(columns={"label": "labels"})
    val_df = val_df.rename(columns={"label": "labels"})
    ds_train = Dataset.from_pandas(train_df)
    ds_val = Dataset.from_pandas(val_df)
    ds_train = ds_train.map(tokenize, batched=True)
    ds_val = ds_val.map(tokenize, batched=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    if device == "cuda":
        print(f"\nGPU is available: {torch.cuda.get_device_name(0)}")
    else:
        print("\nNo GPU detected\n")
#   Handle class imbalance
    labels = train_df["labels"].values
    class_counts = np.bincount(labels)
    weights = torch.tensor([1.0 / c for c in class_counts], dtype=torch.float32)
    loss_fn = CrossEntropyLoss(weight=weights.to("cuda" if torch.cuda.is_available() else "cpu"))

    # Custom trainer class
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            loss = loss_fn(logits.view(-1, model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    args = TrainingArguments(
        output_dir=out_dir,
        evaluation_strategy="no",
        save_strategy="no",
        load_best_model_at_end=False,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        report_to="none",
        logging_dir="./logs"
    )
    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        tokenizer=tokenizer,
        compute_metrics=compute_bert_metrics)
# Train the model
    trainer.train()

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(tokenizer_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(tokenizer_dir)

    return model, tokenizer

# Task 4.4- Comprehensive Evaluation
# Evaluate Transformer Model
def evaluate_transformerModel(model, tokenizer, test_df, model_name, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"\nEvaluating on device: {device.upper()}")
    labels = test_df["label"].values.astype(int).tolist()
    texts = test_df["clean_text"].tolist()
    inputs = tokenizer(texts,
                       padding=True,
                       truncation=True,
                       max_length=256,
                       return_tensors="pt")
    dataset = TensorDataset(inputs['input_ids'],
                            inputs['attention_mask'])
    EVAL_BATCH_SIZE = 16
    dataloader = DataLoader(dataset, batch_size=EVAL_BATCH_SIZE)
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
            all_preds.extend(preds)
            all_probs.extend(probs)

    preds = np.array(all_preds)
    probs = np.array(all_probs)
    labels = np.array(labels)
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, pos_label=1)
    rec = recall_score(labels, preds, pos_label=1)
    f1 = f1_score(labels, preds, pos_label=1)
    roc_auc = roc_auc_score(labels, probs)
    print(f"\nEvaluation Results for {model_name}:")
    print(f"\nAccuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    plot_confusion_matrix(labels, preds, model_name)
    plot_roc_curve(labels, probs, model_name)

    return {
        "Model": model_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "ROC-AUC": roc_auc,
        "y_true": labels,
        "y_pred": preds,
    }


def evaluate_model(model, X_test, y_test, model_name):
    # Convert sparse matrix to dense
    if hasattr(X_test, "toarray"):
        X_test = X_test.toarray()
    y_pred = model.predict(X_test)
    y_proba = None
    # Convert Y label into 0 and 1
    if y_test.dtype == object:
        y_true_num = y_test.map({"human": 1, "ai": 0}).values
    else:
        y_true_num = y_test.values
    if y_pred.dtype == object:
        y_pred_num = pd.Series(y_pred).map({"human": 1, "ai": 0}).values
    else:
        y_pred_num = y_pred

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_proba = model.decision_function(X_test)
    else:
        y_proba = None
    # Result metrics
    results = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true_num, y_pred_num),
        "Precision": precision_score(y_true_num, y_pred_num, pos_label=1),  # Human=1
        "Recall": recall_score(y_true_num, y_pred_num, pos_label=1),  # Human=1
        "F1": f1_score(y_true_num, y_pred_num, pos_label=1),  # Human=1
    }
    if y_proba is not None:
        results["ROC-AUC"] = roc_auc_score(y_true_num, y_proba)
        # 5. Plotting (Requires y_proba for ROC)
        plot_confusion_matrix(y_true_num, y_pred_num, model_name)
        plot_roc_curve(y_true_num, y_proba, model_name)
    print(f"\nResult for {model_name}:")
    for k, v in results.items():
        if isinstance(v, (float, int)):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
    results["y_true"] = y_true_num
    results["y_pred"] = y_pred_num
    return results

# 4.4 Analysis Misclassified Examples
def error_analysis(model_name, X_test_df, y_true, y_pred, save_path="reports/error_analysis.csv"):
    print(f"Error Analysis Report — {model_name}")
    analysis_df = X_test_df.copy()
    analysis_df["y_true"] = y_true
    analysis_df["y_pred"] = y_pred
    analysis_df["error"] = analysis_df["y_true"] != analysis_df["y_pred"]
    analysis_df["predicted"] = analysis_df["y_pred"].map({0: "ai", 1: "human"})
    analysis_df["label"] = analysis_df["y_true"].map({0: "ai", 1: "human"})
    errors = analysis_df[analysis_df["error"]]
    print(f"Total misclassified samples: {len(errors)} / {len(analysis_df)}")
    # FP: Predicted Human (1), Actual AI (0)
    fp_errors = errors[(errors['y_pred'] == 1) & (errors['y_true'] == 0)].head(3)
    # FN: Predicted AI (0), Actual Human (1)
    fn_errors = errors[(errors['y_pred'] == 0) & (errors['y_true'] == 1)].head(3)
    print("\n--- False Positives (FP): AI classified as Human ---")
    for _, row in fp_errors.iterrows():
        print(f"\n- Text: {row['text'][:100]}... | True: {row['label']} | Predicted: {row['predicted']}")
    print("\n--- False Negatives (FN): Human classified as AI ---")
    for _, row in fn_errors.iterrows():
        print(f"\n- Text: {row['text'][:100]}... | True: {row['label']} | Predicted: {row['predicted']}")
    os.makedirs("reports", exist_ok=True)
    errors.to_csv(save_path, index=False, encoding="utf-8-sig")
    error_rate = len(errors) / len(analysis_df)
    print(f"\nOverall Error Rate: {error_rate*100:.2f}%")
    print(f"Misclassified samples saved to {save_path}")
    return errors

# 5.1 Compare the results of all models
def compare_models(metrics_paths=("reports/final_models_comparison.csv",
                           "reports/evaluation_results.csv"),
                   save_csv="reports/models_comparison_final.csv",
                   save_fig="reports/figures/models_comparison_bar.png"):
    os.makedirs("reports/figures", exist_ok=True)
    frames = []
    for p in metrics_paths:
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["Model"], keep="last")

    metric_cols = ["Accuracy","Precision","Recall","F1","ROC-AUC"]
    df_ranked = df.sort_values(by="F1", ascending=False)
    df_ranked.to_csv(save_csv, index=False, encoding="utf-8-sig")

    ax = df_ranked.set_index("Model")[metric_cols].plot(kind="bar", figsize=(9,5))
    plt.title("Model Comparison (higher is better)")
    plt.ylabel("Score")
    plt.ylim(0,1)
    plt.tight_layout()
    plt.savefig(save_fig, dpi=300)
    plt.close()
    return df_ranked

# 5.2 Extract Best Features
def get_feature_importance(model, tfidf_feature_names, engineeredFeatures, top_n=20,
                       save_csv="reports/feature_importance.csv",save_fig="reports/figures/feature_importance_top.png"):
    os.makedirs("reports/figures", exist_ok=True)
#   Extract the Features
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        # For LR, SVM
        importances = np.abs(model.coef_[0])
    else:
        print("Model does not support feature importance extraction.")
        return None
    #   Merage the Features
    feature = list(tfidf_feature_names) + list(engineeredFeatures)
    if len(importances) != len(feature):
        print(f"Warning: Feature length mismatch ({len(importances)} importances vs {len(feature)} names)")
        min_len = min(len(importances), len(feature))
        importances, feature = importances[:min_len], feature[:min_len]
    feature_importance_df = pd.DataFrame({
        "Feature": feature,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)
    # Save the results
    feature_importance_df.to_csv(save_csv, index=False, encoding="utf-8-sig")
    # Plotting the results
    top_features = feature_importance_df.head(top_n)
    plt.figure(figsize=(8, 6))
    plt.barh(top_features["Feature"][::-1], top_features["Importance"][::-1])
    plt.title(f"Top {top_n} Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(save_fig, dpi=300)
    plt.close()
    print(f"\nTop {top_n} features saved to {save_csv}")
    print(f"Figure saved to {save_fig}")
    print(top_features)
    return feature_importance_df
