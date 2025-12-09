import os
import torch
import joblib
import warnings
import numpy as np
import pandas as pd
from src.visualization import save_pairplot,plot_confusion_matrix, plot_roc_curve
from src.data_preparation import preprocess_dataset,process_and_save_features, split_dataset
from src.modeling import (baselineModel, traditionalModels, transformerModel, evaluate_transformerModel,
                          evaluate_model, error_analysis, get_feature_importance, compare_models)
from scipy.sparse import hstack
from sklearn.preprocessing import StandardScaler

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # For store all models results
    finalResults=[]
    features =[
        "specialCharRatio",
        "countSemicolons",
        "countPrepositions",
        "countSecondPerson",
        "emotionalValenceScore"]
    # To prevent Data Leakage for features
    scaler = StandardScaler()
    # Data Preprocessing
    df = preprocess_dataset()
    print(f"Cleaned data: {df.head()}\n")
    # Feature Engineering
    df_features = process_and_save_features()
    print(f"Feature Engineering: {df_features.head()}\n")
    save_pairplot(df_features, features, "reports/figures/FeaturePairplot.png")
    # Split the data into training, validation, and test sets
    train_df, val_df, test_df = split_dataset()
    # Convert text labels to numbers Human-> 1 AI-> 0
    pd.set_option('future.no_silent_downcasting', True)
    train_df['label'] = train_df['label'].replace({'ai': 0, 'human': 1}).astype('int64')
    val_df['label'] = val_df['label'].replace({'ai': 0, 'human': 1}).astype('int64')
    test_df['label'] = test_df['label'].replace({'ai': 0, 'human': 1}).astype('int64')

    # Standerlization Features
    train_df[features] = scaler.fit_transform(train_df[features])
    val_df[features] = scaler.transform(val_df[features])
    test_df[features] = scaler.transform(test_df[features])
    joblib.dump(scaler, "models/scaler.pkl")
    # Model Building, Training
    log_model, tfidf = baselineModel(train_df, val_df)
    svm, xgb = traditionalModels(train_df, val_df, tfidf, features)
    bert_model, bert_tokenizer = transformerModel(train_df, val_df,model_name="aubmindlab/bert-base-arabertv02")
    camelModel, camelTokenizer = transformerModel(train_df, val_df, model_name="CAMeL-Lab/bert-base-arabic-camelbert-mix")
    araelectra_model, araelectra_tokenizer = transformerModel(train_df, val_df, model_name="aubmindlab/araelectra-base-discriminator")

    # Train Traditional ML on Features
    tfidfLoad = joblib.load("models/tfidf.pkl")
    X_test_tfidf = tfidfLoad.transform(test_df["clean_text_stemmed"])
    X_test_feat = test_df[features].values
    X_test_combined = hstack([X_test_tfidf, X_test_feat])
    # To ensure all the values in label either 0 or 1 else will be 0
    y_test = test_df["label"].fillna(0)

    log_results = evaluate_model(log_model, X_test_tfidf, y_test, "Logistic Regression")
    finalResults.append(log_results)
    #Training on 10005 Features for SVM Model
    svm_results = evaluate_model(svm, X_test_combined, y_test, "SVM")
    finalResults.append(svm_results)
    #Training on 10005 Features for XGB Model
    xgb_results = evaluate_model(xgb, X_test_combined, y_test, "XGBoost")
    finalResults.append(xgb_results)
    bert_results = evaluate_transformerModel(bert_model, bert_tokenizer, test_df,"AraBERT",device)
    finalResults.append(bert_results)
    camel_results = evaluate_transformerModel(camelModel, camelTokenizer, test_df,"CAMeL-Mix",device)
    finalResults.append(camel_results)
    araelectra_results = evaluate_transformerModel(araelectra_model, araelectra_tokenizer, test_df,"AraELECTRA",device)
    finalResults.append(araelectra_results)

    results_path = "reports/evaluation_results.csv"
    results_to_save = [{k: v for k, v in res.items() if k not in ['y_true', 'y_pred']} for res in finalResults]
    results_df = pd.DataFrame(results_to_save)
    os.makedirs("reports", exist_ok=True)
    results_df.to_csv(results_path, index=False, encoding="utf-8-sig")
    print(f"\nEvaluation results saved to {results_path}")

    # 4. Error Analysis (Task 4.4)
    print("\n4.4 Running Final Error Analysis")

    # Best Model based on F1-Score
    bestModel = max(finalResults, key=lambda x: x['F1'])

    print(f"\nBest Model (based on F1-Score) is: {bestModel['Model']} ")
    # Perform error analysis
    error_analysis(
        model_name=bestModel['Model'],
        X_test_df=test_df,
        y_true=bestModel['y_true'],
        y_pred=bestModel['y_pred']
    )
    # 5.1 Feature Importance
    # 5.1 Model Comparison and Visualization (Task 5.1)
    print("\n 5.1 Running Model Comparison")
    summary_df = compare_models(metrics_paths=[results_path],
                                save_csv="reports/final_comparison_summary.csv")
    print("Final ranked model comparison saved and visualized.")
    print(summary_df)
    # 5.2 Best Tradition Models
    traditional_models = [res for res in finalResults if res['Model'] in ["SVM", "XGBoost", "Logistic Regression"]]
    best_traditional_model_data = max(traditional_models, key=lambda x: x['F1'])
    best_traditional_name = best_traditional_model_data['Model']
    if best_traditional_name == "XGBoost":
        model_to_analyze = joblib.load("models/xgboostModel.pkl")
    elif best_traditional_name == "Logistic Regression":
        model_to_analyze = joblib.load("models/logisticModel.pkl")
    else:  # SVM
        model_to_analyze = joblib.load("models/svmModel.pkl")
    tfidf_loaded = joblib.load("models/tfidf.pkl")
    tfidf_feature_names = tfidf_loaded.get_feature_names_out()
    print(f"\nFeature Importance for Best Traditional Model: {best_traditional_name}")
    get_feature_importance(
        model=model_to_analyze,
        tfidf_feature_names=tfidf_feature_names,
        engineeredFeatures=features
    )

    print("\nProject execution complete.")

if __name__ == "__main__":
    main()