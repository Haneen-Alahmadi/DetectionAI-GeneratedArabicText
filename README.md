#  Detection of AI-Generated Arabic Text: A Data Mining Approach
# AI-Arabic-Text-Detection
**AI Arabic Text Detection is classifier distinguish AI-generated and human-written Arabic text**

## Dataset
- KFUPM-JRCAI/arabic-generated-abstracts 

## Structure
```text
AI-Arabic-Text-Detection/
├── data/
│   ├── raw/
│   │   ├── by_polishing/
│   │   ├── from_title/
│   │   └── from_title_and_content/
│   │       └── dataset_dict.json
│   ├── processed/
│   │   ├── AIArabic_binary.csv
│   │   ├── AIArabic_binary_clean.csv
│   │   └── AIArabic_features.csv
│   └── external/
│
├── notebooks/
│   ├── phase1Data_acquisition.ipynb
│   └── phase2Preprocessing.ipynb
│
├── models/
│   ├── araelectra-base-discriminator
│   ├── bert-base-arabertv02/
│   ├── bert-base-arabic-camelbert-mix/
│   ├── logisticModel.pkl
│   ├── scaler.pkl
│   ├── svmModel.pkl
│   ├── tfidf.pkl
│   └── xgboostModel.pkl
│
├── reports/
│   ├── figures/
│   │   ├── commonTermsForAI.png
│   │   ├── commonTermsForHuman.png
│   │   ├── confusion_AraELECTRA.png
│   │   ├── confusion_AraBERT.png
│   │   ├── confusion_CAMeL-Mix.png
│   │   ├── confusion_Logistic Regression.png
│   │   ├── confusion_SVM.png
│   │   ├── confusion_XGBoost.png
│   │   ├── feature_importance_top.png
│   │   ├── FeaturePairpair.png
│   │   ├── models_comparison_bar.png
│   │   ├── NumberOfText.png
│   │   ├── roc_AraELECTRA.png
│   │   ├── roc_AraBERT.png
│   │   ├── roc_CAMeL-Mix.png
│   │   ├── roc_Logistic Regression.png
│   │   ├── roc_SVM.png
│   │   └── roc_XGBoost.png
│   ├── presentations/
│   ├── error_analysis.csv
│   ├── evaluation_results.csv
│   ├── feature_importance.csv
│   └── final_comparison_summary.csv
│
├── src/
│   ├── data_preparation.py
│   ├── modeling.py
│   ├── utils.py
│   └── visualization.py
│
├── docs/
├── main.py
├── requirements.txt
├── environment.yml
├── .gitignore
└── README.md
               
          

## requirements
- pip install -r requirements
- pip uninstall torch torchvision torchaudio -y
- pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
