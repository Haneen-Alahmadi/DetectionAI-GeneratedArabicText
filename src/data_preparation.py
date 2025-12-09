import re
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer

# Normalizing Data

def normalize_arabic(text):
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "و", text)
    text = re.sub("ئ", "ي", text)
    text = re.sub("ة", "ه", text)
    # text = re.sub("[^ء-ي\s]", "", text) # I should deleted because i neeed feature specialCharRatio and countSemicolons
    text = re.sub(r'(.)\1+', r"\1\1", text) # removing longation
    text = re.sub(r'\s+', ' ', text).strip() # removing spaces
    # text = re.sub(r'[^\u0600-\u06FF\s]', '', text) # removing non-Arabic characters # I should deleted because i neeed feature specialCharRatio and countSemicolons
    return text

# Removing Diacritics

def remove_diacritics(text):
    diacritics = re.compile("""
                             ّ    | # Shadda
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """, re.VERBOSE)
    return re.sub(diacritics, '', text)

# Removing Stop Words

def remove_stop_words(text):
    stops = set(stopwords.words("arabic"))
    words = text.split()
    filtered = [w for w in words if w not in stops and len(w) > 1]
    return " ".join(filtered)

# Stemming

def stem_text(text):
    ARstemmer = ISRIStemmer()
    words = text.split()
    words = [ARstemmer.stem(w) for w in words]
    return " ".join(words)

def preprocess_text(text, applyStemming=True):
    text = normalize_arabic(text)
    text = remove_diacritics(text)
    text = remove_stop_words(text)
    if applyStemming:
        text = stem_text(text)

    return text


def preprocess_dataset(
    input_path="data/processed/AIArabic_binary.csv",
    output_path="data/processed/AIArabic_binary_clean.csv"):

    if os.path.exists(output_path):
        print(f"Cleaned dataset already exists at {output_path}")
        return pd.read_csv(output_path)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found at: {os.path.abspath(input_path)}")

    df = pd.read_csv(input_path)
    # Remove the duplicated rows
    rows_before = len(df)
    df.drop_duplicates(inplace=True)
    rows_after = len(df)
    print(f" deleted {rows_before - rows_after} frequent rows")
    # To dealing with Implicit Missingness
    df["text"] = df["text"].fillna('').astype(str)
#   For Bert Model
    df["clean_text"] = df["text"].astype(str).apply(lambda x: normalize_arabic(remove_diacritics(x)))
#   For Logistic and Traditional Models
    df["clean_text_stemmed"] = df["text"].astype(str).apply(lambda x: preprocess_text(x, applyStemming=True))
    df = df[(df["clean_text"].str.strip() != "") & (df["clean_text_stemmed"].str.strip() != "")]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return df

# Feature 8: Special Characters Ratio
def specialCharRatio(text):
    if not isinstance(text, str) or len(text) == 0:
        return 0
    specials = re.findall(r"[^a-zA-Z0-9\s\u0621-\u064A]", text)
    return len(specials) / len(text)

# Feature 29: Number of Semi-Colons
def countSemicolons(text):
    if not isinstance(text, str):
        return 0
    return text.count("؛") + text.count(";")

# Feature 50: Number of Prepositions
def countPrepositions(text):
    if not isinstance(text, str):
        return 0
    prepositions = ["متى","حاشا","ك","مذ","منذ","عدا","مع","من", "إلى", "على", "في", "عن", "ب", "ل", "حتى"]
    return sum(text.split().count(p) for p in prepositions)

# Feature 71: 2nd Person Words
def countSecondPerson(text):
    if not isinstance(text, str):
        return 0
    second_person_words = ["أنت", "أنتم", "أنتما","أنتن", "تفعل", "تعمل", "تفكر", "تريد", "تقول", "تستطيع", "تكتب"]
    return sum(text.split().count(w) for w in second_person_words)

# Feature 92: Emotional Valence Score
def emotionalValenceScore(text):

    if not isinstance(text, str) or not text.strip():
        return 0.0
    positive = {"سعيد", "فرح", "كريم", "فخور", "مؤدب", "هادئ",
        "صادق", "شجاع", "مضحك", "محب", "مرتاح", "متفائل"}

    negative= {"غاضب", "حزين", "خائف", "محبط", "منزعج", "عنيد",
        "جشع", "كسول", "ساذج", "مكتئب", "قلق", "خائب"}

    neutral= {"متفاجئ", "مشغول", "فضولي", "محرج", "محايد"}

    words = text.split()
    score = 0
    for w in words:
        if w in positive:
            score += 1
        elif w in negative:
            score -= 1
        elif w in neutral:
            score += 0
    valence = score / len(words)

    return round(valence, 4)

# Create 5 features to classify labels

def process_and_save_features(
        input_path="data/processed/AIArabic_binary_clean.csv",
        output_path="data/processed/AIArabic_features.csv"):

    if os.path.exists(output_path):
        print(f"Feature file already exists at {output_path}")
        return pd.read_csv(output_path)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found at: {os.path.abspath(input_path)}")

    df = pd.read_csv(input_path)
    df["specialCharRatio"] = df["clean_text_stemmed"].apply(specialCharRatio)
    df["countSemicolons"] = df["clean_text_stemmed"].apply(countSemicolons)
    df["countPrepositions"] = df["clean_text_stemmed"].apply(countPrepositions)
    df["countSecondPerson"] = df["clean_text_stemmed"].apply(countSecondPerson)
    df["emotionalValenceScore"] = df["clean_text_stemmed"].apply(emotionalValenceScore)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return df

# Splitting the dataset into 70% training 15% validation 15% testing
def split_dataset(
        input_path="data/processed/AIArabic_features.csv",
        output_dir="data/splits"):

    train_path = f"{output_dir}/train.csv"
    val_path = f"{output_dir}/val.csv"
    test_path = f"{output_dir}/test.csv"
    if all(os.path.exists(p) for p in [train_path, val_path, test_path]):
        print(f"Split files already exist in '{output_dir}', loading them...")
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        return train_df, val_df, test_df

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_path)
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df["label"])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df["label"])
    train_df.to_csv(f"{output_dir}/train.csv", index=False, encoding="utf-8-sig")
    val_df.to_csv(f"{output_dir}/val.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(f"{output_dir}/test.csv", index=False, encoding="utf-8-sig")
    print(f"Train: {len(train_df)}, Validate,: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df