# train_models.py
import json, os, joblib, argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import classification_report

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def preprocess_df(df):
    # required fields: clause, risk_level, contract_type, suggested_alternative (optional)
    df = df.dropna(subset=['clause', 'risk_level', 'contract_type']).reset_index(drop=True)
    df['clause'] = df['clause'].astype(str)
    df['risk_level'] = df['risk_level'].astype(str)
    df['contract_type'] = df['contract_type'].astype(str)
    # fill suggested_alternative with blank if missing
    if 'suggested_alternative' not in df.columns:
        df['suggested_alternative'] = ""
    else:
        df['suggested_alternative'] = df['suggested_alternative'].fillna("")
    return df

def make_embeddings(sentences, model_name, batch_size=64):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences, show_progress_bar=True, batch_size=batch_size, convert_to_numpy=True)
    return embeddings, model

def train_lightgbm(X_train, y_train, X_val, y_val, num_class):
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    params = {
        "objective": "multiclass",
        "num_class": num_class,
        "metric": "multi_logloss",
        "verbosity": -1,
        "seed": 42
    }
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=500,
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)]
    )
    return model


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    df = load_json(args.input)
    df = preprocess_df(df)

    # Label encoders
    le_risk = LabelEncoder(); y_risk = le_risk.fit_transform(df['risk_level'])
    le_type = LabelEncoder(); y_type = le_type.fit_transform(df['contract_type'])

    # embeddings
    print("Computing embeddings with", args.embed_model)
    embeddings, sbert_model = make_embeddings(df['clause'].tolist(), args.embed_model, batch_size=32)
    # Save embeddings + suggestions for nearest-neighbour suggestions
    np.save(os.path.join(args.output_dir, "train_embeddings.npy"), embeddings)
    # Save suggestions mapping: list of dicts {clause, suggested_alternative}
    suggestions = df[['clause','suggested_alternative']].to_dict(orient='records')
    with open(os.path.join(args.output_dir, "train_suggestions.json"), 'w', encoding='utf-8') as f:
        json.dump(suggestions, f, ensure_ascii=False, indent=2)

    # Save model name for inference to reload same SBERT
    joblib.dump(args.embed_model, os.path.join(args.output_dir, "embed_model_name.joblib"))

    # Train risk model
    X_train, X_val, y_train, y_val = train_test_split(embeddings, y_risk, test_size=0.15, random_state=42, stratify=y_risk)
    print("Training risk model...")
    risk_model = train_lightgbm(X_train, y_train, X_val, y_val, num_class=len(le_risk.classes_))
    # Evaluate
    y_pred = np.argmax(risk_model.predict(X_val), axis=1)
    print("Risk model report:")
    print(classification_report(y_val, y_pred, target_names=le_risk.classes_))
    # Save
    joblib.dump((risk_model, 'lgb'), os.path.join(args.output_dir, "risk_model.joblib"))
    joblib.dump(le_risk, os.path.join(args.output_dir, "le_risk.joblib"))

    # Train contract type model (clause-level supervision; document aggregation later)
    X_train_t, X_val_t, y_train_t, y_val_t = train_test_split(embeddings, y_type, test_size=0.15, random_state=42, stratify=y_type)
    print("Training contract type model...")
    type_model = train_lightgbm(X_train_t, y_train_t, X_val_t, y_val_t, num_class=len(le_type.classes_))
    y_pred_t = np.argmax(type_model.predict(X_val_t), axis=1)
    print("Contract type model report:")
    print(classification_report(y_val_t, y_pred_t, target_names=le_type.classes_))
    joblib.dump((type_model, 'lgb'), os.path.join(args.output_dir, "type_model.joblib"))
    joblib.dump(le_type, os.path.join(args.output_dir, "le_type.joblib"))

    # Save SBERT model to disk (the name enough, but saving object optional)
    joblib.dump(sbert_model, os.path.join(args.output_dir, "sbert_model.joblib"))

    print("All models & artifacts saved to", args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="path to json dataset")
    parser.add_argument("--output_dir", default="models", help="output folder for models")
    parser.add_argument("--embed_model", default="paraphrase-multilingual-MiniLM-L12-v2", help="sentence-transformer model name")
    args = parser.parse_args()
    main(args)
