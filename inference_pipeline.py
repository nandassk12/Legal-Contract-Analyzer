# inference_pipeline.py
import os, json, joblib, argparse, hashlib
from datetime import datetime, timezone
from sentence_transformers import SentenceTransformer
import numpy as np
import pdfplumber, docx, re
from langdetect import detect
from sklearn.neighbors import NearestNeighbors
from pathlib import Path
import spacy
import pytesseract
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from googletrans import Translator
translator = Translator()


# Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\NEKILESH\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# Load English spaCy model
nlp = spacy.load("en_core_web_sm")

translator = Translator()

def file_hash(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def extract_text(file_path):
    ext = Path(file_path).suffix.lower()
    text = ""

    if ext == ".pdf":
        try:
            texts=[]
            with pdfplumber.open(file_path) as pdf:
                for p in pdf.pages:
                    texts.append(p.extract_text() or "")
            text = "\n".join(texts).strip()
        except Exception as e:
            print(f"[WARN] pdfplumber failed: {e}")

        if not text.strip():
            print("[INFO] No text found via pdfplumber, switching to OCR...")
            try:
                pages = convert_from_path(file_path, dpi=300)
                for page in pages:
                    text += pytesseract.image_to_string(page, lang="eng+tam+hin") + "\n"
            except Exception as e:
                print(f"[ERROR] OCR extraction failed: {e}")
                text = ""
        return text

    elif ext in [".docx", ".doc"]:
        doc = docx.Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])

    else:  # plain text
        with open(file_path,'r',encoding='utf-8') as f:
            return f.read()

def segment_clauses(text, language_hint=None):
    try:
        if language_hint and language_hint.lower().startswith('en'):
            doc = nlp(text)
            sents = [s.text.strip() for s in doc.sents if s.text.strip()]
        else:
            lang = detect(text) if not language_hint else language_hint
            if lang.startswith('en'):
                doc = nlp(text)
                sents = [s.text.strip() for s in doc.sents if s.text.strip()]
            else:
                sents = re.split(r'\n+|(?<=[।.!?])\s+', text)
                sents = [s.strip() for s in sents if s.strip()]
    except Exception:
        sents = [l.strip() for l in text.split("\n") if l.strip()]

    return [s for s in sents if len(s) >= 5]

def load_artifacts(models_dir):
    embed_model_name = joblib.load(os.path.join(models_dir, "embed_model_name.joblib"))
    try:
        sbert = joblib.load(os.path.join(models_dir, "sbert_model.joblib"))
    except Exception:
        sbert = SentenceTransformer(embed_model_name)

    risk_model, risk_type = joblib.load(os.path.join(models_dir, "risk_model.joblib"))
    le_risk = joblib.load(os.path.join(models_dir, "le_risk.joblib"))
    type_model, type_type = joblib.load(os.path.join(models_dir, "type_model.joblib"))
    le_type = joblib.load(os.path.join(models_dir, "le_type.joblib"))

    train_embeddings = np.load(os.path.join(models_dir, "train_embeddings.npy"))
    with open(os.path.join(models_dir, "train_suggestions.json"), 'r', encoding='utf-8') as f:
        suggestions = json.load(f)

    nn = NearestNeighbors(n_neighbors=1, metric='cosine').fit(train_embeddings)

    return {
        "sbert": sbert,
        "embed_name": embed_model_name,
        "risk_model": risk_model,
        "risk_type": risk_type,
        "le_risk": le_risk,
        "type_model": type_model,
        "type_type": type_type,
        "le_type": le_type,
        "train_embeddings": train_embeddings,
        "suggestions": suggestions,
        "nn": nn
    }

def predict_from_clauses(clauses, models):
    emb = models['sbert'].encode(clauses, show_progress_bar=False, convert_to_numpy=True)
    if isinstance(emb, np.ndarray) and emb.ndim == 1:
        emb = emb.reshape(1, -1)

    # Risk prediction
    if models['risk_type'] == 'lgb':
        probs_r = models['risk_model'].predict(emb)
        preds_r = np.argmax(probs_r, axis=1)
    else:
        preds_r = models['risk_model'].predict(emb)
    labels_r = models['le_risk'].inverse_transform(preds_r)

    # Contract type prediction
    if models['type_type'] == 'lgb': 
        probs_t = models['type_model'].predict(emb)
        preds_t = np.argmax(probs_t, axis=1)
    else:
        preds_t = models['type_model'].predict(emb)
    labels_t = models['le_type'].inverse_transform(preds_t)

    return labels_r, labels_t, emb

from langdetect import detect

def nearest_suggestion(clause, clause_emb, models):
    """
    Get nearest neighbor suggestion for a clause and translate it
    to the clause's original language if needed.
    """
    # Ensure embeddings are 2D
    if clause_emb.ndim == 1:
        clause_emb = clause_emb.reshape(1, -1)

    # Find nearest neighbor in training embeddings
    dist, idx = models['nn'].kneighbors(clause_emb, n_neighbors=1, return_distance=True)
    idx0 = int(idx[0][0])
    suggestion = models['suggestions'][idx0].get('suggested_alternative', clause)

    # Detect clause language
    try:
        lang = detect(clause)
    except:
        lang = 'en'

    # Translate suggestion if clause language is not English
    try:
        if lang != 'en':
            suggestion_translated = translator.translate(suggestion, dest=lang)
            return suggestion_translated.text
        else:
            return suggestion
    except:
        return suggestion


def aggregate_doc_type(clause_types):
    if len(clause_types) == 0:
        return "Unknown"
    return max(set(clause_types), key=clause_types.count)

def aggregate_overall_risk(risk_labels):
    if any(r.lower() == 'high' for r in risk_labels):
        return "High"
    if any(r.lower() == 'medium' for r in risk_labels):
        return "Medium"
    return "Low"

def simple_summary(doc_text, contract_type):
    first_line = doc_text.strip().split("\n",1)[0][:300]
    return f"This appears to be a {contract_type}. Key items include payments, delivery or service terms, confidentiality, termination and dispute resolution. First line: {first_line}"

def explain_clause_model(clause, clause_emb, models, risk_label=None):
    """
    Explain clause using nearest neighbor + translate to clause language.
    """
    from langdetect import detect

    # Detect clause language
    try:
        lang = detect(clause)
    except:
        lang = 'en'

    # Get nearest suggestion
    if clause_emb.ndim == 1:
        clause_emb = clause_emb.reshape(1, -1)
    dist, idx = models['nn'].kneighbors(clause_emb, n_neighbors=1, return_distance=True)
    idx0 = int(idx[0][0])
    suggestion = models['suggestions'][idx0].get('suggested_alternative', clause)

    # Translate suggestion if needed
    try:
        if lang != 'en':
            translated = translator.translate(suggestion, dest=lang)
            explanation = translated.text
        else:
            explanation = suggestion
    except:
        explanation = suggestion

    # Append risk info in English (optional: translate too)
    if risk_label:
        explanation += f" ⚠️ {risk_label} risk: review carefully."

    return explanation



def analyze_document(file_path, models_dir, user_id="user_anon"):
    text = extract_text(file_path).strip()
    if not text:
        return {
            "document_id": Path(file_path).name,
            "contract_type": "Unknown",
            "summary": "No readable text could be extracted from the document.",
            "overall_risk_level": "Unknown",
            "key_clauses": [],
            "audit": {
                "file_hash": file_hash(file_path),
                "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
                "user_id": user_id,
                "model_version": "v1"
            }
        }

    try:
        lang = detect(text)
    except Exception:
        lang = 'en'

    clauses = segment_clauses(text, language_hint=lang)
    if not clauses:
        clauses = [text]

    artifacts = load_artifacts(models_dir)
    risk_labels, clause_types, clause_embeddings = predict_from_clauses(clauses, artifacts)

    doc_type = aggregate_doc_type(list(clause_types))
    overall_risk = aggregate_overall_risk(list(risk_labels))

    key_clauses = []

    for cl, rlab, emb in zip(clauses, risk_labels, clause_embeddings):
        risk_factor = rlab if rlab.lower() in ("medium", "high") else None
        clause_info = {
            "clause": cl,
            "risk_level": risk_factor,
            "explanation": explain_clause_model(cl, emb, artifacts,risk_factor)
        }

        if rlab.lower() == "high":
            suggestion = nearest_suggestion(cl,emb, artifacts)
            clause_info["suggested_alternative"] = suggestion

        key_clauses.append(clause_info)

    return {
        "document_id": Path(file_path).name,
        "contract_type": doc_type,
        "summary": simple_summary(text, doc_type),
        "overall_risk_level": overall_risk,
        "key_clauses": key_clauses,
        "audit": {
            "file_hash": file_hash(file_path),
            "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
            "user_id": user_id,
            "model_version": "v1"
        }
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="path to doc file (pdf/docx/txt)")
    parser.add_argument("--models_dir", default="models", help="models dir")
    parser.add_argument("--user", default="user_unknown")
    parser.add_argument("--output", default=None, help="optional output file (json)")
    args = parser.parse_args()

    res = analyze_document(args.file, args.models_dir, args.user)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
    else:
        import sys
        print(json.dumps(res, ensure_ascii=False, indent=2).encode(
            sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding))
