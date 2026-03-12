# Legal Contract Analyzer

An AI-powered contract analysis system that extracts important clauses, detects potential legal risks, checks compliance with Indian regulations, and generates structured risk reports from legal documents.

## Overview

The Legal Contract Analyzer helps users quickly understand complex contracts by automatically identifying key clauses and highlighting risky or non-compliant terms. The system processes contracts in multiple formats and provides structured insights for faster legal review.

## Features

- Contract text extraction from **PDF, DOCX, and TXT files**
- **Clause-level analysis** for important legal provisions
- **Risk detection** for potentially unfair or ambiguous clauses
- **Compliance checks** based on Indian legal regulations
- **Automated contract summarization**
- **Risk report generation** in multiple formats
- **Audit trail and analytics dashboard**
- Interactive **Streamlit web interface**

## Workflow

1. Upload a contract document.
2. Extract and preprocess contract text.
3. Detect contract type and identify key clauses.
4. Analyze clauses to detect legal risks.
5. Perform compliance checks against regulatory rules.
6. Generate a summary and structured risk report.

## Tech Stack

**Backend**
- Python
- NLTK
- Regex-based NLP processing

**Frontend**
- Streamlit

**Data Processing**
- NumPy
- Pandas

**Document Processing**
- PyPDF2
- python-docx
- ReportLab

**Database**
- SQLite (Audit Trail)

## Supported File Formats

- PDF
- DOCX
- TXT

## How to Run

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Start the application:

```
streamlit run legalapp.py
```

3. Open the application in your browser and upload a contract file.

## Future Improvements

- Multilingual contract analysis
- Machine learning–based clause classification
- Larger legal datasets for model training
- Integration with legal document management systems

## Disclaimer

This tool provides automated contract insights for educational and research purposes. It should not replace professional legal advice.
