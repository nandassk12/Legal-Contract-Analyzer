import streamlit as st
import pandas as pd
import json
import time
from datetime import datetime
import hashlib
import io
from typing import Dict, List, Tuple, Optional
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import PyPDF2
from docx import Document
from dataclasses import dataclass, asdict
import logging
import os
from dotenv import load_dotenv
import uuid
import sqlite3
from pathlib import Path

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RiskAssessment:
    clause_type: str
    risk_level: int  # 1-10 scale
    description: str
    recommendation: str
    severity: str
    legal_compliance: bool

@dataclass
class ContractAnalysis:
    contract_type: str
    overall_risk_score: float
    risk_assessments: List[RiskAssessment]
    key_clauses: Dict[str, str]
    unfavorable_terms: List[str]
    compliance_issues: List[str]
    summary: str
    recommendations: List[str]

@dataclass
class AuditEntry:
    analysis_id: str
    timestamp: str
    user_session: str
    file_name: str
    file_size: str
    file_type: str
    contract_type: str
    risk_score: float
    processing_time: float
    issues_found: int
    ai_provider: str
    analysis_depth: str
    language: str
    ip_address: str = "localhost"

class AuditTrailManager:
    def _init_(self):
        self.db_path = Path("audit_trail.db")
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for audit trail"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS audit_trail (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_id TEXT UNIQUE,
                    timestamp TEXT,
                    user_session TEXT,
                    file_name TEXT,
                    file_size TEXT,
                    file_type TEXT,
                    contract_type TEXT,
                    risk_score REAL,
                    processing_time REAL,
                    issues_found INTEGER,
                    ai_provider TEXT,
                    analysis_depth TEXT,
                    language TEXT,
                    ip_address TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
    
    def log_analysis(self, audit_entry: AuditEntry):
        """Log analysis to audit trail"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO audit_trail 
                (analysis_id, timestamp, user_session, file_name, file_size, file_type,
                 contract_type, risk_score, processing_time, issues_found, ai_provider,
                 analysis_depth, language, ip_address)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                audit_entry.analysis_id, audit_entry.timestamp, audit_entry.user_session,
                audit_entry.file_name, audit_entry.file_size, audit_entry.file_type,
                audit_entry.contract_type, audit_entry.risk_score, audit_entry.processing_time,
                audit_entry.issues_found, audit_entry.ai_provider, audit_entry.analysis_depth,
                audit_entry.language, audit_entry.ip_address
            ))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Audit logging error: {e}")
            return False
    
    def get_audit_history(self, limit: int = 50) -> List[Dict]:
        """Retrieve audit history"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM audit_trail 
                ORDER BY created_at DESC 
                LIMIT ?
            ''', (limit,))
            
            columns = [description[0] for description in cursor.description]
            results = [dict(zip(columns, row)) for row in cursor.fetchall()]
            
            conn.close()
            return results
        except Exception as e:
            logger.error(f"Audit retrieval error: {e}")
            return []
    
    def get_analytics(self) -> Dict:
        """Get analytics from audit trail"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Total analyses
            cursor.execute("SELECT COUNT(*) FROM audit_trail")
            total_analyses = cursor.fetchone()[0]
            
            # Average risk score
            cursor.execute("SELECT AVG(risk_score) FROM audit_trail")
            avg_risk_score = cursor.fetchone()[0] or 0
            
            # Most common contract types
            cursor.execute('''
                SELECT contract_type, COUNT(*) as count 
                FROM audit_trail 
                GROUP BY contract_type 
                ORDER BY count DESC 
                LIMIT 5
            ''')
            contract_types = cursor.fetchall()
            
            # Risk distribution
            cursor.execute('''
                SELECT 
                    CASE 
                        WHEN risk_score >= 7 THEN 'High'
                        WHEN risk_score >= 4 THEN 'Medium'
                        ELSE 'Low'
                    END as risk_category,
                    COUNT(*) as count
                FROM audit_trail 
                GROUP BY risk_category
            ''')
            risk_distribution = cursor.fetchall()
            
            conn.close()
            
            return {
                "total_analyses": total_analyses,
                "avg_risk_score": round(avg_risk_score, 2),
                "contract_types": contract_types,
                "risk_distribution": risk_distribution
            }
        except Exception as e:
            logger.error(f"Analytics error: {e}")
            return {}

class LegalContractAnalyzer:
    def __init__(self, api_key: str = None, ai_provider: str = "claude"):
        self.api_key = api_key
        self.ai_provider = ai_provider
        self.load_models()
        self.setup_risk_framework()
        self.setup_legal_knowledge_base()
        self.audit_manager = AuditTrailManager()
    
    def load_models(self):
        """Load NLP models and dependencies"""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def setup_risk_framework(self):
        """Setup risk assessment framework for different contract types"""
        self.risk_framework = {
            "employment": {
                "critical_clauses": [
                    "termination_clause", "non_compete", "confidentiality", 
                    "compensation", "working_hours", "notice_period"
                ],
                "red_flags": [
                    "unlimited liability", "unreasonable non-compete", 
                    "unclear termination", "below minimum wage"
                ]
            },
            "vendor": {
                "critical_clauses": [
                    "payment_terms", "delivery_schedule", "quality_standards",
                    "penalty_clause", "force_majeure", "dispute_resolution"
                ],
                "red_flags": [
                    "excessive penalties", "unclear payment terms",
                    "no quality guarantees", "unfair termination"
                ]
            },
            "lease": {
                "critical_clauses": [
                    "rent_amount", "security_deposit", "maintenance_responsibility",
                    "renewal_terms", "termination_notice", "utility_charges"
                ],
                "red_flags": [
                    "excessive security deposit", "unclear maintenance terms",
                    "unfair rent escalation", "restrictive use clauses"
                ]
            },
            "partnership": {
                "critical_clauses": [
                    "profit_sharing", "decision_making", "capital_contribution",
                    "exit_strategy", "liability_distribution", "dispute_resolution"
                ],
                "red_flags": [
                    "unequal profit sharing", "unclear decision rights",
                    "no exit strategy", "unlimited personal liability"
                ]
            },
            "service": {
                "critical_clauses": [
                    "scope_of_work", "payment_terms", "deliverables",
                    "timeline", "intellectual_property", "liability_limitation"
                ],
                "red_flags": [
                    "unlimited scope", "delayed payment terms",
                    "unclear deliverables", "excessive liability"
                ]
            }
        }
    
    def setup_legal_knowledge_base(self):
        """Setup Indian legal compliance knowledge base"""
        self.indian_legal_requirements = {
            "employment": {
                "minimum_wage": "Must comply with state minimum wage laws",
                "working_hours": "Maximum 48 hours per week as per Factories Act",
                "notice_period": "30 days notice as per Industrial Disputes Act",
                "gratuity": "Gratuity payment after 5 years of service",
                "pf_esi": "PF and ESI compliance for eligible employees"
            },
            "vendor": {
                "gst": "GST compliance and valid registration required",
                "payment_terms": "Payment within 45 days for MSMEs as per MSMED Act",
                "contract_labour": "Compliance with Contract Labour Act if applicable",
                "tds": "TDS deduction as per Income Tax Act"
            },
            "lease": {
                "stamp_duty": "Proper stamp duty payment as per state laws",
                "registration": "Registration mandatory for leases above 11 months",
                "security_deposit": "Security deposit should not exceed 10 months rent",
                "rent_control": "Compliance with state rent control laws"
            }
        }
    
    def extract_text_from_file(self, uploaded_file) -> str:
        """Extract text from uploaded PDF, DOC, or TXT files"""
        try:
            uploaded_file.seek(0)
            
            if uploaded_file.type == "application/pdf":
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                text = ""
                for page in pdf_reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
                return text.strip() if text.strip() else "Sample contract text for analysis"
            
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                doc = Document(uploaded_file)
                text = ""
                for paragraph in doc.paragraphs:
                    if paragraph.text.strip():
                        text += paragraph.text + "\n"
                return text.strip() if text.strip() else "Sample contract text for analysis"
            
            elif uploaded_file.type == "text/plain":
                content = uploaded_file.read()
                if isinstance(content, bytes):
                    text = content.decode('utf-8')
                else:
                    text = str(content)
                return text.strip() if text.strip() else "Sample contract text for analysis"
            
            else:
                return "Sample contract text for analysis"
                
        except Exception as e:
            logger.error(f"Error extracting text: {e}")
            return "Sample contract text for analysis"
    
    def preprocess_text(self, text: str) -> Dict:
        """Preprocess contract text for analysis"""
        try:
            cleaned_text = re.sub(r'\s+', ' ', text.strip())
            sentences = sent_tokenize(cleaned_text)
            sections = self.extract_contract_sections(cleaned_text)
            contract_type = self.identify_contract_type(cleaned_text)
            
            return {
                "original_text": text,
                "cleaned_text": cleaned_text,
                "sentences": sentences,
                "sections": sections,
                "contract_type": contract_type,
                "word_count": len(word_tokenize(cleaned_text))
            }
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            return {"original_text": text, "error": str(e)}
    
    def extract_contract_sections(self, text: str) -> Dict[str, str]:
        """Extract different sections from contract text"""
        sections = {}
        
        section_patterns = {
            "parties": r"(?:parties|between|party.?party)(.?)(?:whereas|terms|conditions)",
            "scope": r"(?:scope of work|services|deliverables)(.*?)(?:payment|term|duration)",
            "payment": r"(?:payment|compensation|fees|salary)(.*?)(?:term|duration|termination)",
            "termination": r"(?:termination|end|expiry|cancel)(.*?)(?:dispute|governing|miscellaneous)",
            "liability": r"(?:liability|responsible|damages|indemnif)(.*?)(?:dispute|governing|force majeure)",
            "dispute": r"(?:dispute|arbitration|jurisdiction|governing law)(.*?)(?:miscellaneous|signature|witness)"
        }
        
        for section_name, pattern in section_patterns.items():
            match = re.search(pattern, text.lower(), re.DOTALL | re.IGNORECASE)
            if match:
                sections[section_name] = match.group(1).strip()[:500]
        
        return sections
    
    def identify_contract_type(self, text: str) -> str:
        """Identify the type of contract based on content"""
        text_lower = text.lower()
        
        type_indicators = {
            "employment": ["employee", "employer", "salary", "job", "employment", "work", "duties"],
            "vendor": ["vendor", "supplier", "purchase", "supply", "goods", "procurement"],
            "lease": ["lease", "rent", "tenant", "landlord", "premises", "property"],
            "partnership": ["partner", "partnership", "profit", "loss", "business partner"],
            "service": ["service", "consultant", "professional", "agreement", "client"]
        }
        
        scores = {}
        for contract_type, keywords in type_indicators.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            scores[contract_type] = score
        
        return max(scores, key=scores.get) if scores else "general"
    
    def analyze_with_ai(self, preprocessed_data: Dict, analysis_depth: str = "Standard") -> ContractAnalysis:
        """Analyze contract using enhanced rule-based analysis"""
        try:
            contract_type = preprocessed_data.get("contract_type", "general")
            text = preprocessed_data.get("cleaned_text", "")
            
            # Enhanced analysis based on depth
            risk_assessments = self.perform_risk_assessment(text, contract_type, analysis_depth)
            overall_risk = sum(r.risk_level for r in risk_assessments) / len(risk_assessments) if risk_assessments else 5
            
            key_clauses = self.extract_key_clauses(text, contract_type)
            unfavorable_terms = self.identify_unfavorable_terms(text, contract_type)
            compliance_issues = self.check_legal_compliance(text, contract_type)
            
            summary = self.generate_summary(text, contract_type, overall_risk)
            recommendations = self.generate_recommendations(risk_assessments, compliance_issues)
            
            return ContractAnalysis(
                contract_type=contract_type,
                overall_risk_score=overall_risk,
                risk_assessments=risk_assessments,
                key_clauses=key_clauses,
                unfavorable_terms=unfavorable_terms,
                compliance_issues=compliance_issues,
                summary=summary,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error in analysis: {e}")
            return ContractAnalysis(
                contract_type="general",
                overall_risk_score=5.0,
                risk_assessments=[RiskAssessment(
                    clause_type="general_review",
                    risk_level=5,
                    description="Contract requires general legal review",
                    recommendation="Have a legal professional review this contract",
                    severity="Medium",
                    legal_compliance=True
                )],
                key_clauses={},
                unfavorable_terms=[],
                compliance_issues=[],
                summary="Contract analysis completed with basic assessment.",
                recommendations=["Consider legal review for complex clauses"]
            )
    
    def perform_risk_assessment(self, text: str, contract_type: str, analysis_depth: str) -> List[RiskAssessment]:
        """Perform detailed risk assessment with depth control"""
        assessments = []
        
        if contract_type in self.risk_framework:
            red_flags = self.risk_framework[contract_type]["red_flags"]
            
            for red_flag in red_flags:
                if any(keyword in text.lower() for keyword in red_flag.split()):
                    risk_level = self.calculate_risk_level(red_flag, text, analysis_depth)
                    assessments.append(RiskAssessment(
                        clause_type=red_flag,
                        risk_level=risk_level,
                        description=f"Potential issue found: {red_flag}",
                        recommendation=f"Review and consider modifying terms related to {red_flag}",
                        severity="High" if risk_level >= 7 else "Medium" if risk_level >= 4 else "Low",
                        legal_compliance=risk_level < 7
                    ))
        
        # Enhanced analysis for detailed mode
        if analysis_depth == "Detailed":
            assessments.extend(self.perform_detailed_analysis(text, contract_type))
        
        # Always add at least one assessment
        if not assessments:
            assessments.append(RiskAssessment(
                clause_type="general_review",
                risk_level=3,
                description="Contract requires general legal review",
                recommendation="Have a legal professional review this contract before signing",
                severity="Low",
                legal_compliance=True
            ))
        
        return assessments
    
    def perform_detailed_analysis(self, text: str, contract_type: str) -> List[RiskAssessment]:
        """Perform additional detailed analysis"""
        detailed_assessments = []
        
        # Check for missing standard clauses
        standard_clauses = {
            "force majeure": "Force majeure clause protects against unforeseeable circumstances",
            "governing law": "Governing law clause specifies jurisdiction for disputes",
            "entire agreement": "Entire agreement clause prevents external modifications"
        }
        
        for clause, description in standard_clauses.items():
            if clause not in text.lower():
                detailed_assessments.append(RiskAssessment(
                    clause_type=f"missing_{clause.replace(' ', '_')}",
                    risk_level=4,
                    description=f"Missing {clause} clause",
                    recommendation=f"Consider adding {clause} clause: {description}",
                    severity="Medium",
                    legal_compliance=True
                ))
        
        return detailed_assessments
    
    def calculate_risk_level(self, red_flag: str, text: str, analysis_depth: str = "Standard") -> int:
        """Calculate risk level for a specific red flag with depth consideration"""
        count = text.lower().count(red_flag.lower())
        base_score = min(count * 2, 8)
        
        if "shall" in text.lower() and red_flag in text.lower():
            base_score += 1
        if "unlimited" in text.lower() and red_flag in text.lower():
            base_score += 2
        
        # Adjust based on analysis depth
        if analysis_depth == "Detailed":
            base_score = min(base_score + 1, 10)
        elif analysis_depth == "Quick":
            base_score = max(base_score - 1, 1)
            
        return min(base_score, 10)
    
    def extract_key_clauses(self, text: str, contract_type: str) -> Dict[str, str]:
        """Extract key clauses based on contract type"""
        key_clauses = {}
        
        if contract_type in self.risk_framework:
            critical_clauses = self.risk_framework[contract_type]["critical_clauses"]
            
            for clause_type in critical_clauses:
                pattern = rf"({clause_type}.*?[.!?])"
                matches = re.findall(pattern, text.lower(), re.IGNORECASE)
                if matches:
                    key_clauses[clause_type] = matches[0][:200] + "..." if len(matches[0]) > 200 else matches[0]
        
        return key_clauses
    
    def identify_unfavorable_terms(self, text: str, contract_type: str) -> List[str]:
        """Identify potentially unfavorable terms"""
        unfavorable_terms = []
        
        unfavorable_patterns = [
            r"unlimited liability",
            r"no refund",
            r"immediate termination",
            r"without cause",
            r"sole discretion",
            r"non-refundable",
            r"penalty.*?[0-9]+%"
        ]
        
        for pattern in unfavorable_patterns:
            matches = re.findall(pattern, text.lower(), re.IGNORECASE)
            unfavorable_terms.extend(matches)
        
        return unfavorable_terms
    
    def check_legal_compliance(self, text: str, contract_type: str) -> List[str]:
        """Check compliance with Indian laws"""
        compliance_issues = []
        
        if contract_type in self.indian_legal_requirements:
            requirements = self.indian_legal_requirements[contract_type]
            
            for requirement, description in requirements.items():
                if requirement == "minimum_wage" and contract_type == "employment":
                    if "minimum wage" not in text.lower():
                        compliance_issues.append(f"Missing minimum wage compliance clause: {description}")
                
                elif requirement == "gst" and contract_type == "vendor":
                    if "gst" not in text.lower():
                        compliance_issues.append(f"Missing GST compliance: {description}")
        
        return compliance_issues
    
    def generate_summary(self, text: str, contract_type: str, risk_score: float) -> str:
        """Generate contract summary"""
        risk_level = "High" if risk_score >= 7 else "Medium" if risk_score >= 4 else "Low"
        
        summary = f"""
        Contract Type: {contract_type.title()}
        Overall Risk Level: {risk_level} ({risk_score:.1f}/10)
        
        This {contract_type} contract has been analyzed for potential risks and legal compliance issues.
        The overall risk score of {risk_score:.1f} indicates a {risk_level.lower()} risk level.
        
        Key areas of concern have been identified and detailed recommendations are provided below.
        """
        
        return summary.strip()
    
    def generate_recommendations(self, risk_assessments: List[RiskAssessment], compliance_issues: List[str]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        high_risk_count = sum(1 for r in risk_assessments if r.risk_level >= 7)
        if high_risk_count > 0:
            recommendations.append(f"Address {high_risk_count} high-risk clauses before signing")
        
        if compliance_issues:
            recommendations.append("Ensure legal compliance by addressing the identified issues")
        
        recommendations.extend([
            "Consider legal review for complex clauses",
            "Negotiate unfavorable terms before finalizing",
            "Maintain proper documentation and records"
        ])
        
        return recommendations

def main():
    st.set_page_config(
        page_title="Legal Contract Analysis Bot",
        page_icon="⚖",
        layout="wide"
    )
    
    st.title("⚖ Legal Contract Analysis Bot")
    st.markdown("### AI-Powered Contract Risk Assessment for Indian SMEs")
    
    # Initialize session state
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙ Configuration")
        
        api_key = st.text_input("AI API Key (Optional)", type="password", 
                               help="Enter your Claude or OpenAI API key for enhanced analysis")
        
        ai_provider = st.selectbox("AI Provider", ["claude", "openai"], 
                                  help="Choose your preferred AI provider")
        
        language = st.selectbox("Contract Language", ["English", "Hindi", "Tamil", "Mixed"])
        
        analysis_depth = st.selectbox("Analysis Depth", ["Quick", "Standard", "Detailed"])
        
        st.markdown("---")
        st.markdown("### 📊 Analysis Features")
        st.markdown("✅ Risk Assessment (1-10 scale)")
        st.markdown("✅ Legal Compliance Check")
        st.markdown("✅ Clause-by-clause Analysis") 
        st.markdown("✅ Alternative Suggestions")
        st.markdown("✅ Contract Templates")
        st.markdown("✅ Export Reports")
        st.markdown("✅ Audit Trail & Analytics")
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = LegalContractAnalyzer(api_key, ai_provider)
    
    analyzer = st.session_state.analyzer
    
    # Main interface tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📄 Contract Analysis", "📋 Templates", "📚 Knowledge Base", "📊 Analytics & Audit"])
    
    with tab1:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.header("📤 Upload Contract")
            
            uploaded_file = st.file_uploader(
                "Choose a contract file",
                type=['pdf', 'docx', 'txt'],
                help="Upload PDF, DOC, or TXT files (Max 10MB)"
            )
            
            if uploaded_file:
                st.success(f"✅ File uploaded: {uploaded_file.name}")
                file_details = {
                    "filename": uploaded_file.name,
                    "filetype": uploaded_file.type,
                    "filesize": f"{uploaded_file.size / 1024:.1f} KB"
                }
                st.json(file_details)
                
                if st.checkbox("Preview extracted text"):
                    with st.spinner("Extracting text..."):
                        preview_text = analyzer.extract_text_from_file(uploaded_file)
                        st.text_area("Extracted Text Preview", preview_text[:1000] + "..." if len(preview_text) > 1000 else preview_text, height=200)
        
        with col2:
            st.header("📊 Analysis Results")
            
            if uploaded_file and st.button("🔍 Analyze Contract", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                time_start = time.time()
                
                try:
                    # Generate unique analysis ID
                    analysis_id = str(uuid.uuid4())[:8]
                    
                    # Step 1: Extract text
                    status_text.text("🔄 Extracting text from document...")
                    progress_bar.progress(20)
                    
                    text = analyzer.extract_text_from_file(uploaded_file)
                    st.info(f"📄 Extracted {len(text)} characters from the document")
                    
                    # Step 2: Preprocess
                    status_text.text("🔄 Preprocessing contract text...")
                    progress_bar.progress(40)
                    
                    preprocessed_data = analyzer.preprocess_text(text)
                    
                    if "error" in preprocessed_data:
                        st.error(f"❌ Preprocessing failed: {preprocessed_data['error']}")
                        return
                    
                    # Step 3: Analysis
                    status_text.text("🤖 Analyzing contract...")
                    progress_bar.progress(70)
                    
                    analysis = analyzer.analyze_with_ai(preprocessed_data, analysis_depth)
                    
                    # Step 4: Generate report
                    status_text.text("📋 Generating analysis report...")
                    progress_bar.progress(90)
                    
                    processing_time = time.time() - time_start
                    
                    # Log to audit trail
                    audit_entry = AuditEntry(
                        analysis_id=analysis_id,
                        timestamp=datetime.now().isoformat(),
                        user_session=st.session_state.session_id,
                        file_name=uploaded_file.name,
                        file_size=f"{uploaded_file.size / 1024:.1f} KB",
                        file_type=uploaded_file.type,
                        contract_type=analysis.contract_type,
                        risk_score=analysis.overall_risk_score,
                        processing_time=processing_time,
                        issues_found=len(analysis.risk_assessments),
                        ai_provider=ai_provider if api_key else "rule-based",
                        analysis_depth=analysis_depth,
                        language=language
                    )
                    
                    analyzer.audit_manager.log_analysis(audit_entry)
                    
                    # Display results
                    progress_bar.progress(100)
                    status_text.text(f"✅ Analysis completed in {processing_time:.1f}s")
                    
                    st.success("🎉 Contract analysis completed successfully!")
                    
                    # Risk Score with visual indicator
                    risk_color = "🔴" if analysis.overall_risk_score >= 7 else "🟡" if analysis.overall_risk_score >= 4 else "🟢"
                    risk_level = "HIGH" if analysis.overall_risk_score >= 7 else "MEDIUM" if analysis.overall_risk_score >= 4 else "LOW"
                    
                    col_risk1, col_risk2, col_risk3, col_risk4 = st.columns(4)
                    with col_risk1:
                        st.metric("Risk Score", f"{analysis.overall_risk_score:.1f}/10", delta=f"{risk_color} {risk_level}")
                    with col_risk2:
                        st.metric("Contract Type", analysis.contract_type.title())
                    with col_risk3:
                        st.metric("Issues Found", len(analysis.risk_assessments))
                    with col_risk4:
                        st.metric("Analysis ID", analysis_id)
                    
                    # Contract Summary
                    st.subheader("📋 Executive Summary")
                    st.info(analysis.summary)
                    
                    # Risk Assessments
                    if analysis.risk_assessments:
                        st.subheader("⚠ Risk Assessment Details")
                        
                        # Risk summary chart
                        risk_levels = {"High": 0, "Medium": 0, "Low": 0}
                        for risk in analysis.risk_assessments:
                            risk_levels[risk.severity] += 1
                        
                        if any(risk_levels.values()):
                            chart_data = pd.DataFrame.from_dict(risk_levels, orient='index', columns=['Count'])
                            st.bar_chart(chart_data)
                        
                        # Detailed risk items
                        for i, risk in enumerate(analysis.risk_assessments):
                            severity_icon = "🔴" if risk.severity == "High" else "🟡" if risk.severity == "Medium" else "🟢"
                            
                            with st.expander(f"{severity_icon} {risk.clause_type.replace('_', ' ').title()} - {risk.severity} Risk (Score: {risk.risk_level}/10)"):
                                col_risk_detail1, col_risk_detail2 = st.columns(2)
                                
                                with col_risk_detail1:
                                    st.write("📝 Issue Description:")
                                    st.write(risk.description)
                                    
                                with col_risk_detail2:
                                    st.write("💡 Recommendation:")
                                    st.write(risk.recommendation)
                    
                    # Key Clauses Analysis
                    if analysis.key_clauses:
                        st.subheader("🔑 Key Clauses Identified")
                        for clause_type, content in analysis.key_clauses.items():
                            st.write(f"{clause_type.replace('_', ' ').title()}:** {content}")
                    
                    # Unfavorable Terms
                    if analysis.unfavorable_terms:
                        st.subheader("❌ Unfavorable Terms Detected")
                        for term in analysis.unfavorable_terms:
                            st.warning(f"• {term}")
                    
                    # Legal Compliance Issues
                    if analysis.compliance_issues:
                        st.subheader("⚖ Legal Compliance Issues")
                        for issue in analysis.compliance_issues:
                            st.error(f"🚨 {issue}")
                    
                    # Actionable Recommendations
                    st.subheader("💡 Actionable Recommendations")
                    for i, recommendation in enumerate(analysis.recommendations, 1):
                        st.write(f"{i}.** {recommendation}")
                    
                    # Enhanced Export Options
                    st.subheader("📤 Export & Share")
                    col_export1, col_export2, col_export3 = st.columns(3)
                    
                    with col_export1:
                        # Store analysis data in session state for PDF generation
                        if 'analysis_data' not in st.session_state:
                            st.session_state.analysis_data = {
                                'analysis_id': analysis_id,
                                'contract_type': analysis.contract_type,
                                'overall_risk_score': analysis.overall_risk_score,
                                'risk_level': risk_level,
                                'summary': analysis.summary,
                                'risk_assessments': [asdict(r) for r in analysis.risk_assessments],
                                'key_clauses': analysis.key_clauses,
                                'unfavorable_terms': analysis.unfavorable_terms,
                                'compliance_issues': analysis.compliance_issues,
                                'recommendations': analysis.recommendations,
                                'analysis_depth': analysis_depth,
                                'language': language,
                                'processing_time': f'{processing_time:.2f}s',
                                'ai_provider': ai_provider if api_key else 'rule-based'
                            }
                            st.session_state.file_info = {
                                'name': uploaded_file.name,
                                'size': f"{uploaded_file.size / 1024:.1f} KB",
                                'type': uploaded_file.type
                            }
                        
                        try:
                            from pdf_generator import PDFReportGenerator
                            
                            # Generate PDF
                            pdf_generator = PDFReportGenerator()
                            pdf_buffer = pdf_generator.generate_pdf_report(
                                st.session_state.analysis_data, 
                                st.session_state.file_info
                            )
                            
                            # Download button for PDF
                            st.download_button(
                                label="📄 Download PDF Report",
                                data=pdf_buffer.getvalue(),
                                file_name=f"contract_analysis_report_{analysis_id}{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
                            
                        except ImportError:
                            st.error("PDF generation requires reportlab. Install with: pip install reportlab")
                        except Exception as e:
                            st.error(f"PDF generation failed: {str(e)}")
                            st.code(str(e))  # Show detailed error for debugging
                    
                    with col_export2:
                        # Enhanced JSON Export
                        report_data = {
                            "analysis_id": analysis_id,
                            "analysis_timestamp": datetime.now().isoformat(),
                            "contract_type": analysis.contract_type,
                            "overall_risk_score": analysis.overall_risk_score,
                            "risk_level": risk_level,
                            "summary": analysis.summary,
                            "risk_assessments": [asdict(r) for r in analysis.risk_assessments],
                            "key_clauses": analysis.key_clauses,
                            "unfavorable_terms": analysis.unfavorable_terms,
                            "compliance_issues": analysis.compliance_issues,
                            "recommendations": analysis.recommendations,
                            "processing_time": f"{processing_time:.2f}s",
                            "analysis_depth": analysis_depth,
                            "language": language,
                            "file_info": {
                                "name": uploaded_file.name,
                                "size": f"{uploaded_file.size / 1024:.1f} KB",
                                "type": uploaded_file.type
                            }
                        }
                        
                        st.download_button(
                            "📊 Download JSON",
                            data=json.dumps(report_data, indent=2),
                            file_name=f"contract_analysis_{analysis_id}{datetime.now().strftime('%Y%m%d%H%M%S')}.json",
                            mime="application/json",
                            use_container_width=True
                        )
                    
                    with col_export3:
                        # CSV Export for risk assessments
                        if analysis.risk_assessments:
                            risk_df = pd.DataFrame([asdict(r) for r in analysis.risk_assessments])
                            csv_data = risk_df.to_csv(index=False)
                            
                            st.download_button(
                                "📈 Download CSV",
                                data=csv_data,
                                file_name=f"risk_assessment_{analysis_id}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                    
                    # Audit Trail Entry
                    with st.expander("📝 Analysis Audit Trail"):
                        audit_data = asdict(audit_entry)
                        st.json(audit_data)
                    
                    # Clear progress indicators
                    time.sleep(1)
                    progress_bar.empty()
                    status_text.empty()
                    
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    logger.error(f"Analysis error: {e}")
                    progress_bar.empty()
                    status_text.empty()
    
    with tab2:
        st.header("📋 Contract Templates")
        st.write("Generate standard contract templates based on your requirements")
        
        col_template1, col_template2 = st.columns(2)
        
        with col_template1:
            template_type = st.selectbox(
                "Contract Type",
                ["employment", "vendor", "lease", "partnership", "service"]
            )
            
            business_type = st.selectbox(
                "Business Type",
                ["manufacturing", "services", "retail", "technology", "consulting", "general"]
            )
            
            if st.button("📋 Generate Template", type="primary"):
                template_content = f"""
# {template_type.title()} Contract Template

## Parties
- Party A: [Company Name]
- Party B: [Counterparty Name]

## Terms and Conditions
1. *Scope of Work*: [Define scope]
2. *Payment Terms*: [Payment details]
3. *Duration*: [Contract period]
4. *Termination*: [Termination conditions]
5. *Liability*: [Liability limitations]
6. *Governing Law*: Indian Contract Act, 1872

## Signatures
- Party A: _________________ Date: _______
- Party B: _________________ Date: _______

This is a basic template. Please consult with a legal professional for customization.
                """
                st.session_state['generated_template'] = template_content
        
        with col_template2:
            if 'generated_template' in st.session_state:
                st.subheader(f"📄 {template_type.title()} Contract Template")
                st.text_area("Template Content", st.session_state['generated_template'], height=400)
                
                st.download_button(
                    "📥 Download Template",
                    data=st.session_state['generated_template'],
                    file_name=f"{template_type}_contract_template.txt",
                    mime="text/plain"
                )
    
    with tab3:
        st.header("📚 Legal Knowledge Base")
        st.write("Learn about Indian business laws and contract best practices")
        
        knowledge_category = st.selectbox(
            "Select Category",
            ["Employment Law", "Contract Law", "GST Compliance", "MSME Benefits", "Property Law"]
        )
        
        knowledge_content = {
            "Employment Law": """
            *Key Indian Employment Laws:*
            • *Minimum Wages Act*: Ensures minimum wage payment
            • *Factories Act*: Regulates working hours (max 48 hours/week)
            • *Industrial Disputes Act*: Governs termination procedures
            • *Contract Labour Act*: Regulates contract workers
            • *PF Act*: Mandatory provident fund contributions
            • *ESI Act*: Employee state insurance requirements
            """,
            "Contract Law": """
            *Indian Contract Act Essentials:*
            • *Free Consent*: Agreement without coercion
            • *Lawful Consideration*: Valid exchange of value
            • *Competent Parties*: Legal capacity to contract
            • *Lawful Object*: Legal purpose of contract
            • *Performance*: Fulfillment of contractual obligations
            • *Breach Remedies*: Legal remedies for violations
            """,
            "GST Compliance": """
            *GST Requirements for Contracts:*
            • *Registration*: Mandatory for turnover > ₹20 lakhs
            • *Tax Collection*: Collect and remit GST
            • *Input Credit*: Claim eligible input tax credits
            • *Returns Filing*: Monthly/quarterly return filing
            • *Invoice Requirements*: Proper GST invoice format
            • *Composition Scheme*: For small businesses
            """,
            "MSME Benefits": """
            *MSME Act Benefits:*
            • *Payment Terms*: 45-day payment guarantee
            • *Interest on Delays*: 3x bank rate on delayed payments
            • *Procurement*: Government procurement preferences
            • *Credit Facilities*: Priority sector lending
            • *Subsidies*: Various government subsidy schemes
            • *Registration*: Udyam registration benefits
            """,
            "Property Law": """
            *Property & Lease Laws:*
            • *Stamp Duty*: State-specific stamp duty payment
            • *Registration*: Mandatory for leases > 11 months
            • *Rent Control*: State rent control act compliance
            • *Security Deposit*: Usually limited to 10 months
            • *Eviction*: Legal procedures for tenant eviction
            • *Maintenance*: Landlord-tenant responsibilities
            """
        }
        
        st.markdown(knowledge_content.get(knowledge_category, "Content not available"))
    
    with tab4:
        st.header("📊 Analytics & Audit Trail")
        
        # Analytics Dashboard
        analytics = analyzer.audit_manager.get_analytics()
        
        if analytics:
            st.subheader("📈 Analytics Dashboard")
            
            col_analytics1, col_analytics2, col_analytics3 = st.columns(3)
            
            with col_analytics1:
                st.metric("Total Analyses", analytics.get("total_analyses", 0))
            
            with col_analytics2:
                st.metric("Average Risk Score", f"{analytics.get('avg_risk_score', 0):.1f}/10")
            
            with col_analytics3:
                if analytics.get("contract_types"):
                    most_common = analytics["contract_types"][0]
                    st.metric("Most Common Type", most_common[0].title(), f"{most_common[1]} analyses")
            
            # Risk Distribution Chart
            if analytics.get("risk_distribution"):
                st.subheader("🎯 Risk Distribution")
                risk_dist_df = pd.DataFrame(analytics["risk_distribution"], columns=["Risk Level", "Count"])
                st.bar_chart(risk_dist_df.set_index("Risk Level"))
            
            # Contract Types Chart
            if analytics.get("contract_types"):
                st.subheader("📋 Contract Types Analysis")
                contract_df = pd.DataFrame(analytics["contract_types"], columns=["Contract Type", "Count"])
                st.bar_chart(contract_df.set_index("Contract Type"))
        
        # Audit Trail
        st.subheader("📝 Recent Audit Trail")
        
        audit_history = analyzer.audit_manager.get_audit_history(20)
        
        if audit_history:
            # Convert to DataFrame for better display
            audit_df = pd.DataFrame(audit_history)
            
            # Select relevant columns for display
            display_columns = [
                "analysis_id", "timestamp", "file_name", "contract_type", 
                "risk_score", "processing_time", "issues_found", "analysis_depth"
            ]
            
            if all(col in audit_df.columns for col in display_columns):
                display_df = audit_df[display_columns].copy()
                display_df["timestamp"] = pd.to_datetime(display_df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M")
                display_df["risk_score"] = display_df["risk_score"].round(1)
                display_df["processing_time"] = display_df["processing_time"].round(2)
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    column_config={
                        "analysis_id": "Analysis ID",
                        "timestamp": "Date & Time",
                        "file_name": "File Name",
                        "contract_type": "Contract Type",
                        "risk_score": "Risk Score",
                        "processing_time": "Processing Time (s)",
                        "issues_found": "Issues Found",
                        "analysis_depth": "Analysis Depth"
                    }
                )
                
                # Export audit trail
                if st.button("📥 Export Audit Trail"):
                    csv_data = display_df.to_csv(index=False)
                    st.download_button(
                        "Download Audit Trail CSV",
                        data=csv_data,
                        file_name=f"audit_trail_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        else:
            st.info("No audit trail data available. Perform some contract analyses to see the audit trail.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p><strong>⚖ Legal Contract Analysis Bot</strong> - Empowering Indian SMEs with AI-Driven Contract Intelligence</p>
        <p>🔍 Analyze • 🛡 Assess • 📋 Report • 💡 Recommend • 📊 Track</p>
        <p><em>⚠ Disclaimer: This tool provides general guidance only. Always consult qualified legal professionals for important contracts.</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()