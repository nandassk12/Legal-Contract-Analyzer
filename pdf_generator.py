from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from datetime import datetime
import io

class PDFReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
    
    def setup_custom_styles(self):
        """Setup custom styles for the PDF report"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        ))
        
        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='CustomSubtitle',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=20,
            alignment=TA_CENTER,
            textColor=colors.darkgreen
        ))
        
        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceBefore=20,
            spaceAfter=10,
            textColor=colors.darkblue,
            borderWidth=1,
            borderColor=colors.darkblue,
            borderPadding=5
        ))
        
        # Risk high style
        self.styles.add(ParagraphStyle(
            name='RiskHigh',
            parent=self.styles['Normal'],
            fontSize=11,
            textColor=colors.red,
            leftIndent=20
        ))
        
        # Risk medium style
        self.styles.add(ParagraphStyle(
            name='RiskMedium',
            parent=self.styles['Normal'],
            fontSize=11,
            textColor=colors.orange,
            leftIndent=20
        ))
        
        # Risk low style
        self.styles.add(ParagraphStyle(
            name='RiskLow',
            parent=self.styles['Normal'],
            fontSize=11,
            textColor=colors.green,
            leftIndent=20
        ))
    
    def generate_pdf_report(self, analysis_data, file_info):
        """Generate a comprehensive PDF report"""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
        
        # Container for the 'Flowable' objects
        elements = []
        
        # Title Page
        elements.extend(self._create_title_page(analysis_data, file_info))
        elements.append(PageBreak())
        
        # Executive Summary
        elements.extend(self._create_executive_summary(analysis_data))
        
        # Risk Assessment Details
        elements.extend(self._create_risk_assessment_section(analysis_data))
        
        # Key Clauses Analysis
        if analysis_data.get('key_clauses'):
            elements.extend(self._create_key_clauses_section(analysis_data))
        
        # Compliance Issues
        if analysis_data.get('compliance_issues'):
            elements.extend(self._create_compliance_section(analysis_data))
        
        # Recommendations
        elements.extend(self._create_recommendations_section(analysis_data))
        
        # Appendix
        elements.extend(self._create_appendix(analysis_data, file_info))
        
        # Build PDF
        doc.build(elements)
        buffer.seek(0)
        return buffer
    
    def _create_title_page(self, analysis_data, file_info):
        """Create the title page"""
        elements = []
        
        # Main title
        elements.append(Paragraph("CONTRACT RISK ANALYSIS REPORT", self.styles['CustomTitle']))
        elements.append(Spacer(1, 30))
        
        # Subtitle
        elements.append(Paragraph("AI-Powered Legal Contract Assessment", self.styles['CustomSubtitle']))
        elements.append(Spacer(1, 50))
        
        # Contract details table
        contract_data = [
            ['Contract File:', file_info.get('name', 'N/A')],
            ['File Size:', file_info.get('size', 'N/A')],
            ['Contract Type:', analysis_data.get('contract_type', 'N/A').title()],
            ['Analysis Date:', datetime.now().strftime('%B %d, %Y')],
            ['Analysis Time:', datetime.now().strftime('%I:%M %p')],
            ['Analysis ID:', analysis_data.get('analysis_id', 'N/A')],
            ['Overall Risk Score:', f"{analysis_data.get('overall_risk_score', 0):.1f}/10"],
            ['Risk Level:', analysis_data.get('risk_level', 'N/A')]
        ]
        
        contract_table = Table(contract_data, colWidths=[2*inch, 3*inch])
        contract_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('BACKGROUND', (1, 0), (1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(contract_table)
        elements.append(Spacer(1, 50))
        
        # Disclaimer
        disclaimer = """
        <b>IMPORTANT DISCLAIMER:</b><br/>
        This automated analysis is provided for informational purposes only and should not be considered as legal advice. 
        The assessment is based on pattern recognition and rule-based analysis. For critical business decisions, 
        please consult with qualified legal professionals who can provide comprehensive legal review and advice 
        specific to your jurisdiction and circumstances.
        """
        elements.append(Paragraph(disclaimer, self.styles['Normal']))
        
        return elements
    
    def _create_executive_summary(self, analysis_data):
        """Create executive summary section"""
        elements = []
        
        elements.append(Paragraph("EXECUTIVE SUMMARY", self.styles['SectionHeader']))
        elements.append(Spacer(1, 12))
        
        # Summary text
        summary_text = analysis_data.get('summary', 'No summary available.')
        elements.append(Paragraph(summary_text, self.styles['Normal']))
        elements.append(Spacer(1, 20))
        
        # Key metrics table
        risk_score = analysis_data.get('overall_risk_score', 0)
        risk_level = analysis_data.get('risk_level', 'Unknown')
        issues_count = len(analysis_data.get('risk_assessments', []))
        
        metrics_data = [
            ['Metric', 'Value', 'Status'],
            ['Overall Risk Score', f'{risk_score:.1f}/10', self._get_risk_status(risk_score)],
            ['Risk Classification', risk_level, self._get_risk_color(risk_level)],
            ['Issues Identified', str(issues_count), 'Review Required' if issues_count > 0 else 'Clean'],
            ['Contract Type', analysis_data.get('contract_type', 'N/A').title(), 'Identified']
        ]
        
        metrics_table = Table(metrics_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(metrics_table)
        elements.append(Spacer(1, 20))
        
        return elements
    
    def _create_risk_assessment_section(self, analysis_data):
        """Create risk assessment details section"""
        elements = []
        
        elements.append(Paragraph("RISK ASSESSMENT DETAILS", self.styles['SectionHeader']))
        elements.append(Spacer(1, 12))
        
        risk_assessments = analysis_data.get('risk_assessments', [])
        
        if not risk_assessments:
            elements.append(Paragraph("No specific risks identified in the contract.", self.styles['Normal']))
            return elements
        
        # Risk summary
        high_risks = [r for r in risk_assessments if r.get('severity') == 'High']
        medium_risks = [r for r in risk_assessments if r.get('severity') == 'Medium']
        low_risks = [r for r in risk_assessments if r.get('severity') == 'Low']
        
        summary_text = f"""
        <b>Risk Distribution:</b><br/>
        • High Risk Issues: {len(high_risks)}<br/>
        • Medium Risk Issues: {len(medium_risks)}<br/>
        • Low Risk Issues: {len(low_risks)}<br/>
        """
        elements.append(Paragraph(summary_text, self.styles['Normal']))
        elements.append(Spacer(1, 15))
        
        # Detailed risk items
        for i, risk in enumerate(risk_assessments, 1):
            risk_style = self._get_risk_style(risk.get('severity', 'Low'))
            
            elements.append(Paragraph(f"<b>{i}. {risk.get('clause_type', 'Unknown').replace('_', ' ').title()}</b>", 
                                    self.styles['Heading3']))
            
            elements.append(Paragraph(f"<b>Risk Level:</b> {risk.get('severity', 'Unknown')} ({risk.get('risk_level', 0)}/10)", 
                                    risk_style))
            
            elements.append(Paragraph(f"<b>Description:</b> {risk.get('description', 'No description available.')}", 
                                    self.styles['Normal']))
            
            elements.append(Paragraph(f"<b>Recommendation:</b> {risk.get('recommendation', 'No recommendation available.')}", 
                                    self.styles['Normal']))
            
            elements.append(Paragraph(f"<b>Legal Compliance:</b> {'Compliant' if risk.get('legal_compliance', True) else 'Non-Compliant'}", 
                                    self.styles['Normal']))
            
            elements.append(Spacer(1, 15))
        
        return elements
    
    def _create_key_clauses_section(self, analysis_data):
        """Create key clauses analysis section"""
        elements = []
        
        elements.append(Paragraph("KEY CLAUSES ANALYSIS", self.styles['SectionHeader']))
        elements.append(Spacer(1, 12))
        
        key_clauses = analysis_data.get('key_clauses', {})
        
        for clause_type, content in key_clauses.items():
            elements.append(Paragraph(f"<b>{clause_type.replace('_', ' ').title()}:</b>", 
                                    self.styles['Heading4']))
            elements.append(Paragraph(content, self.styles['Normal']))
            elements.append(Spacer(1, 10))
        
        return elements
    
    def _create_compliance_section(self, analysis_data):
        """Create legal compliance section"""
        elements = []
        
        elements.append(Paragraph("LEGAL COMPLIANCE ISSUES", self.styles['SectionHeader']))
        elements.append(Spacer(1, 12))
        
        compliance_issues = analysis_data.get('compliance_issues', [])
        
        if not compliance_issues:
            elements.append(Paragraph("No compliance issues identified.", self.styles['Normal']))
            return elements
        
        for i, issue in enumerate(compliance_issues, 1):
            elements.append(Paragraph(f"<b>{i}.</b> {issue}", self.styles['RiskHigh']))
            elements.append(Spacer(1, 8))
        
        return elements
    
    def _create_recommendations_section(self, analysis_data):
        """Create recommendations section"""
        elements = []
        
        elements.append(Paragraph("ACTIONABLE RECOMMENDATIONS", self.styles['SectionHeader']))
        elements.append(Spacer(1, 12))
        
        recommendations = analysis_data.get('recommendations', [])
        
        for i, recommendation in enumerate(recommendations, 1):
            elements.append(Paragraph(f"<b>{i}.</b> {recommendation}", self.styles['Normal']))
            elements.append(Spacer(1, 8))
        
        elements.append(Spacer(1, 20))
        
        # Next steps
        next_steps = """
        <b>RECOMMENDED NEXT STEPS:</b><br/>
        1. Review all high-risk items identified in this report<br/>
        2. Consult with legal counsel for complex clauses<br/>
        3. Negotiate unfavorable terms before contract execution<br/>
        4. Ensure all compliance requirements are met<br/>
        5. Maintain proper documentation and records<br/>
        6. Consider periodic contract reviews for ongoing agreements
        """
        elements.append(Paragraph(next_steps, self.styles['Normal']))
        
        return elements
    
    def _create_appendix(self, analysis_data, file_info):
        """Create appendix section"""
        elements = []
        
        elements.append(PageBreak())
        elements.append(Paragraph("APPENDIX", self.styles['SectionHeader']))
        elements.append(Spacer(1, 12))
        
        # Technical details
        tech_details = f"""
        <b>TECHNICAL ANALYSIS DETAILS</b><br/><br/>
        <b>Analysis Parameters:</b><br/>
        • Analysis Depth: {analysis_data.get('analysis_depth', 'Standard')}<br/>
        • Language: {analysis_data.get('language', 'English')}<br/>
        • Processing Time: {analysis_data.get('processing_time', 'N/A')}<br/>
        • AI Provider: {analysis_data.get('ai_provider', 'Rule-based')}<br/><br/>
        
        <b>File Information:</b><br/>
        • Original Filename: {file_info.get('name', 'N/A')}<br/>
        • File Type: {file_info.get('type', 'N/A')}<br/>
        • File Size: {file_info.get('size', 'N/A')}<br/><br/>
        
        <b>Analysis Methodology:</b><br/>
        This analysis was conducted using a combination of natural language processing, 
        pattern recognition, and rule-based legal assessment techniques. The system evaluates 
        contracts against established risk frameworks and Indian legal compliance requirements.
        """
        
        elements.append(Paragraph(tech_details, self.styles['Normal']))
        elements.append(Spacer(1, 20))
        
        # Footer
        footer_text = f"""
        <b>Report Generated:</b> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}<br/>
        <b>Analysis ID:</b> {analysis_data.get('analysis_id', 'N/A')}<br/>
        <b>System Version:</b> Legal Contract Analysis Bot v2.0<br/><br/>
        
        <i>This report is confidential and intended solely for the use of the recipient. 
        Distribution or reproduction without authorization is prohibited.</i>
        """
        
        elements.append(Paragraph(footer_text, self.styles['Normal']))
        
        return elements
    
    def _get_risk_status(self, risk_score):
        """Get risk status based on score"""
        if risk_score >= 7:
            return "HIGH RISK"
        elif risk_score >= 4:
            return "MEDIUM RISK"
        else:
            return "LOW RISK"
    
    def _get_risk_color(self, risk_level):
        """Get risk color indicator"""
        if risk_level == "HIGH":
            return "🔴 Critical"
        elif risk_level == "MEDIUM":
            return "🟡 Caution"
        else:
            return "🟢 Acceptable"
    
    def _get_risk_style(self, severity):
        """Get appropriate style based on risk severity"""
        if severity == "High":
            return self.styles['RiskHigh']
        elif severity == "Medium":
            return self.styles['RiskMedium']
        else:
            return self.styles['RiskLow']