"""
Create a dummy PDF document for RAG testing
"""

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch

def create_sample_pdf():
    """Create a dummy PDF with IT compliance agreement content including LLM usage and timelines"""
    
    # Define filename
    filename = "sample_IT_compliance_document.pdf"
    
    # Ensure directory exists
    import os
    os.makedirs("data", exist_ok=True)
    
    # Create PDF
    try:
        c = canvas.Canvas(f"data/{filename}", pagesize=letter)
    except Exception as e:
        print(f"Error creating PDF: {e}")
        return
    width, height = letter
    
    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(1*inch, 10*inch, "IT Compliance Agreement for using AI")
    
    content = [
        "1. Data Classification Policy",
        "1.1 Company data is classified into three levels: Public, Internal, and Confidential.",
        "1.2 Public data can be shared externally.",
        "1.3 Internal data is for company use only.",
        "1.4 Confidential data requires special handling and encryption.",
        "1.5 All new data must be classified within 24 hours of creation.",

        "2. LLM Usage Compliance",
        "2.1 All Large Language Model usage must be approved by the IT Security team within 48 hours of request.",
        "2.2 Employees must not input confidential company data into public LLM services.",
        "2.3 Approved LLM tools must be logged and monitored.",
        "2.4 All LLM-generated content must be reviewed for accuracy within 24 hours before use.",

        "3. AI Content Generation Policy",
        "3.1 AI-generated content must be clearly labeled as such.",
        "3.2 Employees using AI tools for content creation must verify factual accuracy within 48 hours.",
        "3.3 No sensitive customer or business data can be processed through AI tools without explicit approval within 72 hours.",
        "3.4 All AI usage must be documented within 24 hours of use.",

        "4. Access Control Standards",
        "4.1 User access is granted based on job role and follows the principle of least privilege.",
        "4.2 Access reviews are conducted quarterly by the 30th of March, June, September, and December.",
        "4.3 Terminated employees have their access revoked immediately upon departure.",
        "4.4 Access requests must be approved within 3 business days.",

        "5. Compliance Monitoring",
        "5.1 Regular audits are conducted to ensure compliance with all IT policies.",
        "5.2 Non-compliance issues are addressed within 30 days of discovery.",
        "5.3 Annual training is required for all employees on security policies and procedures by December 31st each year.",
        "5.4 Compliance reports are due quarterly by the 15th of each quarter."
    ]
    
    # Add content to PDF
    y_position = 9*inch
    c.setFont("Helvetica", 12)
    
    for i, text in enumerate(content):
        if i % 2 == 0:  # Section headers
            c.setFont("Helvetica-Bold", 14)
            c.drawString(1*inch, y_position, text)
            y_position -= 0.3*inch
        else:  # Section content
            c.setFont("Helvetica", 12)
            # Split long text into lines
            words = text.split()
            lines = []
            current_line = ""
            
            for word in words:
                if len(current_line + " " + word) < 70:
                    current_line += " " + word if current_line else word
                else:
                    lines.append(current_line)
                    current_line = word
            
            if current_line:
                lines.append(current_line)
            
            for line in lines:
                c.drawString(1*inch, y_position, line)
                y_position -= 0.2*inch
            
            y_position -= 0.3*inch
    
    c.save()
    print(f"✅ Created {filename}")

if __name__ == "__main__":
    create_sample_pdf() 