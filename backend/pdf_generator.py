import tempfile
import os
import base64
from datetime import datetime
from fpdf import FPDF

def generate_pdf_report(result_dict: dict, filename: str = "report.pdf") -> str:
    """
    Generate a 1-page PDF clinical report based on the prediction result.
    Returns the path to the generated PDF.
    """
    pdf = FPDF()
    pdf.add_page()
    
    # ── Header ──
    pdf.set_font("helvetica", "B", 16)
    pdf.cell(0, 10, "QuCardio Clinical ECG Analysis Report", ln=True, align="C")
    pdf.ln(5)
    
    # ── Metadata ──
    pdf.set_font("helvetica", "", 12)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf.cell(0, 8, f"Date/Time: {timestamp}", ln=True)
    pdf.cell(0, 8, f"Patient / Source File: {filename}", ln=True)
    
    # Sanitize unicode characters that fpdf2 (latin-1) cannot render
    raw_model_name = result_dict.get('model_used', 'Classical SVM')
    safe_model_name = raw_model_name.replace("—", "-").replace("–", "-")
    pdf.cell(0, 8, f"Model Engine: {safe_model_name}", ln=True)
    pdf.ln(5)
    
    # ── Prediction Results ──
    predicted_class = result_dict.get("prediction", "Unknown")
    confidence = result_dict.get("confidence", 0.0)
    conf_level = result_dict.get("confidence_level", "low")
    class_info = result_dict.get("class_info", {})
    
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, f"Primary Finding: {predicted_class}", ln=True)
    
    pdf.set_font("helvetica", "", 12)
    pdf.cell(0, 8, f"Model Confidence: {confidence*100:.1f}% ({conf_level.upper()})", ln=True)
    
    # Triage Rule
    if confidence < 0.75:
        pdf.set_font("helvetica", "B", 12)
        pdf.set_text_color(220, 53, 69) # Red
        pdf.cell(0, 8, "TRIAGE ACTION: REFER TO CARDIOLOGIST (Low Confidence)", ln=True)
        pdf.set_text_color(0, 0, 0) # Reset to black
    else:
        pdf.set_font("helvetica", "B", 12)
        action = class_info.get("action", "No urgent action.")
        # If the class is MI, make it red regardless of confidence
        if "Myocardial_Infarction" in predicted_class:
            pdf.set_text_color(220, 53, 69)
        elif "Arrhythmia" in predicted_class or "History" in predicted_class:
            pdf.set_text_color(255, 165, 0) # Orange
        else:
            pdf.set_text_color(40, 167, 69) # Green
            
        pdf.cell(0, 8, f"RECOMMENDED ACTION: {action}", ln=True)
        pdf.set_text_color(0, 0, 0)
        
    pdf.ln(5)
    pdf.set_font("helvetica", "", 11)
    desc = class_info.get("description", "No description available.")
    pdf.multi_cell(0, 8, f"Description: {desc}")
    pdf.ln(5)
    
    # ── Probabilities ──
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 8, "Class Probabilities:", ln=True)
    pdf.set_font("helvetica", "", 11)
    probs = result_dict.get("probabilities", {})
    for cls_name, prob in probs.items():
        pdf.cell(0, 8, f"  - {cls_name}: {prob*100:.1f}%", ln=True)
        
    pdf.ln(10)
    
    # ── Embed Image ──
    # The result_dict has the preprocessed_image as a base64 string
    img_b64 = result_dict.get("preprocessed_image", "").replace("data:image/png;base64,", "")
    if img_b64:
        # Decode and save to temp file
        img_data = base64.b64decode(img_b64)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tf:
            tf.write(img_data)
            temp_img_path = tf.name
            
        pdf.set_font("helvetica", "B", 12)
        pdf.cell(0, 8, "Preprocessed ECG Input:", ln=True)
        pdf.image(temp_img_path, w=150)
        
        # Cleanup
        os.remove(temp_img_path)
    
    # Save PDF
    out_dir = os.path.join(tempfile.gettempdir(), "qucardio_reports")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"report_{os.urandom(4).hex()}.pdf")
    pdf.output(out_path)
    
    return out_path
