import tempfile
import os
import re
import base64
from datetime import datetime
from fpdf import FPDF


def _sanitize_latin1(text: str) -> str:
    """
    Sanitize text to be safe for fpdf2's latin-1 core fonts.
    Replaces common Unicode-only characters with ASCII equivalents.
    Non-latin-1 characters that cannot be mapped are stripped.
    """
    replacements = {
        "\u2014": "-",  # EM DASH
        "\u2013": "-",  # EN DASH
        "\u2018": "'",  # LEFT SINGLE QUOTATION
        "\u2019": "'",  # RIGHT SINGLE QUOTATION
        "\u201c": '"',  # LEFT DOUBLE QUOTATION
        "\u201d": '"',  # RIGHT DOUBLE QUOTATION
        "\u2026": "...",  # ELLIPSIS
        "\u00b0": "deg",  # DEGREE SIGN
        "\u03c0": "pi",   # GREEK SMALL LETTER PI
        "\u00ae": "(R)",  # REGISTERED SIGN
        "\u00a9": "(C)",  # COPYRIGHT
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    # Strip any remaining non-latin-1 characters
    return text.encode("latin-1", errors="ignore").decode("latin-1")


def generate_pdf_report(result_dict: dict, filename: str = "report.pdf") -> str:
    """
    Generate a 1-page PDF clinical report based on the prediction result.
    Returns the path to the generated PDF.
    Caller is responsible for cleanup (use BackgroundTasks in FastAPI).
    """
    # Sanitize filename — remove path separators and non-latin-1 chars
    safe_filename = _sanitize_latin1(os.path.basename(filename))

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
    pdf.cell(0, 8, f"Source File: {safe_filename}", ln=True)

    # Sanitize model name
    raw_model_name = result_dict.get("model_used", "Classical SVM")
    safe_model_name = _sanitize_latin1(raw_model_name)
    pdf.cell(0, 8, f"Model Engine: {safe_model_name}", ln=True)

    # ── Patient Info (optional) ──
    patient_info = result_dict.get("patient_info", {})
    has_patient = patient_info and any(v for v in patient_info.values() if v)
    if has_patient:
        pdf.ln(3)
        pdf.set_font("helvetica", "B", 12)
        pdf.cell(0, 8, "Patient Information:", ln=True)
        pdf.set_font("helvetica", "", 11)
        if patient_info.get("name"):
            pdf.cell(0, 7, f"  Name:   {_sanitize_latin1(patient_info['name'])}", ln=True)
        if patient_info.get("id"):
            pdf.cell(0, 7, f"  ID:     {_sanitize_latin1(patient_info['id'])}", ln=True)
        if patient_info.get("age"):
            sex_str = f" / {patient_info['sex']}" if patient_info.get("sex") else ""
            pdf.cell(0, 7, f"  Age:    {_sanitize_latin1(str(patient_info['age']))}{sex_str}", ln=True)
        if patient_info.get("doctor"):
            pdf.cell(0, 7, f"  Doctor: {_sanitize_latin1(patient_info['doctor'])}", ln=True)

    pdf.ln(5)

    # ── Prediction Results ──
    predicted_class = result_dict.get("prediction", "Unknown")
    confidence = result_dict.get("confidence", 0.0)
    conf_level = result_dict.get("confidence_level", "low")
    class_info = result_dict.get("class_info", {})

    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, f"Primary Finding: {predicted_class.replace('_', ' ')}", ln=True)

    pdf.set_font("helvetica", "", 12)
    pdf.cell(0, 8, f"Model Confidence: {confidence*100:.1f}% ({conf_level.upper()})", ln=True)

    # Triage Rule — use the backend's class_info.action (correct per Copilot review)
    pdf.set_font("helvetica", "B", 12)
    if confidence < 0.75:
        pdf.set_text_color(220, 53, 69)  # Red
        pdf.cell(0, 8, "TRIAGE ACTION: REFER TO CARDIOLOGIST (Low Confidence)", ln=True)
    else:
        action = _sanitize_latin1(class_info.get("action", "No urgent action."))
        if "Myocardial_Infarction" in predicted_class:
            pdf.set_text_color(220, 53, 69)   # Red — emergency
        elif "Arrhythmia" in predicted_class or "History" in predicted_class:
            pdf.set_text_color(200, 100, 0)   # Orange — follow up
        else:
            pdf.set_text_color(40, 167, 69)   # Green — normal
        pdf.cell(0, 8, f"RECOMMENDED ACTION: {action}", ln=True)
    pdf.set_text_color(0, 0, 0)

    pdf.ln(5)
    pdf.set_font("helvetica", "", 11)
    desc = _sanitize_latin1(class_info.get("description", "No description available."))
    pdf.multi_cell(0, 8, f"Description: {desc}")
    pdf.ln(5)

    # ── Probabilities ──
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 8, "Class Probabilities:", ln=True)
    pdf.set_font("helvetica", "", 11)
    probs = result_dict.get("probabilities", {})
    for cls_name, prob in probs.items():
        pdf.cell(0, 8, f"  - {cls_name.replace('_', ' ')}: {prob*100:.1f}%", ln=True)

    pdf.ln(10)

    # ── Embed Preprocessed ECG Image ──
    img_b64 = result_dict.get("preprocessed_image", "").replace("data:image/png;base64,", "")
    if img_b64:
        img_data = base64.b64decode(img_b64)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tf:
            tf.write(img_data)
            temp_img_path = tf.name
        try:
            pdf.set_font("helvetica", "B", 12)
            pdf.cell(0, 8, "Preprocessed ECG Input:", ln=True)
            pdf.image(temp_img_path, w=150)
        finally:
            os.remove(temp_img_path)

    # ── Disclaimer ──
    pdf.ln(8)
    pdf.set_font("helvetica", "I", 9)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(0, 6, "DISCLAIMER: This report is generated by an AI research system and is NOT a certified medical device. It must not replace professional clinical judgment. Always consult a qualified cardiologist for diagnosis and treatment.")
    pdf.set_text_color(0, 0, 0)

    # Save PDF to temp dir
    out_dir = os.path.join(tempfile.gettempdir(), "qucardio_reports")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"report_{os.urandom(4).hex()}.pdf")
    pdf.output(out_path)

    return out_path
