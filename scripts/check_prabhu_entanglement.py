import pdfplumber

path = "Doc/Refer/QuCardio_Application_of_Quantum_Machine_Learning_for_Detection_of_Cardiovascular_Diseases (1).pdf"
with pdfplumber.open(path) as pdf:
    for i, page in enumerate(pdf.pages):
        text = page.extract_text()
        if text and any(kw in text.lower() for kw in ['entanglement', 'linear', 'circular', 'full', 'topology']):
            print(f"\n=== PAGE {i+1} ===")
            print(text[:4000])
