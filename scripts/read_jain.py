from pdfminer.high_level import extract_text

txt = extract_text("Doc/Refer/s11227-025-07939-8.pdf", maxpages=20)
lines = [l.strip() for l in txt.split("\n") if l.strip()]
print("\n".join(lines))
