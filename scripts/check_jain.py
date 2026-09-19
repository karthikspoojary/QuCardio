from pdfminer.high_level import extract_text

# Jain et al. 2025 - Journal of Supercomputing (s11227 = correct file)
# The s11042 file is actually Elsedimy QPSO-SVM paper
# Let's read both to confirm
files = [
    ("s11042_elsedimy", "Doc/Refer/refer list/s11042-023-16194-z.pdf"),
    ("s11227_jain", "Doc/Refer/refer list/s11042-023-16194-z.pdf"),
]

# The Jain paper should be a different file - check what we have
import os
for f in os.listdir("Doc/Refer/refer list/"):
    print(f)
