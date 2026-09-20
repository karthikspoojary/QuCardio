import re
import sys

filepath = 'Doc/Review/QuCardio_Report_v9.md'
try:
    with open(filepath, 'r') as f:
        content = f.read()
except FileNotFoundError:
    print("File not found")
    sys.exit(1)

# List of Tables
old_tables = """| 7.1 | Model performance comparison | 7.1 |
| 7.2 | Effect of feature encoding range | 7.2 |
| 7.3 | Effect of entanglement topology | 7.2 |
| 7.4 | Pauli feature map comparison | 7.3 |
| 7.5 | McNemar's test results | 7.4 |
| 7.6 | Multi-class ROC-AUC scores | 7.5 |
| 7.7 | Cross-dataset results on Dataset 2 | 7.7 |"""

new_tables = """| 7.1 | Model performance comparison | 7.1 |
| 7.2 | Per-class precision, recall and F1 | 7.1 |
| 7.3 | Effect of feature encoding range | 7.2 |
| 7.4 | Effect of entanglement topology | 7.2 |
| 7.5 | Encoding range fine sweep | 7.2 |
| 7.6 | Pauli feature map comparison | 7.3 |
| 7.7 | McNemar's test results | 7.4 |
| 7.8 | Multi-class ROC-AUC scores | 7.5 |
| 7.9 | Cross-dataset results on Dataset 2 | 7.7 |"""

content = content.replace(old_tables, new_tables)

# List of Abbreviations
old_abbrevs = """| Abbreviation | Full Form |
|---|---|
| API | Application Programming Interface |"""

new_abbrevs = """| Abbreviation | Full Form |
|---|---|
| API | Application Programming Interface |
| FFT | Fast Fourier Transform |
| KTA | Kernel Target Alignment |
| OOD | Out-of-Distribution |
| QPSO | Quantum-behaved Particle Swarm Optimisation |
| VQC | Variational Quantum Circuit |"""

content = content.replace(old_abbrevs, new_abbrevs)

with open(filepath, 'w') as f:
    f.write(content)

print("done")
