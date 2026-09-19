from pdfminer.high_level import extract_text
import os

files = [
    ("1307", "Doc/Refer/refer list/1307.0471v3.pdf"),
    ("Mosca", "Doc/Refer/refer list/Mosca.pdf"),
    ("1341", "Doc/Refer/refer list/13-41.pdf"),
    ("31658", "Doc/Refer/refer list/31658179800805.pdf"),
    ("15LB", "Doc/Refer/refer list/15-LBRJ2232.pdf"),
    ("7051", "Doc/Refer/refer list/7051-Article Text-15693-1-10-20240913.pdf"),
    ("2206", "Doc/Refer/refer list/2206.14200v3.pdf"),
    ("1201", "Doc/Refer/refer list/1201.0490v4.pdf"),
    ("s11042", "Doc/Refer/refer list/s11042-023-16194-z.pdf"),
    ("zkyjy", "Doc/Refer/refer list/zkyjyml2163kpuaxq3yd9g5tayfxppei.pdf"),
]
base = "/home/karthik/projects/QuCardio/"

for key, f in files:
    path = base + f
    try:
        txt = extract_text(path, maxpages=2)
        lines = [l.strip() for l in txt.split("\n") if l.strip()]
        print(f"=== {key} ===")
        print("\n".join(lines[:45]))
        print()
    except Exception as e:
        print(f"ERROR {key}: {e}")
