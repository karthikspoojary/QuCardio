from pdfminer.high_level import extract_text

files = [
    ("Main", "Doc/Report/Main.pdf"),
    ("GC16", "Doc/Report/GC16-Report.pdf"),
    ("GC17", "Doc/Report/GC17_Final Report.pdf"),
    ("Notice", "Doc/Report/Notice.pdf"),
]

for name, path in files:
    try:
        txt = extract_text(path, maxpages=6)
        lines = [l.strip() for l in txt.split("\n") if l.strip()]
        print(f"=== {name} ===")
        print("\n".join(lines[:120]))
        print()
    except Exception as e:
        print(f"ERROR {name}: {e}")
