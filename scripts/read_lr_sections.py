from pdfminer.high_level import extract_text

files = [
    ("GC17_lr", "Doc/Report/GC17_Final Report.pdf"),
    ("GC16_lr", "Doc/Report/GC16-Report.pdf"),
]

for name, path in files:
    try:
        txt = extract_text(path, maxpages=20)
        lines = [l.strip() for l in txt.split("\n") if l.strip()]
        # find lit survey section
        start = 0
        for i, l in enumerate(lines):
            if "Literature" in l or "literature" in l:
                start = i
                break
        print(f"=== {name} lit survey ===")
        print("\n".join(lines[start:start+180]))
        print()
    except Exception as e:
        print(f"ERROR {name}: {e}")
