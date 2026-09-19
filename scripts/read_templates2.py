from pdfminer.high_level import extract_text

files = [
    ("Notice1", "Doc/Report/Notice page2-6 (1).pdf"),
    ("Notice2", "Doc/Report/ProjectFinalPhase&Report(7thSem) (1).pdf"),
    ("GC17_ch1", "Doc/Report/GC17_Final Report.pdf"),
    ("GC16_ch1", "Doc/Report/GC16-Report.pdf"),
    ("Main_ch1", "Doc/Report/Main.pdf"),
]

for name, path in files:
    try:
        txt = extract_text(path, maxpages=12)
        lines = [l.strip() for l in txt.split("\n") if l.strip()]
        print(f"=== {name} ===")
        print("\n".join(lines[:150]))
        print()
    except Exception as e:
        print(f"ERROR {name}: {e}")
