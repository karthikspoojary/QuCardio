import os, json

base = "data/new ecg data"
origin_dir = os.path.join(base, "1_origin")

if not os.path.exists(origin_dir):
    print("Directory not found:", origin_dir)
    print("Available under data/:", os.listdir("data") if os.path.exists("data") else "none")
else:
    files = os.listdir(origin_dir)
    # Label by filename prefix conventions used in Jain et al. Dataset 2
    normal     = [f for f in files if "NSR" in f or f.lower().startswith("normal")]
    arrhythmia = [f for f in files if any(x in f for x in
                  ["AF","PAC","PVC","LBBB","RBBB","STB","ARR","Arr"])]
    other      = [f for f in files if f not in normal and f not in arrhythmia]

    print("Total files in 1_origin:", len(files))
    print("Normal-labelled:        ", len(normal))
    print("Arrhythmia-labelled:    ", len(arrhythmia))
    print("Unlabelled/other:       ", len(other))
    print("Sample filenames:", sorted(files)[:12])

    n = 45  # fixed eval subset size from augmentation_robustness.json
    majority = max(len(normal), len(arrhythmia))
    print()
    print(f"Majority-class baseline on {n}-image subset: {majority}/{n} = {majority/n*100:.1f}%")
    print(f"Clean SVM/QSVC accuracy:  84.4%  (38/45)")
    print(f"SVM under brightness+:    46.7%  (21/45)  <-- below coin-flip AND below majority")
    print(f"QSVC under brightness+:   86.7%  (39/45)")
