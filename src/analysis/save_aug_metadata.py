import json

with open("results/paper/ablation/augmentation_robustness.json") as f:
    data = json.load(f)

data["_metadata"] = {
    "n_images": 45,
    "classes": "Normal (NSR=7) vs Arrhythmia (AF+LBBB+RBBB+SNT+SNB+AVBI=38)",
    "majority_class": "Arrhythmia",
    "majority_class_baseline_pct": 84.4,
    "majority_count": 38,
    "note": (
        "Clean accuracy 84.4% (38/45) equals the majority-class rate exactly. "
        "Under brightness+ the SVM falls to 46.7% (37.7 pp below majority baseline); "
        "the QSVC holds at 86.7% (2.3 pp above majority baseline)."
    )
}

with open("results/paper/ablation/augmentation_robustness.json", "w") as f:
    json.dump(data, f, indent=2)

print("Updated augmentation_robustness.json with majority-class metadata")
print("Majority-class baseline: 84.4% (38/45 Arrhythmia)")
print("SVM brightness+: 46.7% = 37.7 pp BELOW majority baseline")
print("QSVC brightness+: 86.7% = 2.3 pp above majority baseline")
