import os

base = "data/new ecg data/1_origin"
files = [f for f in os.listdir(base) if not f.endswith("Zone.Identifier")]
arrhythmia_prefixes = ["AF","PAC","PVC","LBBB","RBBB","STB","SNT","SNB","AVBI","SVT","VT"]
normal_prefixes = ["NSR"]
# Map SNT/SNB/AVBI etc to Arrhythmia (they are all abnormal rhythms)
normal = [f for f in files if f.startswith("NSR")]
arrhythmia = [f for f in files if not f.startswith("NSR")]
prefixes = sorted(set(f.split("_")[0] for f in files))
print("All prefixes:", prefixes)
print("Normal (NSR):", len(normal))
print("Arrhythmia (all non-NSR):", len(arrhythmia))
n = 45
majority = max(len(normal), len(arrhythmia))
print("Majority class: Arrhythmia %d/%d = %.1f%%" % (majority, n, majority/n*100))
print()
print("--- Augmentation robustness interpretation ---")
print("Clean accuracy (both models): 84.4%% = 38/45")
print("Majority-class baseline:      %.1f%% = %d/45" % (majority/n*100, majority))
print("SVM brightness+:              46.7%% = 21/45  (%.1f pp below majority)" % (majority/n*100 - 46.7))
print("QSVC brightness+:             86.7%% = 39/45  (%.1f pp above majority)" % (86.7 - majority/n*100))
