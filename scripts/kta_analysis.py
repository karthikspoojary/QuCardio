"""Quick KTA sweep analysis for paper/report text."""
import json

with open('results/paper/ablation/kta_all_configs.json') as f:
    kta_data = json.load(f)

unique_ktas = {}
for key, v in kta_data.items():
    if v['kta'] is None:
        continue
    k = (v['encoding'], v['entanglement'], v['reps'])
    unique_ktas[k] = v['kta']

print(f"Unique non-null KTA configs: {len(unique_ktas)}")
vals = sorted(unique_ktas.values())
print(f"KTA range: {min(vals):.6f} to {max(vals):.6f}")

sorted_kta = sorted(unique_ktas.items(), key=lambda x: x[1], reverse=True)
print("Top-10 KTA configs:")
for k, v in sorted_kta[:10]:
    print(f"  {k}: {v:.6f}")

print("\nBottom-5 KTA configs:")
for k, v in sorted_kta[-5:]:
    print(f"  {k}: {v:.6f}")

# The optimal accuracy config
opt = ('minmax_0pi', 'circular', 2)
if opt in unique_ktas:
    print(f"\nOptimal-accuracy config KTA {opt}: {unique_ktas[opt]:.6f}")

# Mean by encoding
from collections import defaultdict
by_enc = defaultdict(list)
for (enc, ent, reps), kta in unique_ktas.items():
    by_enc[enc].append(kta)

print("\nMean KTA by encoding:")
for enc, ktalist in sorted(by_enc.items()):
    print(f"  {enc}: mean={sum(ktalist)/len(ktalist):.6f}")

# Mean by entanglement
by_ent = defaultdict(list)
for (enc, ent, reps), kta in unique_ktas.items():
    by_ent[ent].append(kta)

print("\nMean KTA by entanglement:")
for ent, ktalist in sorted(by_ent.items()):
    print(f"  {ent}: mean={sum(ktalist)/len(ktalist):.6f}")

# Mean by reps
by_reps = defaultdict(list)
for (enc, ent, reps), kta in unique_ktas.items():
    by_reps[reps].append(kta)

print("\nMean KTA by reps:")
for reps, ktalist in sorted(by_reps.items()):
    print(f"  reps={reps}: mean={sum(ktalist)/len(ktalist):.6f}")
