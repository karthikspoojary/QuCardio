import base64
import os

imgs = [
    ("zzfeaturemap_circuit", "results/paper/main/zzfeaturemap_circuit.png", "Figure 5.1: ZZFeatureMap Circuit (9-qubit, reps=2, circular entanglement)"),
    ("classical_baselines", "results/paper/main/classical_baselines.png", "Figure 6.1: All Models Accuracy Comparison"),
    ("confusion_qsvc", "results/paper/main/confusion_matrix_qsvc_tuned.png", "Figure 6.2: QSVC Confusion Matrix"),
    ("confusion_pegasos", "results/paper/main/confusion_matrix_pegasos_tuned.png", "Figure 6.3: Pegasos Confusion Matrix"),
    ("roc", "results/paper/main/roc_curves_all_models.png", "Figure 6.4: ROC Curves All Models and Classes"),
    ("calibration", "results/paper/main/calibration_all_models.png", "Figure 6.5: Confidence Calibration Curves"),
    ("ablation_enc", "results/paper/ablation/ablation_encoding.png", "Figure 6.6: Feature Encoding Range Ablation"),
    ("ablation_ent", "results/paper/ablation/ablation_entanglement.png", "Figure 6.7: Entanglement Topology Ablation"),
    ("ablation_svd", "results/paper/ablation/ablation_svd_dims.png", "Figure 6.8: SVD Dimensionality Ablation"),
    ("ablation_reps", "results/paper/ablation/ablation_reps.png", "Figure 6.9: Circuit Repetitions Ablation"),
    ("augmentation", "results/paper/ablation/augmentation_robustness.png", "Figure 6.10: Augmentation Robustness Study"),
    ("comparison", "results/exploratory/comparison_all_models.png", "Supplementary: All Models Comparison Bar Chart"),
]

sections = []
for key, path, cap in imgs:
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        sections.append((cap, data))
        print(f"OK: {path}")
    else:
        print(f"MISSING: {path}")

# write HTML
html_parts = ['<!DOCTYPE html><html><head><meta charset="utf-8">',
              '<title>QuCardio Report Figures</title>',
              '<style>',
              'body{font-family:Arial,sans-serif;max-width:900px;margin:0 auto;padding:20px;background:#f5f5f5;}',
              'h1{color:#1a237e;border-bottom:3px solid #1a237e;padding-bottom:10px;}',
              '.fig{background:white;border-radius:8px;padding:20px;margin:20px 0;box-shadow:0 2px 8px rgba(0,0,0,0.1);}',
              '.fig img{width:100%;height:auto;border:1px solid #ddd;}',
              '.cap{font-weight:bold;font-size:14px;color:#333;margin-top:10px;text-align:center;}',
              '.info{background:#e8f0fe;border-left:4px solid #1565c0;padding:15px;margin:20px 0;font-size:13px;}',
              '</style></head><body>',
              '<h1>QuCardio &mdash; Report Figures</h1>',
              '<div class="info"><strong>Instructions:</strong> Insert these figures into the report at the sections marked in QuCardio_Report_Content.md. Each figure is labeled with its caption. Right-click any image to save it. These are the actual result images generated from the QuCardio codebase.</div>',
]

for cap, data in sections:
    html_parts.append('<div class="fig">')
    html_parts.append(f'<img src="data:image/png;base64,{data}" alt="{cap}">')
    html_parts.append(f'<div class="cap">{cap}</div>')
    html_parts.append('</div>')

html_parts.append('</body></html>')

with open("Doc/Report/QuCardio_Report_Figures.html", "w") as f:
    f.write("\n".join(html_parts))

print("\nWrote Doc/Report/QuCardio_Report_Figures.html")
