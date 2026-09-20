import docx
from docx.shared import Inches
import os

def add_image_to_doc(doc, img_path, caption=""):
    if os.path.exists(img_path):
        doc.add_paragraph(caption)
        doc.add_picture(img_path, width=Inches(6.0))
        print(f"Added {img_path}")
    else:
        print(f"MISSING: {img_path}")

# --- Paper Images ---
paper_doc = docx.Document()
paper_doc.add_heading('Paper Figures', 0)

paper_images = [
    ("Fig. 1. Pipeline block diagram.", "Doc/report/extracted_images/qucardio_architecture_ieee (2).png"),
    ("Fig. 2. Web application screens.", "Doc/screenshots/doc1_image1.png"), 
    ("Fig. 3. Confusion matrix for the tuned QSVC.", "results/paper/main/confusion_matrix_qsvc_tuned.png"),
    ("Fig. 4. ROC curves.", "results/paper/main/roc_curves_all_models.png"),
    ("Fig. 5. Model accuracy under nine augmentation conditions.", "results/paper/ablation/augmentation_robustness.png"),
    ("Fig. 6. QSVC accuracy across four feature encoding ranges.", "results/paper/ablation/ablation_encoding.png"),
    ("Fig. 7. QSVC accuracy for five entanglement topologies.", "results/paper/ablation/ablation_entanglement.png"),
    ("Fig. 8. QSVC accuracy for four Pauli feature map variants.", "results/paper/ablation/ablation_feature_maps.png"),
    ("Fig. 9. (Top) QSVC accuracy versus SVD components; (Bottom) versus circuit repetitions.", "results/paper/ablation/ablation_svd_dims.png"),
    ("Fig. 9. (Bottom) Circuit repetitions.", "results/paper/ablation/ablation_reps.png")
]

for caption, path in paper_images:
    add_image_to_doc(paper_doc, path, caption)

paper_doc.save('Doc/Review/QuCardio_Paper_Images.docx')

# --- Report Images ---
report_doc = docx.Document()
report_doc.add_heading('Report Images', 0)

report_images = [
    ("System Architecture", "Doc/report/extracted_images/old/fig4_1_system_architecture.png"),
    ("Use Case Diagram", "Doc/report/extracted_images/old/fig4_2_use_case.png"),
    ("Data Flow Diagram", "Doc/report/extracted_images/old/fig4_3_dfd.png"),
    ("Class Diagram", "Doc/report/extracted_images/old/fig4_4_class_diagram.png"),
    ("Sequence Diagram", "Doc/report/extracted_images/old/fig4_5_sequence.png"),
    ("Screenshot 1", "Doc/screenshots/5.1_preprocessing.png"),
    ("Screenshot 2", "Doc/screenshots/doc1_image2.png"),
    ("Screenshot 3", "Doc/screenshots/doc1_image4.png"),
    ("Result: Confusion Matrix QSVC", "results/paper/main/confusion_matrix_qsvc.png"),
    ("Result: Confusion Matrix SVM", "results/paper/main/classical_svm_cm.png"),
    ("Result: ROC Curves", "results/paper/main/roc_curves_all_models.png")
]

for caption, path in report_images:
    add_image_to_doc(report_doc, path, caption)

report_doc.save('Doc/Report/doc1.docx')
print("Done!")
