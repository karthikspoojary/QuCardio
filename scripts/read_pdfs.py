#!/usr/bin/env python3
"""Extract text from all reference PDFs and write to a single text file."""
from pdfminer.high_level import extract_text
import os

pdfs = [
    ("base_paper", "Doc/Refer/QuCardio_Application_of_Quantum_Machine_Learning_for_Detection_of_Cardiovascular_Diseases (1).pdf"),
    ("diagnostics_ozpolat", "Doc/Refer/diagnostics-13-01099.pdf"),
    ("s11227_jain", "Doc/Refer/s11227-025-07939-8.pdf"),
    ("soon_hierarchical", "Doc/Refer/Early cardiovascular disease detection using hierarchical quantum ensemble model.pdf"),
    ("ramkhelawan_hybrid", "Doc/Refer/A_Hybrid_Quantum-Classical_Deep_Learning_Algorithm_for_Efficient_Arrhythmia_Classification.pdf"),
    ("s0010482518", "Doc/Refer/1-s2.0-S0010482518302713-main.pdf"),
    ("tijer_nasreen", "Doc/Refer/TIJER2604073.pdf"),
    ("tsp_iasc", "Doc/Refer/TSP_IASC_32262.pdf"),
    ("usmtd", "Doc/Refer/10.62301-usmtd.1716034-4943339.pdf"),
]

refer_list_pdfs = [
    ("gummadi_qcnn", "Doc/Refer/refer list/Qunatum-based_Convolutional_Neural_Network_Model_for_Efficient_Cardiovascular_Disease_Prediction.pdf"),
    ("explainable_abdulsalam", "Doc/Refer/refer list/Explainable_Heart_Disease_Prediction_Using_Ensembl.pdf"),
    ("deep_resnet_he", "Doc/Refer/refer list/Deep_Residual_Learning_for_Image_Recognition.pdf"),
    ("arrhythmia_systematic", "Doc/Refer/refer list/Automatic_Classification_of_Cardiac_Arrhythmias_Using_Deep_Learning_Techniques_A_Systematic_Review.pdf"),
    ("benchmarking_mimic", "Doc/Refer/refer list/Benchmarking_Deep_Learning_Architectures_for_ECG-Based_Multi-label_Heart_Disease_Prediction_using_MIMIC-IV_Database.pdf"),
    ("qml_systematic", "Doc/Refer/refer list/1-s2.0-S2666307426000112-main.pdf"),
    ("pegasos_original", "Doc/Refer/refer list/s10107-010-0420-4.pdf"),
    ("quanvolution", "Doc/Refer/refer list/1904.04767v1.pdf"),
    ("qml_review_biamonte", "Doc/Refer/refer list/2206.14200v3.pdf"),
    ("havlicek_quantum", "Doc/Refer/refer list/sciadv.aat9004.pdf"),
    ("yildirim_arrhythmia", "Doc/Refer/refer list/s41598-024-55991-w.pdf"),
    ("heart_disease_hybrid_liu", "Doc/Refer/refer list/Heart disease classification using hybrid ML schemes and optimization tactics in healthcare.pdf"),
    ("fcvm_venkatesh", "Doc/Refer/refer list/fcvm-12-1550422.pdf"),
    ("alotaibi_quantum", "Doc/Refer/refer list/7051-Article Text-15693-1-10-20240913.pdf"),
    ("s41598_elsedimy", "Doc/Refer/refer list/s41598-024-61452-1 (1).pdf"),
    ("shilpa_arrhythmia", "Doc/Refer/refer list/zkyjyml2163kpuaxq3yd9g5tayfxppei.pdf"),
    ("pone_journal", "Doc/Refer/refer list/journal.pone.0327928.pdf"),
    ("quantum_cardio_revision", "Doc/Refer/refer list/Quantum_Cardio_Revision.pdf"),
    ("qml_advances", "Doc/Refer/refer list/Quantum_Machine_Learning_Recent_Advances_Challenges_and_Perspectives (1).pdf"),
    ("lbrj", "Doc/Refer/refer list/15-LBRJ2232.pdf"),
    ("mosca", "Doc/Refer/refer list/Mosca.pdf"),
    ("13_41", "Doc/Refer/refer list/13-41.pdf"),
    ("31658", "Doc/Refer/refer list/31658179800805.pdf"),
]

all_pdfs = pdfs + refer_list_pdfs

out_path = "scripts/pdf_extracts.txt"
with open(out_path, "w", encoding="utf-8") as out:
    for name, path in all_pdfs:
        out.write(f"\n\n{'='*80}\n")
        out.write(f"KEY: {name}\nPATH: {path}\n")
        out.write('='*80 + "\n")
        if not os.path.exists(path):
            out.write("[FILE NOT FOUND]\n")
            continue
        try:
            text = extract_text(path)
            # write first 350 lines
            lines = text.split("\n")
            out.write("\n".join(lines[:350]))
            out.write(f"\n\n[... truncated at 350 lines, total lines: {len(lines)} ...]\n")
        except Exception as e:
            out.write(f"[ERROR: {e}]\n")

print(f"Done. Written to {out_path}")
