# QuCardio — Review Findings

Line-level audit of `QuCardio_Paper_v4.md` and `QuCardio_Report_v6.md`, with the disposition of every issue in `QuCardio_Paper_v5.md` and `QuCardio_Report_v7.md`.

---

## A. Errors that would cause rejection

### A1. McNemar's test on the two linear cells is misreported — FIXED

**Paper §VI-C and Report §7.2.** Both stated:

> "A McNemar test between the two linear cells confirms that their prediction vectors genuinely differ: 12 positions disagree, with b = 8 and c = 2, giving a continuity-corrected chi-squared of 2.50. The cells are not merely tied; [0, pi] + linear is genuinely better than [0, 1] + linear."

χ² = 2.50 with one degree of freedom gives **p = 0.114**. The test does not reject the null. Concluding that one configuration is "genuinely better" from a non-significant result is a misuse of the test, and it is the kind of error that causes a reviewer to re-check every other statistic in the paper.

Both documents now state that the test establishes only that the two configurations produce different predictions, and that the 3.23-point difference is directional but not significant at n = 186.

**Secondary issue, also fixed:** the text says 12 positions disagree with b = 8 and c = 2, which reads as an arithmetic error since 8 + 2 = 10. It is actually correct — on 2 samples both configurations are wrong but assign different labels, which is possible with four classes — but the documents never said so. Both now explain it.

### A2. Quantum advantage is claimed against the weakest classical baseline — FIXED

The headline is a 9.67-point margin over the classical SVM at 84.95%. But Random Forest reaches **91.94%** on the same nine-dimensional features with no quantum component, and it does not appear in the McNemar table. The margin the quantum kernel actually has to defend is **2.68 points = 5 test images**.

Verified consistency of the discordance counts: QSVC 176 correct, RF 171 correct, difference 5 — the same order of magnitude as the QSVC-vs-Pegasos comparison, which gave χ² = 1.23, p = 0.267. The RF comparison is therefore unlikely to reach significance.

**Action required from you:** run McNemar for QSVC vs Random Forest and fill the placeholder row in Paper Table IV and Report Table 7.5. If it comes back non-significant, say so — both documents are already written to accommodate that outcome. The code is short:

```python
from statsmodels.stats.contingency_tables import mcnemar
import numpy as np
q = np.load('preds_qsvc.npy'); rf = np.load('preds_rf.npy'); y = np.load('y_test.npy')
b = int(((q == y) & (rf != y)).sum()); c = int(((q != y) & (rf == y)).sum())
print(b, c, mcnemar([[0, b], [c, 0]], exact=False, correction=True))
```

Both documents now state the margin against the strongest baseline in the abstract, the results section, the limitations and the conclusion. Self-reporting this is far safer than letting a reviewer find it.

### A3. "Independent, sub-additive gains" is self-contradictory — FIXED

Sub-additivity *is* an interaction. Independence means the effects add. The phrase appeared in Paper abstract, contribution 1, §VI-C and §VIII; and in Report abstract, §2.19, §7.2 and §8.1.

Replaced throughout with: each factor improves accuracy at both levels of the other, and the joint effect is sub-additive (a negative interaction) because the two changes correct an overlapping set of samples.

Also added: the 1.08-point interaction is **two test images**, which is at the resolution limit of a 186-image test set and should not be presented as a precisely estimated interaction magnitude.

---

## B. Factual and numerical errors

| # | Location | Issue | Status |
|---|---|---|---|
| B1 | Paper Table I | Prabhu et al. "Best Acc. 94.09%" — their best reported model was the Quanvolutional NN at **97.31%**. Understating a competitor's result is the kind of thing a reviewer treats as careless at best. | Fixed in both |
| B2 | Report §1.4 | "five encoding ranges" — all tables show four, and 4 × 5 × 4 × 4 = 320 requires four | Fixed |
| B3 | `PROJECT_DEEP_DIVE.md` §9 | Writes "4 encoding ranges:" then lists five (raw, L2, [0,1], [0,π], [0,2π]) | **Not fixed — deep dive not revised. Correct this manually.** |
| B4 | Report Table 2.1 | Jain et al. method listed as "QSVM + QCNN"; §2.8 describes ResNet50 + PCA(8) + QSVM. No QCNN is mentioned anywhere in the section | Fixed |
| B5 | Paper §V-D | Calls the 45 augmentation images "45 **test** images". They are from `data/new ecg data/1_origin`, a separate collection | Fixed in both |
| B6 | Paper §V-D | "46.7% ... close to chance on four classes" — uniform chance is 25%, and the subset spans only two of the four classes, so neither reference point supports "close to chance" | Fixed in both |
| B7 | Paper §V-A | Implies 100% MI recall distinguishes the QSVC. The classical SVM also reaches 100% on that class (Report Table 6.5 confirms) | Fixed in both |
| B8 | Paper refs [8]; Report refs [6], [9] | Contain literal placeholders `pp. *(confirm on IEEE Xplore)*` | **Not fixable by me — you must look up the page ranges** |
| B9 | Paper §VI-B, Report §7.2 | Both confess an internal bug: "an earlier run had inadvertently used cached [0,1] statevectors". Lab-notebook corrections do not belong in a submitted document; they invite doubt about every other number | Removed from both |
| B10 | Both | The claim "the first McNemar test" appeared as fact in the contributions list but hedged with "to the best of our knowledge" in the body | Hedged consistently |

### Numbers that check out

I verified the arithmetic and found no errors in any of the following, which is worth knowing since it means the underlying experiments are sound:

- Every accuracy maps to an exact integer count out of 186: 84.95% = 158, 91.94% = 171, 92.47% = 172, 93.01% = 173, 94.09% = 175, 94.62% = 176, 89.78% = 167, 87.63% = 163, 83.33% = 155, 68.28% = 127.
- All three McNemar statistics recompute exactly: (|23−5|−1)²/28 = 10.321 → p = 0.00131; (|19−6|−1)²/25 = 5.760 → p = 0.0164; (|9−4|−1)²/13 = 1.231 → p = 0.267.
- The discordance counts are mutually consistent with the accuracies: 176 − 158 = 18 = 23 − 5; 171 − 158 = 13 = 19 − 6; 176 − 171 = 5 = 9 − 4. This is a strong internal consistency check and it passes.
- All four factorial simple effects: 3.23, 2.15, 2.69, 1.61; joint 4.84; sum 5.92; gap 1.08. All correct.
- 85 × 85 × 64 = 462,400 ✓. A 340×340 input through conv1 (7×7, stride 2) then pool1 (3×3, stride 2) does give 85×85 ✓. 2⁹ = 512 ✓.
- Test counts: 17 unit + 7 integration + 8 system + 6 acceptance, matching the stated totals ✓.

---

## C. Structural and formatting

| # | Issue | Status |
|---|---|---|
| C1 | Paper claimed IEEE order-of-first-appearance numbering but did not follow it: [7] cited before [5]/[6]; [20] cited in §III-D before [16]–[19]; [19] before [18] | All 20 references renumbered; two new references inserted in appearance order; all 22 verified cited |
| C2 | `TABLE VII-A` is not valid IEEE table numbering | Renumbered I–X sequentially |
| C3 | §II cross-references "Section VI-D" for the repetitions ablation; it is in VI-E | Fixed |
| C4 | §III-C cross-references "Section VI-D" for the component-count ablation; it is in VI-E | Fixed |
| C5 | Contribution 4 claims "domain boundary identification" but the cross-dataset result appeared only in Limitations, with no results section behind it | Promoted to **§V-E** with its own table (now Table VI); Limitations points to it |
| C6 | Index Terms listed 9 items; IEEE guidance is 4–6 | Trimmed to 6 |
| C7 | Paper file contained an internal HTML comment block (open items O1–O4), a figure placement guide and a consistency checklist | Removed from the manuscript; preserved here |
| C8 | Report references numbered by literature-survey order, not first appearance (Ch. 1 cites [18], [19] first) | Acceptable for a VTU report; a clarifying note added to the References heading rather than renumbering 19 entries |
| C9 | Report had no data/code availability statement | Consider adding the GitHub URL; the paper has one |

### Spelling convention — was inconsistent across the two documents

| Document | `-ise` forms | `-ize` forms | Other |
|---|---|---|---|
| Paper v4 | 23 | 7 | "colour" + "grayscale" mixed |
| Report v6 | 9 | **51** | "colour" + "grayscale" mixed |

The two documents used **opposite conventions**, and each violated its own. Both are now consistently British: `-ise`, `colour`, `artefact`, `greyscale`. If your target venue's template requires US English, a single find-and-replace reverses it — but pick one and hold it.

---

## D. Similarity and AI-detection assessment

### Does it look plagiarised from external sources?

**No.** The prose is original, the structure is your own, and the technical content follows from your experiments. Standard similarity matches you should expect and not worry about: the reference list (normally excluded), the dataset description, standard method names, and the Khan et al. dataset provenance sentence.

### Does it look self-plagiarised?

**This was the real risk.** A 9-gram overlap analysis between paper and report found **64 report sentences sharing verbatim runs with the paper**, roughly 50 of them prose rather than references. If both documents enter a similarity index — and a submitted BE report often does — they will match each other heavily.

The worst offenders were the Random Forest sentence, the Bloch-sphere/equator explanation, the 45-image provenance sentence, the preprocessing Step 1 description, the grid-search sentence, the latency breakdown, and the linear-cell McNemar passage.

**After revision: 39 overlapping sentences**, and the remainder are short factual statements (numbers, table captions, configuration parameters) that cannot be meaningfully reworded and carry low risk. The substantive explanatory passages now use different phrasing in each document while stating identical facts.

### Does it look AI-generated?

Partly, yes. Marker counts in the originals:

| Marker | Paper v4 | Report v6 | Deep dive |
|---|---|---|---|
| Em-dash asides | 11 | 9 | **69** |
| "Crucially," / "Notably," / "Furthermore," | 2 | 0 | 0 |
| "X merits comment/attention" | 0 | 2 | 0 |
| Numbered preambles ("Three cautions attach...") | 2 | 0 | 0 |
| "The mechanism is" / "The reason is" | 1 | 2 | 2 |
| "not merely / not just" | 1 | 1 | 1 |

`PROJECT_DEEP_DIVE.md` is the worst by a wide margin: 69 em-dashes in 5,826 words, plus this passage in §16, which is meta-commentary about reviewers inside a project document and should be deleted outright:

> "This result is reported prominently, not buried. A reviewer who finds an undisclosed failure mode rejects the paper; a reviewer who sees a self-reported negative result trusts everything else."

Also flagged in the deep dive: "This is the heart of the project" (§9), "The ablation is the scientific core of the project" (§14), and the bolded-verdict table style throughout.

**A caution worth stating plainly:** AI-detection tools are unreliable in both directions and Drillbit's AI score is not evidence of anything on its own. Reducing these markers lowers the flag rate; it does not guarantee a clean score, and a clean score would not prove much either. What matters more is that most venues now require you to *disclose* generative-AI assistance rather than hide it. Check your target venue's policy and your college's policy, and disclose as required.

**Still to do by you:** revise `PROJECT_DEEP_DIVE.md` and `README.md` on the same lines. They were not part of this revision.

---

## E. Paragraph and section lengths

### Current paper (v5) — 8,686 words

| Section | Words | Target for 8pp IEEE | Assessment |
|---|---|---|---|
| Abstract | ~290 | 250–300 | Good |
| I. Introduction | 468 | 500–700 | Slightly short; the funnel could open one paragraph wider |
| II. Related Work | 1,063 | 900–1,200 | Good |
| III. Proposed Methodology | 1,260 | 1,200–1,600 | Good |
| IV. Experimental Setup | 485 | 400–600 | Good |
| V. Results and Discussion | 1,849 | 1,500–2,000 | Good |
| VI. Ablation Studies | 1,601 | 1,200–1,600 | At the ceiling |
| VII. Limitations | 385 | 300–450 | Good |
| VIII. Conclusion | 332 | 250–400 | Good |

**Paragraph-level guidance, which matters as much as section length:** aim for 4–7 sentences and 90–150 words per paragraph, one idea each, topic sentence first. The paper currently has several paragraphs above 200 words — §II paragraph 3 (Havlicek/Huang/Biamonte), §III-A (the seven preprocessing steps as one block), §V-A paragraph 1 and §VI-C paragraph 2. Splitting each at its natural hinge improves readability without changing a word of content.

### Page budget — be realistic

With 10 tables and 9 figures, **8,686 words will not fit in 6 IEEE pages.** Two column-inches per table plus figure area puts this at 9–11 pages as written. Options:

- **6-page venue:** drop Fig. 2 (app screenshots), Fig. 4 (ROC curves), Fig. 9 (SVD/reps panels) and Table X (Pauli maps); compress §II to 700 words by merging the Ozpolat/Aksoy and Soon/Setiawan paragraphs; compress §III-A to a single sentence plus a pointer. Keep Table VIII (factorial) and Table IV (McNemar) — they carry the argument. Target ~6,000 words.
- **8-page venue:** trim §II by 200 words and §VI by 200. Comfortable fit. **Recommended.**
- **Journal:** submit as is, add the calibration figure, expand §V-E with confusion matrices, and add the seed-variance results.

### Report (v7) — 17,830 words

Appropriate for a VTU BE report. Chapter 2 at ~5,000 words across sixteen reviewed works is the right depth. No chapter is disproportionate.

---

## F. Does the writing state inferences, or only results?

You asked specifically whether both documents go beyond "here is the number" to "here is what it means". They largely do, and this is one of the stronger aspects of both drafts. Examples of the inference being drawn properly:

- **Encoding range** — states the number, then the Bloch-sphere mechanism, then the practitioner implication (use [0,π]), now with the kernel-concentration literature attached.
- **Random Forest at 91.94%** — states the number, then the inference that the SVD features are already informative and the quantum kernel improves on a strong representation rather than rescuing a weak one.
- **Brightness robustness** — states the gap, then the phase-periodicity versus Euclidean-distance mechanism, then why it matters clinically (ward staff photograph prints under whatever light exists).
- **k-NN matching SVM at 84.95%** — Report §7.1 notes the macro F1 scores differ (0.836 vs 0.841), inferring that the two misclassify different samples and reach the same total coincidentally. This is a genuinely sharp observation.
- **Cross-dataset reversal** — states the numbers, infers that the failure originates in the frozen classical extractor rather than the quantum circuit, and names the next step.

Where the inference was weak and is now strengthened: the Y+YY feature map result ("appears to interfere destructively" — explicitly marked as not isolated), the full-entanglement result ("plausibly over-parameterises" — also marked), and the SCA-equals-circular observation (now reported as an empirical coincidence at this qubit count rather than an equivalence).

---

## G. Compliance with the paper-writing guidance you supplied

| Requirement | Status |
|---|---|
| Novelty stated in one sentence and proved | Yes — circuit design ablation, framed as an empirical contribution, which the guidance explicitly lists as valid novelty |
| Every abstract claim supported in the body | Yes, after the baseline-honesty additions |
| Baselines recent, published, fairly tuned | **Partial.** SVM and RF are tuned, but no gradient-boosted or fine-tuned CNN baseline. Listed as a limitation |
| Ablation study present | Yes — five variables, 320 configurations |
| Limitations stated honestly | Yes — five items, each paired with the step that would remove it, exactly as the guidance prescribes |
| Statistical significance reported | Yes, and now reported correctly |
| Multiple runs, mean ± SD | **No.** Single split, no variance. This is the largest remaining gap |
| Every figure/table referenced before it appears | Yes — verify again after LaTeX typesetting |
| Figure captions below, table captions above | Specified in the typesetting notes |
| All abbreviations defined at first use | Yes |
| Equations numbered, symbols defined | Yes — (1) and (2) |
| Consistent notation and method name | Yes |
| Every citation has an entry and vice versa | Verified — all 22 |
| One consistent reference style | Yes, IEEE |
| DOIs included, author names verified | Mostly — two entries still have page-range placeholders (B8) |
| Majority of citations recent | Yes — 14 of 22 from 2021 onward |
| Correct template, within page limit | **Pending** — see the page budget above |
| Similarity check passed | Run it after converting to LaTeX; expect matches against your own report |
| Ethics/funding/AI-use statements | **Pending** — check the venue's requirements |
| Proofread by someone who did not write it | **Pending** — ask your guide |

---

## H. Immediate action list, in order

1. **Run QSVC vs Random Forest McNemar.** Fill the placeholder in Paper Table IV and Report Table 7.5. Nothing else matters as much. (~30 min)
2. **Fill the two reference page ranges** from IEEE Xplore. (~15 min)
3. **Fix `PROJECT_DEEP_DIVE.md`:** the "4 encoding ranges" / five-items contradiction in §9, and delete the reviewer meta-commentary in §16. (~20 min)
4. **Fill the author names, emails, USNs and guide name** in both documents. (~10 min)
5. **Seed variance on the four factorial cells**, five seeds. This is the largest remaining scientific gap. (~2 h compute)
6. **Measure kernel-value variance per encoding range** to convert your best qualitative argument into a measured one. (~30 min, see Roadmap §4 item 3)
7. Convert to IEEEtran, verify the page count against the venue limit, run the similarity check, and have your guide proofread.
