# QuCardio Report — Change Log

Every change made to `QuCardio_Report_Content.md`, with the reason. Numbers, tables, and results were not altered anywhere.

---

## A. Errors I found that neither review caught

### A1. The 4.84 pp figure was attributed to the wrong cause — most important fix

The original Section 2.18 said:

> "scaling the 9-dimensional SVD-compressed ECG features to the range [0, pi] ... improves QSVC accuracy by **4.84 percentage points** over the [0, 1] range"

Table 6.2 does not support this. It shows [0,1] → [0,pi] as **92.47% → 94.62% = +2.15 pp**. The 4.84 figure comes from Table 6.3 (linear → circular entanglement) and also from the overall base-config-to-ours move (89.78% → 94.62%).

So the report was claiming 4.84 pp for the encoding change, 4.84 pp for the entanglement change, and 4.84 pp for both together. Any evaluator who reads the two tables side by side sees this immediately. It appeared in three places: Section 2.18, Section 5.5, and implicitly in Chapter 7.

**Fixed as:** encoding alone = +2.15 pp (entanglement held fixed), entanglement alone = +4.84 pp, combined base → ours = +4.84 pp — explicitly framed as two factors that **interact rather than add**. That is both defensible from the tables and a more interesting finding than three identical numbers.

### A2. Section 2.2 — "Three classifiers were trained" followed by four

The original listed SVM, QSVC, Pegasos **and** QNN under "Three classifiers were trained and evaluated." Changed to "Four models were reported."

### A3. Section 2.5 — garbled complexity notation

"convergence in O(1 per epsilon) steps" → "O(1/epsilon) steps".

### A4. Section 5.4 — wrong class-imbalance numbers

The original justified `class_weight='balanced'` by citing "the 284:172 imbalance" — those are *total dataset* counts. Class weights are computed from the training split, which is 227:137. Corrected.

### A5. Conclusion — "7 percentage points" was rounded loosely

94.62 − 87.63 = 6.99. Changed to 6.99 pp, since every other delta in the report is given to two decimals.

### A6. Section 3.2 — Performance and Usability contradicted each other

Performance capped latency at 5 s; Usability promised results "within 10 seconds." Both rewritten and made consistent (see B1).

---

## B. Fixes from Review 1 (technical)

### B1. NFR vs QSVC inference time — applied

Section 3.2 Performance now separates the classifiers: classical SVM end-to-end under 2 s; QSVC kernel row against 743 support vectors on the order of 10–15 s, framed as acceptable for non-emergency screening. **Marked `[VERIFY]`** — I do not have your timing run, so substitute the measured wall-clock figure. Usability rewritten to promise a progress indicator rather than a 10-second ceiling.

### B2. Table 6.2 / 6.3 inconsistency — applied, and extended

Added an explicit note under Table 6.2 stating all rows use circular entanglement, and that the base paper's exact configuration corresponds to the 89.78% row in Table 6.3.

**Also added a `[VERIFY]` under Table 6.3**, because the reviews assumed 89.78% = [0,1] + linear but the report never states which encoding the entanglement sweep used. If it was run at [0,pi], the baseline is [0,pi] + linear and the Table 6.2 note needs rewording. Check the sweep logs — this is the one genuine ambiguity an evaluator could press on.

### B3. pool1_pool claim — applied (option b)

Deleted the unsupported "richer in locally discriminative texture than the 2048-dimensional GAP output" comparison. Replaced with methodology attribution to Prabhu et al. [1], plus an explicit statement that the layer ablation was not run and remains future work. Added to the Chapter 7 future-work list so the omission reads as scoped rather than overlooked.

### B4. O(1) → O(N·d) — applied in both places

Appeared in Section 4.4 *and* Section 5.6. Both now read O(N·d) with N = 743 and d = 512 stated, and the actual point made explicitly: no repeated circuit simulation.

### B5. Arrhythmia class count — flagged, not changed

Table 4.1 is internally consistent as written (188 + 46 = 234, column totals reach 929/743/186), so I left the numbers alone rather than introducing an inconsistency. Added a `[VERIFY]` block with the exact command to run and what changes if the answer is 233.

### B6. Section 2.2 — linear entanglement stated up front — applied

The literature survey now states the base paper's [0,1] range *and* linear entanglement in the same sentence, setting up the ablation as the obvious next step rather than revealing it in Chapter 5.

### B7. Pegasos numerical-instability evidence — applied

The explanation (kernel value ≈ 1/512 → step size η_t = 1/(λt) collapses) now sits immediately after the claim, in Section 5.6, rather than trailing it.

### B8. KNN = SVM identical accuracy — applied

Added the note that macro F1 differs (83.61% vs 84.10%), so the two models misclassify different samples. Also added a line on Random Forest at 91.94%, which the original left unexplained.

---

## C. Fixes from Review 2

- **Sections 1.4, 3.1, 5.3** now state the MinMax → ×π step explicitly, so the [0,π] range is consistent from Chapter 1 through Chapter 6.
- `pool1\_pool` escapes **kept**, since your team note says you are compiling in the LaTeX Main.pdf template.
- Repeated paragraph openers removed throughout (see D2).

**Not applied — "Ensure your backend implements DAGSVM rather than 6-model voting."** That is a code question, not a report question. Section 5.6 describes Algorithm 1 evaluating three of six models. If your `predict_algorithm1()` actually does a full six-model vote, fix the code or fix the sentence — but check before submitting, because it is a claim about your own implementation.

---

## D. Prose changes

### D1. Rewrote the Chapter 7 opening

The old opening was "This project designed, implemented, validated, and deployed a complete hybrid..." — four verbs, no content.

The new opening tells what actually happened: the replication produced 89.78% against the paper's 94.09%, the gap did not close under tuning, and the ablations traced it to two interacting factors. **This is grounded entirely in numbers already in your report** — 89.78% is in Table 6.3, 94.09% is in Table 6.1. It is not an invented narrative.

### D2. Removed repeated openers

"This project" appeared 11 times; it now appears 0 times. "This work", "This finding", "This result", "This paper" as sentence openers: 0 each. Replaced with "QuCardio", "we", "our pipeline", "the system", or restructured so the sentence starts with its subject.

### D3. Broke the paragraph-closing summary habit

Many paragraphs ended with a clinching "This X demonstrates/confirms/prevents Y" sentence. Several now end on the evidence itself. Sections 2.9, 2.11, 2.13, 2.16, 4.2, 5.1, 6.1 and 6.6 are the main ones.

### D4. Varied sentence length

Long analytical sentences are now interleaved with short ones. The 45-word run-on in Section 2.18 is split into three. Same treatment in Sections 6.3 and 6.7.

### D5. Broke the parallel step list in Section 5.1

Steps 1–7 all opened identically ("The input is...", "A second OTSU...", "A morphological..."). Step 4 now opens "Horizontal grid lines were handled the same way...", and a substantive sentence was added explaining why the trace survives both subtractions.

### D6. Fixed tense in Chapter 5

Chapter 5 mixed present and past. It is now consistently **past tense for what you did** ("the pipeline applied seven steps", "extraction ran in batches of 8"), with present tense retained only for how the deployed system behaves.

### D7. Loosened the triadic lists

"designed, implemented, validated, and deployed" and similar constructions were reduced or rewritten throughout.

---

## E. Overclaiming — softened

These would have been the easiest things for an examiner to attack:

| Original | Changed to |
|---|---|
| "the **first** McNemar significance test" | "to the best of our knowledge, the first" |
| "**No prior** ECG-QML work has reported this analysis" | "we did not find this analysis reported in any of the papers surveyed in Chapter 2" |
| "one of the first formal statistical validations **in the published literature**" | scoped to the surveyed literature |

I also added a paragraph to Section 6.4 noting that QSVC vs Pegasos is **not** statistically significant (p = 0.267) despite the 2.68 pp accuracy gap. Reporting a null result you already computed strengthens the chapter considerably and is the kind of thing an examiner rewards.

Section 6.7 now opens by saying plainly that this is where the quantum advantage reverses, rather than easing into it.

---

## F. Two review suggestions I deliberately did NOT follow

### F1. "Add localized details — e.g. 'Intel Core i5 processor in SJEC Lab 3'"

Do not do this. It puts an unverifiable claim about your experimental hardware into a technical report. If an examiner asks what machine you trained on and the answer does not match, that is a far worse problem than a stylistic flag. Your Section 3.3 hardware table already states the requirement honestly — if you want real specificity, replace it with the machine you actually used.

### F2. "Add 3–4 informal observations: 'We were surprised to find that...'"

Same reasoning. Writing that you were surprised when you were not is fabricating your research process, and it is the kind of thing that falls apart in a viva when someone asks a follow-up.

What I did instead is make the prose specific and varied on its own terms. Where the report contains a genuinely notable result — Y+YY landing below the unentangled ZFeatureMap, the quantum advantage reversing on Dataset 2 — the revision now says so directly and admits where the mechanism is not established ("we have not isolated the cause and would not claim more than that from a single dataset"). That reads as a person thinking, because it is a person thinking, not a performance of one.

---

## G. Three things to do yourself before submitting

1. **Resolve the three `[VERIFY]` tags.** They are the only places where I could not check the answer against your code.
2. **Read Chapter 5 aloud.** It is the chapter with the most of your own implementation detail in it, and it is where your own phrasing will naturally push out anything left over.
3. **Add specifics only you have.** Which team member built which module, the actual training machine, wall-clock timings, a dead end you tried that did not work. Real detail you can defend is worth more than any amount of rewording.
