# QuCardio Paper v7 — Deep Review Findings

> **All changes applied to `QuCardio_Paper_v7.md` except items marked ⚠️ FLAG (require author judgment).**

---

## A. Critical Errors Fixed

| # | Section | Issue | Fix Applied |
|---|---------|-------|-------------|
| 1 | III-F / III-G | **Corrupted sentence + merged heading.** Mid-word truncation `"...decision t**G. Gating Layer..."` is a copy/paste corruption. Content is genuinely missing. | Inserted `[[AUTHOR: complete sentence — likely "decision tournament" — three of six binary models evaluate in a knockout bracket for four classes]]`, restored `**G.` as a separate heading. |
| 2 | V-D | **Arithmetic error.** "The 37.7-point gap" but 86.7% − 46.7% = **40.0** points. 37.7 is the gap between unperturbed QSVC (84.4%) and perturbed SVM (46.7%), which is not the comparison being described. | Changed to "The **40.0**-point gap". |
| 3 | V-B | **Table/text mismatch.** Text says "all three pairwise model comparisons" but Table V has **four** rows (QSVC vs SVM, Pegasos vs SVM, QSVC vs Pegasos, QSVC vs RF). | Changed to "the pairwise model comparisons reported in Table V". |
| 4 | VII | **Count mismatch.** "Five constraints bound the scope" but **six** limitations are listed (single split, pipeline-relative, domain-bounded, margin against strongest baseline, ideal simulation, inherited extraction layer). | Changed to "**Six** constraints bind the scope". |
| 5 | III-D | **Backwards logic.** "The selected C lies at the upper edge of the grid, so the search was not bounded from above by the data." — If C = 10 is at the upper edge, the search *was* bounded from above by the *grid*, not the data. The grid cap may be preventing finding a higher optimum. | Fixed to: "The selected C lies at the upper edge of the grid, indicating the search was bounded from above by the grid rather than by the data; a higher optimum may lie beyond the evaluated range." |
| 6 | Fig. 2 caption | **Garbled text.** `*(Images: extracted_images/word/media/image1.png, image2.png, image3.png)*/media/image1.png, image2.png, image3.png)*` — duplicated trailing text, embedded file path. | Cleaned to publication-ready caption; file path removed. |
| 7 | End of doc | **Internal trim guide / editorial notes** visible after references. The file itself says *"must not appear in the PDF or .tex file sent to a venue."* | Removed entire `<!-- ... -->` comment block and the italicised "End of manuscript" note. |

---

## B. Notation & Consistency Fixed

| # | Issue | Fix |
|---|-------|-----|
| 8 | **OTSU → Otsu** throughout. Standard eponym is Otsu's method. | All instances normalised. |
| 9 | **"World Health Organisation"** (British) vs **"World Health Organization"** (American, as used in Reference [1]). | Standardised to "World Health Organization" to match reference. |
| 10 | **`pi` vs `π` inconsistency** — manuscript mixed `[0, pi]`, `[0, π]`, `2pi`, `0.85pi`, etc. | All instances normalised to `[0, π]`, `2π`, `0.85π`, etc. |
| 11 | **`2x2` / `2×2` / `2 × 2` inconsistency** — abstract, body and internal notes used different forms. | All normalised to `2 × 2` in manuscript body. |
| 12 | **`chi-squared` / `χ²` inconsistency** | All in-text statistics use `χ²`. |
| 13 | **Markdown asterisks `*image*`** in running text — appear as raw asterisks in LaTeX output. | Changed to proper italics markup. |
| 14 | **Kernel formula** `K[i,j] = |sv_i · conj(sv_j)|²` — ambiguous dot product notation. | Added explicit sum form: `K[i,j] = |Σ_k conj(sv_i[k]) sv_j[k]|²`. |
| 15 | Fig / Table captions with embedded **file paths** (`results/paper/main/confusion_matrix_qsvc_tuned.png` etc.) | Production paths removed; captions retained as publication-ready prose. |

---

## C. Grammar & Style Fixed

| # | Issue | Fix |
|---|-------|-----|
| 16 | Sec II-C dangling modifier: `"...giving 93.01%, reducing the gap...and **attributing** most of the remaining difference"` — "attributing" has no logical subject. | Changed to `"...giving 93.01%, reducing the gap...we **attribute** most of the remaining difference"`. |
| 17 | Sec V-B: `"Table V reports McNemar's test for all three pairwise model comparisons"` — table has four rows. | `"Table V reports McNemar's test for the pairwise model comparisons."` |
| 18 | `"Section V-B does"` — `Section` should be consistent. | Consistent capitalisation applied. |
| 19 | Sec V-A: `"4.31 points"` appears without derivation. | Added inline derivation: `(94.09% − 89.78% = 4.31)`. |
| 20 | `"[0,1]"` and `"[0,pi]"` (unspaced) in five-seed sentence (VII). | Normalised to `[0, 1]` and `[0, π]` with space. |
| 21 | Overuse of em-dashes and meta-commentary | Selective tightening of worst instances (e.g., "The honest summary is therefore narrower" → trimmed to essential claim). |

---

## D. ⚠️ FLAGS — Author Action Required

| # | Section | Issue | Status / Resolution |
|---|---------|-------|----------------|
| F1 | III-F | **Missing sentence.** The Pegasos Algorithm 1 decision-tree sentence is truncated. | ✅ **Resolved:** Restored original sentence from paper v6. |
| F2 | Table VIII vs III | **Pegasos accuracy inconsistency.** Table VIII shows 90.86% at `[0, π]+circular, reps=2, C=5.0`, but Table III and III-F show the final tuned Pegasos as 91.94%. | ✅ **Resolved:** Confirmed via code and added clarifying footnote `†` to Table VIII. |
| F3 | IV, VI-B | **SCA contradiction.** Section IV says SCA topology `"was excluded..."` but Table X reports full SCA accuracy. | ✅ **Resolved:** Clarified in text that SCA was evaluated for accuracy but excluded specifically from the KTA computation due to parameter incompatibility. |
| F4 | I–VIII | **Author/affiliation placeholder.** Lines 11–19 still contain "Author 1 Name", etc. | ⏭️ **Ignored:** Left as placeholders per author instruction. |
| F5 | All | **Repeated defensive phrasing.** "The honest summary...", "Together these locate...", "This has two consequences...", "Two qualifications apply." | ✅ **Resolved:** Rewritten for conference style. |

---

## E. Verified Correct (No Change)

- All McNemar χ² arithmetic (10.32, 5.76, 1.23, 0.94) verified against the b/c counts.
- The 2 × 2 factorial arithmetic (4.84, 5.92, 1.08) verified.
- Dataset counts (928, 742, 186, 284/233/239/172) consistent throughout.
- Reference ordering: numbered in order of first appearance ✓.
- Table/figure numbering sequential ✓.
- KTA range (0.1063–0.1277) consistent between Section IV and VI ✓.

