# Technical Paper Writing
### Writing Effective Research Papers for Publication

**Worked example used throughout:** M. S. H. Sumon et al., "Advancing Brain Tumor MRI Classification with Interpretable Attention-Enhanced Deep Learning," *Proc. 2025 IEEE Int. Women in Engineering (WIE) Conf. Electrical and Computer Engineering (WIECON-ECE)*, Cox's Bazar, Bangladesh, 2025, pp. 539–544.

> Sections marked **[+]** are additions not in the original slide deck.

---

## 1. Research Paper vs. Project Report

Every new project is not necessarily a research paper. The difference is **novelty** and **rigour**.

| Project / Report | Research Paper |
|---|---|
| Documents that a system was built and works | Expects novelty — a new algorithm, model, or theoretical insight |
| Describes implementation and outcomes | Justifies the quality of the contribution, not just the build |
| No obligation to advance the state of the art | Requires rigorous validation against published baselines |
| Success = the deliverable functions | Demonstrates measurable improvement or proves a hypothesis |

**In our example:** The paper introduces an attention-enhanced DenseNet201 (novelty) and validates it on brain-tumor MRI against CNN and transformer baselines with statistical tests (rigour) — that is what makes it a paper, not a report.

### [+] Types of contribution that count as novelty
Not every paper needs a new architecture. A valid contribution can be any of:
- **Algorithmic** — a new method, modification, or optimisation
- **Empirical** — a large-scale comparison or benchmark that did not exist before
- **Dataset** — a new, well-documented, properly licensed dataset
- **Systems** — a deployment that meets constraints nobody had met (latency, energy, cost)
- **Survey / review** — a structured synthesis of a field, with a taxonomy
- **Negative or replication result** — showing a widely cited method fails to generalise

---

## 2. How to Turn a Project into a Research Paper

**1. Identify novelty.** Do not just apply an existing ML model or web stack. Introduce an algorithmic modification, a novel data pipeline, or a unique optimisation strategy.

**2. Perform benchmarking.** Compare your solution against existing, *published* baselines using standard metrics — accuracy, latency, throughput, energy efficiency.

**In our example:** The novelty is fusing DenseNet201 with Multi-Head Self-Attention plus Grad-CAM interpretability; the benchmarking pits it against InceptionV3, ViT and TinyViT on one MRI dataset — a template for both moves.

### [+] Practical steps to make the jump
1. **Find the gap first.** Read 15–25 recent papers in the sub-domain. Write down what each one *cannot* do. Your paper answers one of those.
2. **Freeze the research question** into one sentence: *"Can X improve Y under constraint Z?"* If you cannot write it, you do not have a paper yet.
3. **Use a public, citable dataset** wherever possible — reviewers distrust private data with no access statement.
4. **Reproduce at least one baseline yourself.** Quoting a number from another paper on a different split is not a fair comparison.
5. **Run each experiment multiple times** with different random seeds; report mean ± standard deviation, not a single lucky run.
6. **Decide the target venue before writing** — it fixes the page limit, template, and expected depth.

---

## 3. Parts of an Effective Paper — IMRaD

IMRaD = the four core sections of a standard scientific paper: **I**ntroduction, **M**ethods, **R**esults, **a**nd **D**iscussion.

1. **Abstract** — 250–300 word summary: goal, methods, results, conclusion.
2. **Introduction** — background, literature, the research gap, and your objective.
3. **Methodology** — materials, data, and procedures — detailed enough to reproduce.
4. **Results & Discussion** — data in charts/tables, then interpretation of what it means.
5. **Conclusion** — recap of key points, limitations, and future directions.

### [+] The full front-to-back order of a submitted manuscript
Title → Authors and affiliations → Abstract → Keywords/Index Terms → Introduction → Related Work → Methodology → Experimental Setup → Results → Discussion → Ablation → Limitations → Conclusion & Future Work → Acknowledgements → References → Appendix (optional)

### [+] Recommended writing order
Write **Methodology → Results → Introduction → Related Work → Conclusion → Abstract → Title** last. The abstract and title are easiest once you know what the paper actually proved.

---

## 4. The Abstract

A concise 250–300 word summary of the whole paper — a reader's quick overview.

- State the main goal, context, or central problem
- Summarise the methods, models, or algorithms used
- Highlight any novel technique or framework applied
- Present the primary quantitative or qualitative results
- State the main takeaway and its significance
- Show how the work addresses the core gap

**In our example:** task → attention-enhanced DenseNet201 method → 99.66% test accuracy → clinical significance.

### [+] A sentence-by-sentence abstract formula
1. **Context** (1–2 sentences) — why the problem matters.
2. **Gap** (1 sentence) — what existing work fails to do.
3. **Proposal** (1–2 sentences) — what you built, named explicitly.
4. **Method** (1–2 sentences) — dataset, setup, evaluation protocol.
5. **Results** (1–2 sentences) — hard numbers, and the delta over the best baseline.
6. **Significance** (1 sentence) — what this enables.

### [+] Abstract rules
- No citations, no figures, no undefined abbreviations, no equations.
- Never use phrases like "results are promising" without a number.
- Must stand alone — it will be read in isolation on IEEE Xplore, Scopus, and Google Scholar.

---

## [+] 5. Title and Keywords

**Title**
- Specific, not generic — *"Attention-Enhanced DenseNet201 for Interpretable Brain Tumor MRI Classification"*, not *"A Deep Learning Approach"*.
- Ideally names the method, the task, and the differentiator.
- 10–15 words; avoid questions, puns, and unexplained acronyms.

**Keywords / Index Terms**
- 4–6 terms; these drive indexing and discoverability.
- Mix broad (*deep learning*) with specific (*multi-head self-attention*, *Grad-CAM*, *medical imaging*).
- Do not repeat words already in the title if you can add new coverage instead.
- For IEEE, prefer terms from the **IEEE Thesaurus**.

---

## 6. Introduction: What · Why · How

Open broad, then narrow to your specific challenge.

- **What?** What is the paper about? Give precise context and the essential terminology and concepts.
- **Why?** Why is this topic relevant now? Offer novel insight and name the problem your research solves.
- **How?** How did you do the research? Preview the key elements in order.

Start from a broad technical context before focusing on the gap.

**In our example:** the Introduction opens with the global burden of brain tumors, then narrows to the gaps — interpretability and real-time efficiency.

### [+] The funnel, in paragraphs
1. Broad domain and its importance (with citations)
2. Current approaches and what they achieved
3. The specific limitation that remains
4. Your proposal, in one sentence
5. Bulleted contributions
6. Paper roadmap

---

## 7. Introduction: Stating Your Contributions

End the introduction with an explicit map of the paper.

- Include a "map" of what will be discussed, in order.
- Use a short bulleted list to state 2–3 major contributions explicitly.
- Close with a paragraph on paper structure — e.g. "Section II covers related work, Section III the proposed architecture, Section IV the results…"

**In our example:** the Introduction states the proposed contribution — an attention-enhanced DenseNet201 (MHSA) that targets those gaps.

### [+] Writing contributions well
- Begin each with a verb: *We propose… We introduce… We demonstrate… We release…*
- Make each one **verifiable** — a reviewer should be able to point to the section that proves it.
- Attach a number where you can: *"…improving F1 by 3.2 points over the strongest baseline."*
- Three strong contributions beat six weak ones. Do not pad.

---

## 8. Literature Review / Related Work

Survey the field to justify why your paper is needed.

- Introduce the sub-domain and define key concepts and frameworks
- Highlight baseline models, standard datasets, and SOTA benchmarks
- Group prior work chronologically, thematically, or by method
- Evaluate strengths and limitations (cost, scalability, generalisation)
- State what is missing or under-explored — this justifies your paper
- Differentiate your solution from existing baselines

**In our example:** Related Works summarises each prior study with its method and reported accuracy.

### [+] Doing it properly
- **Synthesise, do not list.** "Author A did X. Author B did Y. Author C did Z." is a bibliography, not a review. Group by approach and compare them.
- **End every group with a critique** that points toward your gap.
- **A comparison table** (method / dataset / metric / reported result / limitation) is the single highest-value element you can add here.
- **Prefer recent work** — for fast-moving fields, most citations should be from the last 3–5 years, alongside the foundational classics.
- **Search systematically:** IEEE Xplore, ACM DL, ScienceDirect, SpringerLink, PubMed, Google Scholar, Semantic Scholar, Connected Papers. Track everything in Zotero or Mendeley from day one.
- **Be fair to prior work.** Reviewers are often the authors of the papers you are criticising.

---

## 9. Methodology

Detailed enough that another researcher could reproduce the study.

- **Data & preprocessing:** datasets, cleaning, normalisation, augmentation, train/val/test splits
- **Proposed algorithm:** equations, pseudo-code, or algorithmic steps unique to your work
- Explain modifications to existing models and novel design choices
- State hyperparameters, training configs, baselines, and metrics

**In our example:** Section III begins with data collection and the overall pipeline; the proposed model is given as a labelled block diagram (Fig. 4, captioned below).

### [+] The reproducibility checklist
- Dataset name, version, source URL, licence, and class distribution
- Exact split sizes and whether the split is stratified or subject-wise
- Preprocessing pipeline in order, with parameter values
- Model architecture: layer sizes, activation functions, parameter count
- Optimiser, learning rate and schedule, batch size, epochs, early-stopping criterion, loss function
- Hardware (GPU model, RAM), framework and version, total training time
- Random seeds and number of runs
- A link to a code repository — even an anonymised one during review

### [+] Equations and notation
- Number every equation; refer to them as "(1)", "(2)" in the text.
- Define every symbol the first time it appears; keep one symbol per concept.
- Punctuate equations as part of the sentence — they are grammatical objects.
- Use pseudo-code (algorithm environment) for procedures, not prose paragraphs.

---

## 10. Results and Discussion

Present the empirical outcomes, then interpret what they mean.

- State the evaluation metrics (accuracy, F1, execution time, throughput, memory)
- Present findings in well-formatted tables, charts, or graphs
- Interpret in depth: why did the method behave this way?
- Relate results to your hypothesis and to prior studies

**In our example:** training/validation curves show stable convergence; a results table compares every model's accuracy, precision, recall, and F1.

### [+] Results vs Discussion — keep them distinct
| Results | Discussion |
|---|---|
| What the numbers are | What the numbers mean |
| Objective, descriptive | Interpretive, argued |
| No speculation | Mechanisms, comparisons, implications |

### [+] Choosing metrics
- **Accuracy alone is misleading on imbalanced data.** Report precision, recall, F1 (macro and weighted), AUC-ROC, and a confusion matrix.
- Report **efficiency** too: parameters, FLOPs, inference latency, memory — especially if you claim real-time or edge suitability.
- Include **statistical significance**: paired t-test, McNemar's test, Wilcoxon signed-rank, Cohen's κ, or confidence intervals. A 0.3% gap with no significance test is not a result.
- **Never report a number you cannot reproduce**, and never round upward in your favour.

### [+] Common results-section failures
- Reporting test-set accuracy that was used for model selection (leakage)
- Comparing against baselines you tuned less carefully than your own method
- Charts without axis labels, units, or legends
- Claiming "state of the art" without checking the current leaderboard

---

## 11. Ablation Studies & Limitations

Isolate what each component contributes — and be honest about weaknesses.

- Ablate components to show the isolated contribution of each module
- Vary key hyperparameters (learning rate, batch size, thresholds) and report the effect
- Honestly discuss where the approach underperforms or hits trade-offs
- Give critical insight into constraints to motivate future work

**In our example:** a comparative-evaluation table (AUC, Cohen κ, McNemar) benchmarks every model, caption on top.

### [+] How to design an ablation
Start from the full model and remove **one** component at a time, keeping everything else fixed. A clean ablation table has one row per configuration and shows the metric dropping as each piece is removed — that is what proves the component earns its place.

### [+] Writing limitations without weakening the paper
State them plainly and pair each with a mitigation or a future direction: *"Evaluation is limited to a single-institution dataset; cross-site validation is required before clinical deployment."* Reviewers punish hidden weaknesses far more than acknowledged ones.

---

## 12. Conclusion

- Remind the reader of the central problem or thesis
- Synthesise the primary empirical or theoretical results
- State how you resolved the gap or improved on baselines
- Explain the broader significance of the findings
- State the limitations (dataset size, hardware, edge cases)
- Outline concrete future work — new datasets, efficiency, integration

**The mirror of your abstract.** The abstract promises; the conclusion delivers. Every claim you opened with should be answered here — with results, limits, and a path forward.

> **Tip:** never introduce a brand-new result in the conclusion.

### [+] Also
- Do not simply copy sentences from the abstract — rewrite at a higher level.
- Make future work **specific**. "We will explore other domains" is filler; "We will validate on the multi-site BraTS 2023 cohort and quantise the model for Jetson deployment" is a plan.

---

## 13. Citations & References

Each claim in running text must be tied to its source.

### Styles covered
- **APA** — author–date, hanging indent
- **MLA** — "Works Cited," author–page
- **IEEE** — numbered `[n]` citations; the style used by the worked example and most engineering venues

### Best practice

**Do**
- Cite the original source, not a second-hand summary
- Match every in-text citation to a reference entry, and vice-versa
- Keep one style (APA / IEEE / MLA) consistent throughout
- Use a reference manager — Zotero, Mendeley, or LaTeX + a `.bib` file

**Don't**
- Cite material you have not actually read
- Leave referencing to the final hour (missing DOIs, wrong names)
- Rely on non-peer-reviewed preprints (arXiv, SSRN) as primary sources

**In our example:** a real IEEE reference list — numbered, consistent, complete.

### IEEE referencing in Microsoft Word
Word's built-in reference manager can generate and update IEEE-numbered citations automatically.
Walkthrough: <https://medium.com/academicianhelp/ieee-referencing-using-microsoft-word-66c855181d64>

**Preferred for LaTeX users:** store references in a `.bib` file and let BibTeX / biblatex format them.

### [+] Extra rules
- IEEE numbers citations **in order of first appearance**, starting at [1].
- Cite as part of the sentence: *"Prior work [4] showed…"* — not *"[4] showed…"* at the start of a sentence in most styles.
- Include the **DOI** where available; check author names against the publisher page, not Google Scholar's auto-generated BibTeX (it is frequently wrong).
- Avoid excessive self-citation and citation padding — reviewers notice.
- Do not cite Wikipedia, blog posts, or course slides as evidence for a technical claim.

---

## 14. Labelling Figures & Tables

- **Figures** are labelled at the **bottom** (caption below the graphic)
- **Tables** are labelled at the **top** (caption above the table)
- Position them at the top or bottom of columns — never mid-column
- Refer to every figure/table in the text before it appears
- Use words, not just symbols, on axes — and include units

**In our example:** figures are captioned underneath and tables on top — exactly these two rules.

### [+] Figure quality
- Use **vector formats** (PDF/EPS/SVG) for plots; 300+ DPI for raster images.
- Font size inside a figure should roughly match the body text — shrink the figure, not the labels.
- Make figures legible in **greyscale** and distinguishable for colour-blind readers (vary line style and marker shape, not just colour).
- Captions should be self-contained: a reader skimming only figures should still understand the result.
- IEEE numbering: **Fig. 1, Fig. 2…** and **TABLE I, TABLE II…** (tables use Roman numerals).

---

## [+] 15. Choosing Where to Publish

**Conference vs journal**
- *Conference:* faster (weeks to a few months), fixed deadline, 4–8 pages, requires attendance/presentation and a registration fee. Good for a first paper.
- *Journal:* slower (6–18 months), no page pressure, deeper review, higher prestige.

**How to evaluate a venue**
- Indexed by **Scopus / Web of Science / IEEE Xplore / DBLP**?
- Published by a recognised body — IEEE, ACM, Springer, Elsevier?
- Real programme committee with named, findable academics?
- Sensible acceptance rate and a genuine review timeline?

**Predatory-venue warning signs**
- Unsolicited flattering email invitations
- Promises of acceptance in days, or "guaranteed publication"
- Fees demanded before review
- Vast, unrelated scope ("Engineering, Medicine, Management and Arts")
- No clear peer-review description
- Check against Beall's list heuristics, DOAJ, and Think–Check–Submit.

**Also check before writing a single page:** the **Call for Papers** — scope, page limit, template (IEEE `IEEEtran` LaTeX / Word template), blind vs non-blind review, submission deadline, and camera-ready date.

---

## [+] 16. Academic Writing Style

- **Tense:** present tense for established facts and for describing your paper ("Section III presents…"); past tense for what you did and found ("We trained…", "The model achieved…").
- **Voice:** "We propose…" is standard and accepted in engineering venues. Avoid heavy passive constructions that hide who did what.
- **Precision over flourish.** Cut "very", "quite", "significantly" (unless statistically significant), "novel" repeated in every paragraph.
- **One idea per paragraph**, with a topic sentence first.
- **Define every abbreviation once**, at first use, then use it consistently.
- **Hedge appropriately.** "The results suggest" is safer and more credible than "This proves".
- **Consistency:** one spelling convention (US or UK), one symbol per variable, one name for your method throughout.
- **Avoid overclaiming.** "Outperforms all existing methods" invites a reviewer to find a counterexample.

---

## [+] 17. Ethics, Integrity, and Disclosure

- **Plagiarism:** run a similarity check (Turnitin / iThenticate); most venues want under ~15% with no single source dominant. Paraphrasing without citation is still plagiarism.
- **Self-plagiarism:** reusing your own earlier text or results without citation is a violation.
- **Authorship:** everyone listed must have contributed intellectually; everyone who contributed must be listed. Agree on order early to avoid disputes.
- **Data ethics:** human-subject or patient data needs ethics-board approval and consent; state the approval reference in the paper.
- **AI-tool disclosure:** most publishers (IEEE, Elsevier, Springer) now require disclosure of generative-AI assistance in writing, and forbid listing an AI as an author.
- **Conflicts of interest and funding** must be declared.
- **No duplicate submission.** Submitting the same manuscript to two venues simultaneously is grounds for a ban.
- **Data/code availability statement** — increasingly expected, and it strengthens the paper.

---

## [+] 18. From Submission to Publication

1. **Format to the template** — IEEE two-column, correct margins, no page-limit violation.
2. **Submit** via the venue's system (EDAS, CMT, EasyChair, ScholarOne) and keep the manuscript ID.
3. **Peer review** — typically 2–4 reviewers.
4. **Decision:** Accept / Minor revision / Major revision / Reject / Reject-and-resubmit.
5. **Respond to reviewers** with a point-by-point rebuttal letter: quote each comment, state the change, give the page/line. Be courteous even when a reviewer is wrong — explain with evidence.
6. **Camera-ready** — apply final formatting, pass IEEE PDF eXpress, sign the copyright form.
7. **Register and present** — for conferences, at least one author must register and present, or the paper is withdrawn from the proceedings.
8. **After publication** — add the DOI to ORCID, Google Scholar, ResearchGate, and your résumé.

**Most common reasons papers get rejected**
- No clear novelty, or novelty not articulated
- Weak or unfair baselines
- Insufficient evaluation (one dataset, one run, no significance testing)
- Poor writing that obscures the contribution
- Out of scope for the venue
- Missing related work the reviewer considers essential
- Template, page-limit, or anonymity violations

---

## [+] 19. Pre-Submission Checklist

**Content**
- [ ] The contribution is stated in one sentence, and the paper proves it
- [ ] Every claim in the abstract is supported in the body
- [ ] Baselines are recent, published, and fairly tuned
- [ ] Ablation study present
- [ ] Limitations stated honestly
- [ ] Statistical significance reported

**Presentation**
- [ ] Every figure and table is referenced in the text before it appears
- [ ] Figure captions below, table captions above
- [ ] All abbreviations defined at first use
- [ ] All equations numbered, all symbols defined
- [ ] Consistent notation and method name throughout

**References**
- [ ] Every in-text citation has a reference entry and vice-versa
- [ ] One consistent style
- [ ] DOIs included, author names verified
- [ ] Majority of citations recent

**Compliance**
- [ ] Correct template, within page limit
- [ ] Anonymised if the venue is double-blind
- [ ] Similarity check passed
- [ ] Ethics/funding/AI-use statements included
- [ ] Proofread by someone who did not write it

---

## 20. Sample Paper — the Worked Example, End to End

> [1] M. S. H. Sumon et al., "Advancing Brain Tumor MRI Classification with Interpretable Attention-Enhanced Deep Learning," in *Proc. 2025 IEEE Int. Women in Engineering (WIE) Conf. Electrical and Computer Engineering (WIECON-ECE)*, Cox's Bazar, Bangladesh, 2025, pp. 539–544.

Sections modelled: **Abstract · Related Work · Methodology · Results · Evaluation · References**

*One well-chosen paper models every section you will write. Read it the way you want yours to be read.*

---

## [+] 21. Tools Worth Setting Up

| Purpose | Tools |
|---|---|
| Writing | Overleaf (LaTeX, `IEEEtran`), MS Word with IEEE template |
| References | Zotero, Mendeley, JabRef, BibTeX/biblatex |
| Finding papers | IEEE Xplore, ACM DL, Semantic Scholar, Connected Papers, Litmaps |
| Figures | Matplotlib/Seaborn, draw.io, Inkscape, TikZ |
| Grammar | Grammarly, LanguageTool, Writefull (academic-specific) |
| Similarity | Turnitin, iThenticate |
| Reproducibility | GitHub + a README with exact commands, Weights & Biases, Docker |
