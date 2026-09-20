# QuCardio — Known Issues, Solutions and Roadmap

Working document. Covers two open technical problems, dashboard work that would help a demo or a clinical pilot, and the feature additions that would raise the novelty of the next paper.

---

## 1. Problem: the gatekeeper accepts line-art diagrams as ECGs

### What is happening

A UML sequence diagram, a line chart or similar line art passes the MobileNetV2 gatekeeper and reaches the classifier, which then returns a confident cardiac diagnosis for it.

This is not a threshold that needs adjusting. It is a training distribution problem, and no amount of tuning the decision threshold will fix it, because the model has genuinely never been given a reason to separate the two.

Think about what the gatekeeper was trained on: ECG images as positives, and natural photographs, faces, cars and screenshots as negatives. From those examples the cheapest feature that separates the classes is roughly "thin dark strokes on a light background, sparse ink, long horizontal runs, near-zero colour saturation". A sequence diagram matches every part of that description. So does a line chart, sheet music, an architectural drawing and graph paper. The model is not malfunctioning; it learned exactly what the data taught it, and the data never contained a line drawing that was not an ECG.

The heuristic fallback has the same blind spot. Aspect ratio, colour saturation and dark-pixel row coverage are all satisfied by a sequence diagram. The column-oscillation check is the only criterion that could separate them, and a diagram with many vertical lifelines produces enough column-wise variation to pass it.

### Fix 1 — hard negatives (do this first, largest effect for the effort)

Retrain the gatekeeper with the negative class dominated by things that look like an ECG but are not:

| Negative category | Source | Suggested count |
|---|---|---|
| UML diagrams (sequence, class, flowchart) | PlantUML / Mermaid rendered to PNG in bulk | 300–500 |
| Line and area charts | Matplotlib, random data, axes on and off | 300–500 |
| Sheet music | IMSLP public domain scans | 200 |
| Graph paper, engineering drawings, blank grids | Render or scrape | 200 |
| Handwritten notes, tables, forms | Any open document dataset | 200 |
| Seismograph and other non-cardiac waveform traces | Search or synthesise | 150 |
| Original negatives (photos, faces, screenshots) | Existing set | keep |

Generating the first two categories is scriptable in under an hour and they are the two that actually cause the failure. Rendering 400 PlantUML diagrams with randomised participant counts, message counts and note blocks is a short loop.

Two training details matter. Keep the positive-to-negative ratio near 1:2, not 1:1, since false accepts are the failure that hurts. And apply the *same* preprocessing at gatekeeper training time that the deployed path applies, otherwise the model is evaluated on a different distribution to the one it was trained on.

### Fix 2 — a periodicity check the gatekeeper cannot fake

This is the discriminative signal a sequence diagram does not have, and it is worth adding as a second gate regardless of how the retraining goes. A real ECG is quasi-periodic: the R-peaks repeat at roughly 0.6–1.2 s intervals, which on a standard 25 mm/s print is a repeating structure every 15–30 mm horizontally. A sequence diagram has vertical structure but no dominant horizontal period.

```python
import numpy as np, cv2

def ecg_periodicity_score(gray):
    """Returns (score, period_px). Real ECG strips score high; line art scores low."""
    # ink profile per column, after removing DC
    ink = (gray < 128).astype(np.float32).sum(axis=0)
    ink = ink - ink.mean()
    if ink.std() < 1e-6:
        return 0.0, 0

    # normalised autocorrelation of the column profile
    ac = np.correlate(ink, ink, mode='full')[len(ink)-1:]
    ac = ac / (ac[0] + 1e-9)

    # ignore trivial lags; look for a beat in a plausible band
    lo, hi = max(8, len(ink)//60), max(20, len(ink)//4)
    band = ac[lo:hi]
    if band.size == 0:
        return 0.0, 0
    k = int(np.argmax(band))
    return float(band[k]), lo + k
```

On the training set, compute this score for all 928 ECGs and for the negative set, then pick a cut-off at roughly the 2nd percentile of the ECG distribution. In practice genuine ECG strips land around 0.35–0.7 and line art lands below 0.15, so the separation is wide and the threshold is not delicate.

Two more cheap signals worth combining with it:

- **Row-band energy.** An ECG concentrates its ink in horizontal bands (the leads). Project ink onto rows, and check that the top 40% of rows by ink hold most of the total. Diagrams spread ink more evenly.
- **Stroke-angle histogram.** Sobel gradients on an ECG give a broad distribution of edge orientations, because the trace is curved. A UML diagram is dominated by exactly 0° and 90°. If more than about 70% of gradient energy sits within ±5° of the two axes, it is line art.

### Fix 3 — fail safe at the output, not only at the input

Even with a better gate, something unexpected will get through. Two guards:

1. **Confidence floor with abstention.** The system already flags predictions below 55%. Add a hard abstention band: below 40%, return "unable to classify — verify the image is a standard 12-lead ECG print" instead of a class name. An out-of-distribution input usually produces a flat probability vector, so this catches a large share of what the gate misses.
2. **Feature-space distance check.** You already have the 742 training feature vectors. At inference, compute the Mahalanobis distance from the new 9-D vector to the training distribution, and reject beyond a percentile threshold calibrated on the training set. This costs microseconds, needs no retraining, and is the same mechanism that would have caught the PTB-XL problem in Section 2. It is the single highest-value addition on this list because it protects against *every* out-of-distribution input, not just line art.

### How to write it up

Do not claim the gatekeeper is solved. In the report this is already recorded in Section 6.3 and Section 3.7 and listed in Section 8.2. In the paper, one sentence in Limitations is enough. A self-reported failure mode with a stated remedy costs nothing; one a reviewer finds themselves costs the paper.

---

## 2. Problem: accuracy collapses on PTB-XL signals rendered with Matplotlib

### What is happening

This is the same failure as the cross-dataset result in Section 7.7, not a separate bug — and recognising that is useful, because it means you already have a measured, published-quality characterisation of it.

The models were trained on photographs and scans of paper ECG prints from one hospital. A Matplotlib rendering of a PTB-XL signal is a different kind of image in almost every respect that your pipeline is sensitive to:

| Property | Training data (Khan et al.) | Matplotlib render |
|---|---|---|
| Grid | Printed red/pink clinical grid | None, or a thin grey grid |
| Trace | Ink on paper, variable width, bleed | Uniform 1 px anti-aliased line |
| Background | Off-white, uneven lighting, shadows | Pure white, perfectly flat |
| Margins | Tight crop around printed strip | Large whitespace, axes, tick labels, title |
| Scaling | 25 mm/s, 10 mm/mV, fixed by the machine | Whatever the default axes chose |
| Lead layout | 12-lead 3×4 grid plus rhythm strip | Usually a single lead, or stacked subplots |
| Noise | Scan artefacts, staples, skew | None |

Your preprocessing pipeline then makes this worse rather than better. The grid-removal morphology has no grid to remove, so it attacks the trace itself; the contour crop latches onto the axes box rather than the waveform; and the OTSU polarity check behaves differently on a pure-white background. By the time ResNet50 sees the image, it is far outside the distribution the SVD was fitted on, and the nine compressed components land in a region the support vectors do not cover.

### Fix 1 — render PTB-XL to imitate the training distribution

This is the right first move, it requires no retraining, and it is testable in an afternoon. Do not feed raw Matplotlib output. Render deliberately:

```python
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def render_ecg_like_print(signal, fs=500, seconds=10, mm_per_s=25, mm_per_mV=10, dpi=200):
    """Render a PTB-XL lead to imitate a clinical paper print."""
    n = int(seconds * fs)
    sig = signal[:n]
    width_mm, height_mm = seconds * mm_per_s, 40          # 250 mm x 40 mm strip
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])                        # no margins at all

    t = np.arange(n) / fs
    # clinical grid: 1 mm minor, 5 mm major, pink on white
    ax.set_xlim(0, seconds); ax.set_ylim(-2, 2)
    for step, colour, lw in ((1/mm_per_s, "#f4c7c3", 0.4), (5/mm_per_s, "#e88b84", 0.8)):
        ax.set_xticks(np.arange(0, seconds + step, step), minor=(lw < 0.6))
    ax.grid(which="both", color="#f0b0ab", linewidth=0.5)

    ax.plot(t, sig, color="black", linewidth=1.1, solid_joinstyle="round")
    ax.set_xticklabels([]); ax.set_yticklabels([])
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0)
    return fig
```

The parameters that matter, in order of importance:

1. **No axes, no labels, no title, no margins.** `fig.add_axes([0,0,1,1])`. The contour crop in Step 1 of your pipeline must find the waveform, not an axes box.
2. **A printed-style grid in the right colour.** Your grid-removal morphology expects a grid. Give it one, and let it do its job. Removing the grid at render time instead is the second-best option, but then you are also skipping two steps of the pipeline, which changes the distribution again.
3. **Correct physical scaling.** 25 mm/s horizontally and 10 mm/mV vertically. This is what makes the QRS complex the right shape relative to the image, and shape is what `pool1_pool` responds to.
4. **Trace width around 1.0–1.3 px at 200 dpi**, not the Matplotlib default of 1.5 at 100 dpi, which renders too thick relative to the grid after resizing.
5. **Multi-lead layout if you want to match closely.** The Khan images are full 12-lead prints in a 3×4 arrangement. A single-lead strip is a different image class. Rendering the same 3×4 plus rhythm strip is the closest match available.

Then push the render through your existing seven-step preprocessing and compare the 340×340 output side by side with a real one. If they do not look alike to your eye, the classifier will not treat them alike either. Iterate on the render until they do. This visual check is the whole method, and it is faster than any metric.

### Fix 2 — quantify the shift before trying to fix accuracy

Before changing any model, measure where the rendered images land:

```python
# after SVD + MinMax, for train set and for rendered PTB-XL set
mu, cov = X_train.mean(0), np.cov(X_train.T) + 1e-6*np.eye(9)
inv = np.linalg.inv(cov)
d_train = [np.sqrt((x-mu) @ inv @ (x-mu)) for x in X_train]
d_ptb   = [np.sqrt((x-mu) @ inv @ (x-mu)) for x in X_ptb]
```

If the PTB-XL distances sit far above the training distribution, the problem is the render and Fix 1 applies. If they overlap but accuracy is still poor, the problem is label semantics (see Fix 4) and no amount of render tuning will help. Knowing which of these two situations you are in saves days.

### Fix 3 — domain adaptation, in increasing order of cost

If the render matching gets you partway but not far enough:

1. **Refit only the scaler.** Cheapest possible adaptation. Keep SVD and the classifier, refit MinMax on a small labelled PTB-XL sample. Often recovers a surprising amount, since much of the shift is a global scale and offset.
2. **CORAL alignment.** Align the second-order statistics of the PTB-XL 9-D features to the training features with a whitening-then-recolouring transform. Roughly ten lines of NumPy, no labels required, no retraining.
3. **Fine-tune ResNet50 on mixed domains.** Unfreeze the last block and train on Khan plus rendered PTB-XL with a domain-adversarial or simple mixed-batch objective. This is the item already listed as future work in the report, and it is the one that would genuinely fix cross-domain behaviour.
4. **Train on rendered signals from the start.** If you can render PTB-XL at scale, you have access to far more than 928 images. A model trained on 10,000 rendered ECGs and evaluated on the Khan scans is a different and arguably stronger project than the current one.

### Fix 4 — check the labels actually mean the same thing

Worth confirming before any of the above. PTB-XL uses SCP-ECG diagnostic statements with a five-superclass grouping: NORM, MI, STTC, CD, HYP. Your four classes are Normal, Arrhythmia, Myocardial Infarction and History of MI. These do not line up cleanly:

- PTB-XL has no direct "History of MI" superclass; old infarcts appear inside MI subclasses.
- Your "Arrhythmia" corresponds to rhythm statements, which in PTB-XL sit partly in CD and partly outside the superclass grouping.
- PTB-XL records carry multiple labels per record; yours are single-label.

If the mapping is wrong, low accuracy is correct behaviour and no engineering will improve it. Map explicitly, drop records whose superclass has no counterpart in your four classes, and report the mapping in any write-up.

### How to frame this in the paper

Do not present it as a failure. The cross-dataset result in Section V-E already makes the point, and a second independent domain confirming it strengthens the claim rather than weakening it. If you run the PTB-XL experiment properly, it becomes a second data point in the same argument: the quantum kernel's decision boundaries are precise and domain-tied, and the classical feature extractor is the component that fails to transfer. That is a more interesting finding than an accuracy number.

---

## 3. Dashboard features worth adding

Grouped by what they buy you. The demo column matters for a viva or a conference presentation; the clinical column matters if anyone ever pilots this.

### High value for a demo or viva

| Feature | Why it lands |
|---|---|
| **Side-by-side original and preprocessed image** | The preprocessing is seven non-obvious steps and nobody believes it works until they see the grid disappear. You already return the preprocessed thumbnail — show them together with a slider. |
| **Live circuit visualiser** | Render the actual nine-qubit ZZFeatureMap with the uploaded sample's angles bound in. Showing the real circuit for the real image is the moment where a quantum project stops looking like a classical project with a label. |
| **Encoding-range toggle** | Let the user switch the same image between [0,1] and [0,π] and watch the prediction and confidence change. This is your headline research finding, made interactive in one control. |
| **Kernel row heatmap** | Show the 742 kernel values for the uploaded image as a strip coloured by class. The bright band over one class *is* the classification, visually. |
| **Three-model race** | `/predict/all` already exists. Show the three models resolving in parallel with their timings, not as a static table. |
| **Ablation explorer** | The 320-configuration sweep is your main contribution and it is currently invisible in the interface. A filterable table or parallel-coordinates plot over all 320 rows, with the selected configuration highlighted, turns a paper table into something an examiner can play with. |

### High value for a clinical pilot

| Feature | Why it matters |
|---|---|
| **Abstention state** | Section 1 Fix 3. A fourth outcome beyond the four classes: "cannot assess". Clinically this is the single most important addition, because a confident wrong answer is worse than no answer. |
| **Triage queue view** | Batch upload currently returns a list. Sort it by severity and confidence so an MI at 96% sits at the top and a Normal at 58% is flagged for review. That is the actual workflow the project claims to serve. |
| **Per-case reviewer note and agree/disagree capture** | Lets a clinician record whether they accepted the suggestion. After a few hundred cases you have a real-world validation set, which is worth more than any further simulation. |
| **Confidence explained in words** | "94% confident" means little to a non-technical user. "94% confident — historically, predictions at this confidence are correct about 94 times in 100" converts your ECE result into something actionable. |
| **Preprocessing failure warning** | If the contour crop removes more than some fraction of the image, or the ink ratio is anomalous, warn before classifying rather than after. |
| **Audit export** | CSV or PDF export of the history table, date-filtered. Any clinical deployment needs this and it is an hour of work. |
| **Offline / PWA mode** | Already in your future work. The target setting is a rural centre with poor connectivity; a browser-cached frontend plus a locally hosted backend is the honest deployment story. |

### Lower priority but cheap

- Dark mode and a print stylesheet for the PDF preview.
- Keyboard shortcuts for upload and classify, for batch work.
- A "sample ECGs" row of four one-click demo images, one per class. Removes the awkward file-hunting pause in a live demo.
- Model card panel: training data, intended use, known failure modes, last updated. Increasingly expected for anything medical, and you already have all the content.

---

## 4. Additions that would raise novelty for the next paper

Ordered by value per unit of effort.

**1. Seed variance across the factorial cells.** Not novel, but it is the difference between a conference paper and a journal paper, and it is the first thing any reviewer will ask for. Five seeds on four cells is twenty runs of a fifteen-second kernel build. Do this regardless.

**2. Kernel Target Alignment across all 320 configurations.** You already compute KTA for the optimal configuration (0.1081). Computing it for all 320 and asking whether it predicts the accuracy ordering is a genuine contribution: it would give practitioners a label-aware selection criterion that does not require training a classifier per configuration. If alignment tracks accuracy, that is a useful and publishable result. If it does not, that is a useful negative result about KTA as a selection heuristic. Either outcome is worth writing up, and the compute is trivial because the statevectors are already cached.

**3. Kernel concentration measured directly.** ✅ *Done.* Off-diagonal kernel statistics computed for 200 training samples across all four encoding ranges under circular entanglement. Results inserted into Paper §VI-A and Report §5.6. Under [0,1]: mean=0.0139, std=0.0736, effective rank=94.6. Under [0,π]: mean=0.0060, std=0.0514, effective rank=130.4. Mean drops 57%, rank rises 38%, confirming the Bloch-sphere mechanism quantitatively. Results saved to `results/paper/ablation/kernel_concentration.json`.

**4. Noise-model evaluation.** Run the optimal configuration under a realistic Qiskit Aer noise model at nine qubits, and report the accuracy drop. This directly addresses the "ideal simulation" limitation and is a self-contained experiment. Reviewers of quantum papers ask for it routinely.

**5. Domain-adapted extractor.** The cross-dataset reversal is currently a limitation. If fine-tuning the extractor closes the gap, the result becomes a contribution: evidence that quantum kernel brittleness across domains originates in the classical front end and is fixable there. That reframes your negative result as a diagnosed and treated one.

**6. Encoding range swept finely near π.** Your [0,π] optimum sits exactly at the point where the endpoints collide modulo 2π. Sweeping 0.8π to 1.0π in steps would show whether the true optimum is slightly below π, which would be a small but genuinely new finding about ZZFeatureMap encoding that nobody has reported.

**7. Per-class ablation.** ✅ *Partially done.* Per-class P/R/F1 table added to Paper §V-A (Table III-B) and Report §7.1 (Table 7.1-B). Key finding: the QSVC's largest advantage is on History of MI (F1 0.923 vs. 0.754 SVM) and the weakest class for all models is Arrhythmia. Whether the topology change specifically helps Arrhythmia vs. History of MI requires per-class comparison of the four factorial cells — this is still open.

**8. Quanvolutional Neural Network.** Prabhu et al. report 97.31% with a QNN, which is above everything you have. Implementing it would let you compare kernel and quanvolutional approaches under one pipeline, one split and one test set — a comparison nobody has run fairly. This is the highest-ceiling item and also the most work.

---

## 5. Honest assessment of the project

The engineering is well beyond what a four-person BE project usually produces. The separation of offline training from online inference, the cached-statevector trick that turns quantum inference into dense linear algebra, the leakage control on SVD and scaler fitting, the calibration wrapper, the audit log and the containerised deployment are all things that many published papers in this area do not have.

The research contribution is real but narrower than the accuracy numbers suggest, and this is worth being clear-eyed about. Your headline accuracy of 94.62% is 0.53 points above the base paper, which is one test image and means nothing. What is genuinely yours is: nobody had varied the encoding range or the entanglement topology and reported what happens; nobody had significance-tested the comparison; nobody had calibrated; nobody had deployed; and nobody had reported where it stops working. That is a legitimate empirical contribution and it is exactly the kind of thing a good conference paper is for.

The weakest point is that the quantum-versus-classical comparison rests on a classical baseline set that is too small. Random Forest, at 91.94% with no quantum machinery at all, is close enough that the quantum advantage claim will not survive contact with a determined reviewer unless you either run the significance test and report the result honestly, or broaden the baseline set. Reporting it honestly is both easier and more defensible.

The strongest point, oddly, is the cross-dataset failure. Reporting a result that undercuts your own headline is the single clearest signal to a reviewer that the rest of the numbers are trustworthy. Keep it prominent.
