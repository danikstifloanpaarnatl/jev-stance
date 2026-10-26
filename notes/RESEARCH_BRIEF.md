# JEV-stance Research Brief

**Track:** jev-stance (stance-detection R&D on JEV / jevtree)  
**Working system name:** **StanceJEV**  
**Host workspace:** `/workspace/jev-stance`  
**Code reference (read-only):** `/workspace/jevtree_pub` → `github.com/qzqdz/jevtree`  
**Primary scientific anchors:** Yan Yu et al. CoVer (DASFAA 2025 / arXiv:2502.19954); Decision Hijacking (arXiv:2609.28613) as *writing template only*

---

## 1. Problem

Stance detection asks, for text \(x\) and target \(t\), which label in
\(\mathcal{Y}=\{\texttt{Favor},\texttt{Against},\texttt{None}\}\) is expressed.
LLM-only pipelines are accurate but expensive and opaque; SLM-only pipelines are cheap but brittle on implicit cues (CoVer). Separately, JEV / jevtree already provide *typed probabilistic decisions*—a closed action set plus reported probabilities and auditable DecisionSOP traces—but they are not specialized for stance monitoring.

**Gap:** There is no first-class stance track that (i) maps \(\mathcal{Y}\) onto JEV choice-typed decisions, (ii) imports CoVer’s context reconstruction / batch reasoning / SLM consistency verification into the JEV decision path, and (iii) evaluates with the same metric discipline used in typed-decision security papers (probability shifts, registered contrasts, staged ablations)—without turning the paper into an attack study.

---

## 2. Insight

Stance labels are already a closed, schema-defined action set. Treating them as JEV `choice` questions yields:

- a reported probability vector \(p(y \mid s, \mathcal{Y})\) instead of free-form generation;
- DecisionSOP / IG-tree growth from feature tables (jevtree path: data+NL goal → FeatureTable → tree/SOP/trace);
- a natural place to insert CoVer-style *consistency gates*: when LLM batch reasoning and SLM verification disagree, defer to consistency-weighted aggregation rather than trusting a single generative call.

**Positioning**

| Axis | CoVer | Decision Hijacking | **StanceJEV (this work)** |
|------|-------|--------------------|---------------------------|
| Task | ZSSD accuracy + query cost | Prompt-injection on typed decisions | Stance via typed decisions + consistency path |
| Output | Label (+ likelihoods) | Attacker-target vs safe actions | Favor / Against / None with \(p\) and trace |
| Writing craft used | Scientific claims on SD | Staged metrics, SESOI, clean framing | Borrow *craft*, not attack topic |

---

## 3. System sketch (StanceJEV)

```
tweet x_raw + target t
        │
        ▼
[CR] Context Reconstruction  (knowledge attach + SE sentence filter; SLM embedder)
        │
        ▼
[BR] Batch Reasoning         (LLM over batch B → explanation e_i + P_LLM,i)
        │
        ▼
[CV] Consistency Verification (SLM on e_i → P_SLM,i; consistency score κ_i)
        │
        ├── κ_i ≥ τ  → emit typed JEV choice decision (argmax / report p)
        └── κ_i < τ  → re-query ≤ M times; then consistency-weighted aggregate
        │
        ▼
[SOP] DecisionSOP / jevtree run+trace  (auditable path; optional IG feature tree)
```

**Typed decision interface (conceptual):**

```json
{
  "type": "choice",
  "instructions": "Stance of the text toward the target.",
  "criteria": {
    "favor": "Supports / favors the target",
    "against": "Opposes / is against the target",
    "none": "Neutral / unrelated / no clear stance"
  }
}
```

State \(s\) carries reconstructed text, target, optional knowledge snippets, and prior κ history. Output is always in \(\mathcal{Y}\) with reported probabilities—never free text as the decision.

---

## 4. Experiment plan

### Must-have (paper cannot claim primary results without these)

1. **SJ-1 Zero-shot accuracy.** SemEval-2016 (leave-one-target-out), VAST, P-Stance. Metric: \(F_{\text{AVG}}\) (Favor/Against macro-F1; VAST protocol as in CoVer). Compare StanceJEV vs Task-Demo LLM, CoVer-style pipeline, JointCL / COLA numbers as reported baselines.
2. **SJ-2 Consistency gate ablation.** Full vs w/o CR / w/o CV / w/o weighted agg. Report \(F_{\text{AVG}}\) and mean LLM queries/tweet \(Q_{\text{AVG}}\).
3. **SJ-3 Typed-decision fidelity.** On the same items, compare (a) free-form LLM label parse vs (b) JEV choice schema. Metrics: label agreement, mean \(\max_y p(y)\), fraction of schema-valid outputs (should be 100% for typed).

### High-value

4. **SJ-4 Batch-size scaling.** Batches \(\{1,8,16,32\}\) × 1–2 LLMs; show \(F_{\text{AVG}}\) vs \(Q_{\text{AVG}}\) (CoVer Fig.4 style).
5. **SJ-5 Trace audit sample.** Manual audit of \(N\) DecisionSOP traces for explanation–label alignment (registered rubric).
6. **SJ-6 Cross-target transfer.** Hold out one Sem16 target entirely; report degradation vs CoVer.

### Nice-to-have

7. jevtree IG-grown feature tree from labeled stance FeatureTable as a non-LLM baseline branch.
8. Robustness to noisy/irrelevant context (relatedness-style contrast, *inspired by* DH-3 craft, not injection ASR).
9. Latency / \$ cost table for monitoring-scale batch sizes.

---

## 5. Success criteria (internal)

- Draft paper mirrors Decision Hijacking section craft (staged SJ-*, narrow claims, placeholders for unset numbers).
- ≥5 Yan Yu citations used substantively; ~30 total real cites.
- Clear non-overlap claim: *not* a hijacking/attack paper; *is* stance system R&D on typed decisions.
- `src/` documents plug-in interfaces only—no fabricated benchmark numbers in code stubs.
- Human can run SJ-1/SJ-2 once API keys + datasets are available without redesigning the method.

---

## 6. Risks / open items

- edgexpert JEV project path still has Meta-Jev vs jevtree naming ambiguity; use public `jevtree` + TypeSafe Jev docs as interfaces.
- CoVer SLM training details must be reimplemented or approximated; mark training hyperparams as placeholders until reproduced.
- AVATAR arXiv:2412.12145 was withdrawn; cite ACL 2025 published AVATAR (`from Benign import Toxic`) as the stable venue record while retaining the requested arXiv id in the Yan Yu cite list.

---

## 7. Deliverables in this workspace

| Path | Role |
|------|------|
| `notes/RESEARCH_BRIEF.md` | This brief (planning OK) |
| `notes/YAN_YU_CITATIONS.md` | Yan Yu cite → paper usage map |
| `paper/paper.tex` | Target-acceptance manuscript (hijack-template craft) |
| `paper/references.bib` | Merged ~29 cites (template + stance + Yan Yu) |
| `paper/refs.bib` | Mirror of `references.bib` |
| `src/README.md` | Plug-in interfaces only |


---

## 8. Rebase note (hijack template craft)

Canonical manuscript craft cloned from **real** jev_hijack template:

- Box template: `/workspace/jev-stance/template_hijack/paper.tex`
- Server (when online): `/home/qzqdz/Desktop/project/jev_project/jev_hijack/`
- Stance sibling on server: `/home/qzqdz/Desktop/project/jev_project/jev-stance/`

Document class / preamble / titlesec / fancyhdr / `cite`+`plain` bib style /
definition environments / TikZ figure craft follow that template.
Stance research content (CoVer → typed JEV) is unchanged in substance.

**Build (once TeX available):**
```bash
cd /workspace/jev-stance/paper
pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex
```

**Yan Yu keys (yan_yu_extra.bib):** `yan2025cover`, `yan2025benign`,
`yan2025confusion`, `yan2026red`, `yan2025poisonswarm`, `yan2026searchattack`.
