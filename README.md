# StanceJEV

**Typed Probabilistic Stance Detection with Consistency-Verified Decisions**

StanceJEV casts zero-shot stance detection as a **typed** probabilistic choice over `{Favor, Against, None}`. A CoVer-style context reconstruction (CR) → batch reasoning (BR) → consistency verification (CV) stack feeds a **Jev / DecisionSOP** terminal contract that returns `p(y)` and an auditable trace. Schema-valid labels are guaranteed by construction.

- Paper (PDF): [`paper/paper.pdf`](paper/paper.pdf)
- Author: Hazel Bloomer (Independent Researcher) — `hazel.bloomer.research@gmail.com`
- Related system: [jevtree](https://github.com/qzqdz/jevtree)

## Results (summary)

| Dataset | Metric | Score | Q_AVG (remote) | Schema-valid |
|---------|--------|-------|----------------|--------------|
| SemEval-2016 Task A (n=1,249) | F_AVG | 0.785 | 2.0 | 100% |
| VAST zero-shot (n=1,460) | F_macro3 | 0.833 | 2.0 | 100% |

`Q_AVG` counts remote BR + Jev calls only; local CV / CR embedder do not count toward the remote API budget. See `results/sj1_semeval_api.json` and `results/sj1_vast_api.json`.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
# Minimal deps used by the package (adjust if your env already has them):
pip install numpy scipy scikit-learn requests
export PYTHONPATH=src
```

API keys (OpenRouter / optional LLM for BR) go in a local `secrets/.env` or `.env` — **never commit them**.

## Smoke test (no API)

```bash
export PYTHONPATH=src
bash scripts/run_sj1_smoke.sh
# or:
python3 -m stancejev smoke --input data/fixtures/synthetic_stance.jsonl --out results/sj1_smoke.json
```

## Datasets (not redistributed)

Raw SemEval tweets and VAST dumps are **not** in this repo (`data/raw/`, `data/processed/` are gitignored).

1. Follow [`data/README.md`](data/README.md) to obtain SemEval-2016 Task 6 and VAST.
2. Convert with `python3 scripts/prepare_datasets.py`.
3. Run evaluation scripts under `scripts/` (requires configured API keys).

## Layout

```
src/stancejev/   # Python package (CR, BR, CV, DecisionSOP, metrics, CLI)
scripts/         # dataset prep + SJ-1 runners
paper/           # ACL-style LaTeX + PDF
data/fixtures/   # synthetic samples for smoke tests
results/         # summary metrics JSON only
notes/           # experiment / publication notes
```

## Citation

If you use StanceJEV, please cite the accompanying paper (`paper/paper.tex`) and the Jev / DecisionSOP stack ([jevtree](https://github.com/qzqdz/jevtree)).

## License

MIT — see [`LICENSE`](LICENSE).

---

## 中文简介

StanceJEV 把零样本立场检测做成 **typed** 概率决策：在 `{Favor, Against, None}` 上输出 `p(y)` 与可审计轨迹，而不是再解析自由文本。流水线为 CR → BR → CV → Jev/DecisionSOP；远程查询预算只计 BR + Jev。

- 论文 PDF：`paper/paper.pdf`
- 安装：`export PYTHONPATH=src`，依赖见上文
- 冒烟（无 API）：`bash scripts/run_sj1_smoke.sh`
- **不 redistributing** SemEval / VAST 原文；获取与转换见 `data/README.md`
- 密钥放本地 `secrets/.env`，勿提交

作者：Hazel Bloomer · MIT License
