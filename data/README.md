# StanceJEV datasets

Raw tweet/comment text stays under `data/raw/` (gitignored). Processed JSONL under `data/processed/` (gitignored).

## SemEval-2016 Task 6 (Subtask A)

- Source: official Sheffield NLP Dropbox mirror of SemEval-2016 Task 6 labelled files
  (`SemEval2016-Task6-subtaskA-testdata-gold.txt`, `semeval2016-task6-trainingdata.txt`).
- Upstream task page: https://alt.qcri.org/semeval2016/task6/
- NRC terms: free for research; do not redistribute raw tweets — point others to Saif Mohammad / SemEval.
- Convert: `python3 scripts/prepare_datasets.py`
- Outputs: `processed/semeval2016_test.jsonl` (n=1249), `semeval2016_train.jsonl` (n=2814)

## VAST (Allaway & McKeown, 2020)

- Source: https://github.com/emilyallaway/zero-shot-stance/tree/master/data/VAST
- Labels: `0=con→Against`, `1=pro→Favor`, `2=neutral→None`
- Zero-shot split: `seen?=0` → `processed/vast_zero_shot_test.jsonl` (n=1460)
- Optional native-only (excludes synthetic neutrals `type_idx=4`): `vast_zero_shot_test_native.jsonl`

## Schema

```json
{"id":"...","text":"...","target":"...","stance":"Favor|Against|None","label":"..."}
```
