"""StanceJEVPipeline: CR → BR → CV → typed decision, with M re-query rounds.

API mode:
  - BR uses mimo (JEVTREE_LLM_*) for explanation (+ soft P_LLM)
  - Final typed probs from OpenRouter Jev Decisions API
  - CV κ between Jev probs and heuristic SLM on explanation
Mock mode:
  - BR mock keyword heuristic; no Jev call
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

from .br import BatchReasoner
from .cr import ContextReconstruction
from .cv import ConsistencyVerifier
from .decision import call_jev_stance, emit_typed_decision, jev_configured
from .types import BatchItem, Prediction


@dataclass
class StanceJEVPipeline:
    cr: ContextReconstruction = field(default_factory=ContextReconstruction)
    br: BatchReasoner = field(default_factory=BatchReasoner)
    cv: ConsistencyVerifier = field(default_factory=ConsistencyVerifier)
    max_rounds: int = 2  # M: initial + re-queries
    use_jev_decision: bool | None = None  # None → auto (api backend + keys)

    def _want_jev(self) -> bool:
        if self.use_jev_decision is not None:
            return self.use_jev_decision
        return self.br.backend in ("api", "jev") and jev_configured()

    def predict_one(self, item: BatchItem) -> Prediction:
        return self.predict([item])[0]

    def predict(self, items: list[BatchItem]) -> list[Prediction]:
        if not items:
            return []
        reconstructed = [self.cr.reconstruct(it) for it in items]
        first = self.br.reason_batch(items, reconstructed)
        want_jev = self._want_jev()
        preds: list[Prediction] = []
        n_items = len(items)
        for idx, (it, rec, br0) in enumerate(zip(items, reconstructed, first), start=1):
            rounds_data: list[tuple[dict[str, float], float]] = []
            queries = 0
            cur_expl = br0.explanation
            llm_probs = br0.probs
            jev_meta: dict = {}
            kappa = 0.0
            final_probs = llm_probs

            for r in range(1, self.max_rounds + 1):
                if r > 1:
                    br_r = self.br.reason_one(it, rec)
                    cur_expl, llm_probs = br_r.explanation, br_r.probs
                queries += 1  # BR query

                if want_jev:
                    jev = call_jev_stance(text=rec, target=it.target)
                    queries += 1  # Jev decision query
                    final_probs = jev["probs"]
                    jev_meta = {
                        "jev_choice": jev["choice"],
                        "jev_confidence": jev["confidence"],
                        "jev_model": jev["model"],
                        "jev_usage": jev.get("usage"),
                        "llm_probs": llm_probs,
                    }
                    kappa, _ = self.cv.kappa(final_probs, cur_expl, rec, it.target)
                else:
                    final_probs = llm_probs
                    kappa, _ = self.cv.kappa(final_probs, cur_expl, rec, it.target)

                rounds_data.append((final_probs, kappa))
                if self.cv.accept(kappa):
                    break

            if len(rounds_data) > 1 and not self.cv.accept(kappa):
                final_probs = self.cv.weighted_aggregate(rounds_data)
                kappa = rounds_data[-1][1]

            backend = self.br.backend
            if want_jev and not str(backend).endswith("jev"):
                backend = f"{self.br.backend}+jev"
            elif want_jev and backend == "jev":
                backend = "jev"

            pred = emit_typed_decision(
                item_id=it.id,
                target=it.target,
                probs=final_probs,
                explanation=cur_expl,
                kappa=kappa,
                rounds=len(rounds_data),
                queries=queries,
                reconstructed_text=rec,
                backend=backend,
                meta_extra=jev_meta or None,
            )
            preds.append(pred)
            if idx == 1 or idx % 25 == 0 or idx == n_items:
                print(f"[stancejev] {idx}/{n_items} backend={backend} kappa={kappa:.3f}", file=sys.stderr, flush=True)
        return preds
