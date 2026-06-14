# geneal Risk-Aware Target Nomination — Implementation Plan (Plan 5)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or executing-plans. Steps use `- [ ]`.

**Goal:** Build the project's headline contribution — active-learning **risk-aware target nomination**: select a batch of K knockout targets that are simultaneously (1) **lethal** in the cancer line (efficacy), (2) **selective** = not common-essential (toxicity hedge), and (3) **mechanistically hedged** = not concentrated in one pathway (portfolio risk against pathway-level toxicity/undruggability/failure). Produce the three figures: greedy's concentration risk, the efficacy–risk Pareto, and pathway-dropout robustness.

**The reframing that drives this (north star):** diversity here is NOT active-learning loop efficiency (we proved that's unnecessary — the candidate space is already sparse). It is **downstream portfolio risk management**. Greedy over-concentrates (30 picks in 5–6 pathways on several lines); explicit diversity caps per-pathway exposure at a quantifiable potency cost. **The lethality-vs-concentration tradeoff IS the deliverable** — the efficacy-vs-hedge frontier a target-nominator picks a risk tolerance on. This turns every prior "diversity costs quality" result from a bug into the central, practically-motivated finding.

**Architecture:** Reuse the whole geneal stack. New: (a) a **selective-lethality target** (lethal in line minus common-essential baseline), (b) **portfolio-risk metrics** (pathway concentration; surviving-targets-under-pathway-dropout), (c) a thin **single-shot nomination harness** that, given a fitted surrogate's predictions, compares selection methods and emits the risk metrics + figures. CORUM (continuous co-complex / pathway membership) is the mechanism graph (already fetched: `data/processed/depmap/corum_sim.parquet`). No new heavy deps.

**Tech Stack:** existing geneal stack (numpy, pandas, scipy, gpytorch, plotly, jinja2, pytest). CORUM + DepMap already on disk.

**The normal-proxy problem (resolved):** DepMap is all cancer lines — no true normal tissue. Toxicity is proxied by **common-essential genes**: genes lethal across most cell lines are pan-essential and would kill normal cells too. So selective lethality = lethal in the target line but NOT common-essential. This is standard DepMap practice (selective dependencies vs common essentials) and avoids fabricating a normal line. `common_essential_score(gene) = fraction of cell lines where the gene is strongly lethal (effect < -0.5)`, computed genome-wide from the full matrix.

**Conventions:** lethality = −(Chronos effect), higher=better. selective_lethality = lethality_in_line − λ·common_essential_lethality (λ tunable, default chosen so the two terms are comparably scaled). All experiment knobs are CLI/config params.

---

## File Structure

```
src/geneal/metrics/portfolio.py        # CREATE: pathway_concentration, dropout_robustness, n_pathways
src/geneal/data/selective.py           # CREATE: common_essential_score, build_selective_dataset
src/geneal/models/hedged_selection.py  # CREATE: HedgedKDPP (q=selective-lethality, S=CORUM continuous, tempered)
scripts/run_risk_nomination.py         # CREATE: unified experiment + 3 figures
tests/test_portfolio.py                # CREATE
tests/test_selective.py                # CREATE
tests/test_hedged_selection.py         # CREATE
data/RISK_NOMINATION_RESULTS.md        # ARTIFACT: results + honest read
```

---

## Task 1: Portfolio-risk metrics

**Files:** Create `src/geneal/metrics/portfolio.py`, `tests/test_portfolio.py`.

Metrics operate on a selected gene-index set + a pathway/complex membership structure (dict gene_idx -> set of pathway ids, OR a binary co-membership matrix). We use CORUM complexes as pathways.

- [ ] **Step 1: failing test `tests/test_portfolio.py`**

```python
# tests/test_portfolio.py
import numpy as np
from geneal.metrics.portfolio import (n_pathways_covered, pathway_concentration,
                                       dropout_robustness)

# gene -> set of pathway ids
MEMBERSHIP = {0: {"A"}, 1: {"A"}, 2: {"A"}, 3: {"B"}, 4: {"C"}, 5: set()}


def test_n_pathways_covered():
    # genes 0,1,2 in A; 3 in B; 4 in C -> 3 pathways (gene 5 has none)
    assert n_pathways_covered([0, 1, 2, 3, 4, 5], MEMBERSHIP) == 3


def test_pathway_concentration_max_fraction():
    # 0,1,2,3 -> A has 3 of 4 genes-with-pathways -> 3/4 = 0.75
    assert np.isclose(pathway_concentration([0, 1, 2, 3], MEMBERSHIP), 0.75)


def test_pathway_concentration_spread_is_low():
    # 2(A),3(B),4(C) -> each pathway 1/3
    assert np.isclose(pathway_concentration([2, 3, 4], MEMBERSHIP), 1/3)


def test_dropout_robustness_counts_survivors():
    # value per gene; if pathway A is dropped (toxic), genes 0,1,2 lost.
    # robustness = expected fraction of portfolio VALUE surviving a random
    # single-pathway dropout. With pathways {A:3 genes, B:1, C:1}, dropping A
    # (prob 1/3) loses 3/5 genes; B or C loses 1/5 each. Equal-value genes:
    # expected survivors = 1 - mean over pathways of (genes_in_pathway/total)
    sel = [0, 1, 2, 3, 4]  # gene 5 excluded (no pathway)
    val = np.ones(6)
    r = dropout_robustness(sel, MEMBERSHIP, val)
    # pathways A(3),B(1),C(1); total value 5; drop A->lose3, B->lose1, C->lose1
    # expected surviving fraction = 1 - mean(3/5,1/5,1/5) = 1 - (5/15)=1-0.333=0.667
    assert np.isclose(r, 0.667, atol=1e-2)
```

- [ ] **Step 2: run, confirm fail.**

- [ ] **Step 3: implement `src/geneal/metrics/portfolio.py`**

```python
# src/geneal/metrics/portfolio.py
from __future__ import annotations
from typing import Sequence
import numpy as np


def _pathways_of(sel, membership):
    return {g: membership.get(g, set()) for g in sel}


def n_pathways_covered(sel: Sequence[int], membership: dict) -> int:
    """Number of DISTINCT pathways/complexes the selected genes touch."""
    paths = set()
    for g in sel:
        paths |= membership.get(g, set())
    return len(paths)


def pathway_concentration(sel: Sequence[int], membership: dict) -> float:
    """Max fraction of the (pathway-annotated) portfolio in any single pathway.

    1.0 = all annotated picks in one pathway (max concentration risk);
    low = spread across pathways. Genes with no pathway are excluded from the
    denominator (can't assess their pathway risk)."""
    counts: dict = {}
    annotated = 0
    for g in sel:
        ps = membership.get(g, set())
        if not ps:
            continue
        annotated += 1
        for p in ps:
            counts[p] = counts.get(p, 0) + 1
    if annotated == 0:
        return 0.0
    return max(counts.values()) / annotated


def dropout_robustness(sel: Sequence[int], membership: dict,
                       value: np.ndarray) -> float:
    """Expected fraction of portfolio VALUE that survives if a single random
    pathway is eliminated (e.g. found toxic/undruggable).

    For each pathway present, dropping it removes all selected genes in it;
    robustness = 1 - mean over present pathways of (lost value / total value).
    Higher = more robust to pathway-level failure. Genes with no pathway always
    survive (no pathway risk)."""
    sel = list(sel)
    value = np.asarray(value, dtype=float)
    total = float(value[sel].sum())
    if total <= 0:
        return 0.0
    present: dict = {}
    for g in sel:
        for p in membership.get(g, set()):
            present.setdefault(p, 0.0)
            present[p] += float(value[g])
    if not present:
        return 1.0  # nothing annotated -> nothing to drop
    lost_fracs = [v / total for v in present.values()]
    return 1.0 - float(np.mean(lost_fracs))
```

- [ ] **Step 4: run, confirm 4 passed.**
- [ ] **Step 5: commit** `feat: add portfolio-risk metrics (concentration, dropout robustness)`.

---

## Task 2: Selective-lethality dataset (efficacy minus common-essential)

**Files:** Create `src/geneal/data/selective.py`, `tests/test_selective.py`.

- [ ] **Step 1: failing test `tests/test_selective.py`**

```python
# tests/test_selective.py
import numpy as np
import pandas as pd
from geneal.data.selective import common_essential_score, build_selective_dataset


def _ge(tmp_path):
    # 3 genes x 4 lines. g0 pan-essential (lethal everywhere)=toxic.
    # g1 selectively lethal in L1 only. g2 inert.
    df = pd.DataFrame(
        {"L1": [-2.0, -2.0, 0.0], "L2": [-2.0, 0.0, 0.0],
         "L3": [-2.0, 0.0, 0.1], "L4": [-2.0, 0.1, 0.0]},
        index=pd.Index(["A (1)", "B (2)", "C (3)"], name="gene"))
    df.columns.name = "cell_line"
    p = tmp_path / "ge.parquet"; df.to_parquet(p); return p


def test_common_essential_score(tmp_path):
    ge = pd.read_parquet(_ge(tmp_path))
    ces = common_essential_score(ge, thresh=-0.5)
    # A lethal in all 4 lines -> 1.0 ; B lethal in 1/4 -> 0.25 ; C -> 0
    assert np.isclose(ces["A (1)"], 1.0)
    assert np.isclose(ces["B (2)"], 0.25)
    assert np.isclose(ces["C (3)"], 0.0)


def test_build_selective_target_downweights_common_essential(tmp_path):
    ge = pd.read_parquet(_ge(tmp_path))
    emb = pd.DataFrame(np.eye(3), index=pd.Index([1, 2, 3], name="entrez"))
    ds, aux = build_selective_dataset(ge, emb, cell_line="L1", lam=1.0, thresh=-0.5)
    i_a = ds.gene_names.index("A (1)"); i_b = ds.gene_names.index("B (2)")
    # A: lethality 2.0 but common-ess 1.0 -> selective ~ 2 - 1*(mean lethality of A across lines=2) = 0
    # B: lethality 2.0 in L1, common-ess 0.25 -> selective stays high
    # so B (selective) should outrank A (pan-essential/toxic)
    assert ds.target[i_b] > ds.target[i_a]
    assert "common_essential" in aux
```

- [ ] **Step 2: run, confirm fail.**

- [ ] **Step 3: implement `src/geneal/data/selective.py`**

```python
# src/geneal/data/selective.py
from __future__ import annotations
import numpy as np
import pandas as pd
from geneal.data.dataset import Dataset
from geneal.data.depmap import parse_entrez


def common_essential_score(gene_effect: pd.DataFrame, thresh: float = -0.5) -> pd.Series:
    """Fraction of cell lines where each gene is strongly lethal (effect < thresh).

    High = pan-essential = toxicity proxy (would kill normal cells too)."""
    return (gene_effect < thresh).mean(axis=1)


def build_selective_dataset(gene_effect: pd.DataFrame, embeddings: pd.DataFrame,
                            cell_line: str, lam: float = 1.0,
                            thresh: float = -0.5):
    """Dataset whose target is SELECTIVE lethality:
        target = lethality_in_line - lam * (common-essential lethality)
    where lethality = -effect, and the common-essential penalty is the gene's
    MEAN lethality across all lines weighted by how pan-essential it is. High
    target = lethal in THIS line but not a general essential (lower toxicity).
    Genes need an effect in this line and an embedding (by Entrez)."""
    if cell_line not in gene_effect.columns:
        raise KeyError(cell_line)
    ces = common_essential_score(gene_effect, thresh)         # 0..1 per gene
    mean_leth = -(gene_effect.mean(axis=1))                    # avg lethality across lines
    col = gene_effect[cell_line].dropna()
    emb_index = set(embeddings.index)
    rows, target, names, ce_list = [], [], [], []
    for label, eff in col.items():
        ent = parse_entrez(label)
        if ent not in emb_index:
            continue
        leth = -float(eff)
        # penalty: pan-essential genes (high ces, high mean lethality) are toxic
        penalty = lam * float(ces[label]) * float(mean_leth[label])
        rows.append(embeddings.loc[ent].to_numpy(dtype=float))
        target.append(leth - penalty)
        names.append(label); ce_list.append(float(ces[label]))
    if not names:
        raise ValueError("no genes with effect+embedding")
    ds = Dataset(np.vstack(rows), np.asarray(target), names)
    aux = {"common_essential": np.asarray(ce_list)}
    return ds, aux
```

- [ ] **Step 4: run, confirm 2 passed.**
- [ ] **Step 5: commit** `feat: add selective-lethality dataset (efficacy minus common-essential toxicity proxy)`.

---

## Task 3: Hedged selection (q=selective-lethality, S=CORUM, tempered)

**Files:** Create `src/geneal/models/hedged_selection.py`, `tests/test_hedged_selection.py`.

A Selection strategy that takes an EXTERNAL similarity matrix (CORUM) rather than deriving S from the surrogate. q = acquisition on the (selective) target; S = continuous CORUM co-membership; greedy-MAP on `L=diag(q^β) S diag(q^β)`. β controls efficacy↔hedge. This is the honest mechanism-diversity selector — we KNOW β won't trace a smooth frontier in a sparse graph (documented), so we ALSO expose a hard **per-pathway cap** mode (cap m genes per pathway) which is the practical hedging lever.

- [ ] **Step 1: failing test `tests/test_hedged_selection.py`**

```python
# tests/test_hedged_selection.py
import numpy as np
from geneal.models.hedged_selection import HedgedSelect


def test_capped_selection_limits_per_pathway():
    # 6 genes, q descending; genes 0,1,2 in pathway A, 3,4,5 singletons.
    membership = {0: {"A"}, 1: {"A"}, 2: {"A"}, 3: {"B"}, 4: {"C"}, 5: {"D"}}
    q = np.array([0.9, 0.85, 0.8, 0.4, 0.3, 0.2])
    sel = HedgedSelect(mode="cap", cap=1).select_idx(q, membership, K=3)
    # cap=1 per pathway: can take only 1 from A -> picks gene0(A), then 3(B),4(C)
    assert 0 in sel
    assert not (1 in sel and 2 in sel)  # never 2 from pathway A beyond cap
    assert len(sel) == 3


def test_greedy_mode_ignores_pathways():
    membership = {i: {"A"} for i in range(6)}
    q = np.arange(6.0)[::-1]
    sel = HedgedSelect(mode="greedy").select_idx(q, membership, K=3)
    assert sorted(sel) == [0, 1, 2]  # top-3 by q regardless of pathway
```

- [ ] **Step 2: run, confirm fail.**

- [ ] **Step 3: implement `src/geneal/models/hedged_selection.py`**

```python
# src/geneal/models/hedged_selection.py
from __future__ import annotations
import numpy as np


class HedgedSelect:
    """Select K targets by quality q with optional pathway-hedging.

    mode='greedy' : top-K by q (concentration-blind baseline).
    mode='cap'    : greedy by q but at most `cap` genes per pathway (hard hedge).
    mode='dpp'    : quality-tempered k-DPP with an external similarity S
                    (continuous mechanism graph). Provided for completeness; the
                    practical hedging lever is 'cap' (DPP frontier is flat in
                    sparse graphs — see CORUM_VALIDATION.md)."""

    def __init__(self, mode: str = "cap", cap: int = 2, beta: float = 1.0,
                 jitter: float = 1e-9):
        self.mode = mode
        self.cap = cap
        self.beta = beta
        self.jitter = jitter

    def select_idx(self, q, membership, K, S=None) -> list[int]:
        q = np.asarray(q, dtype=float)
        order = np.argsort(q)[::-1]
        if self.mode == "greedy":
            return order[:K].tolist()
        if self.mode == "cap":
            used: dict = {}
            chosen: list[int] = []
            for i in order:
                ps = membership.get(int(i), set())
                # a gene with no pathway is always allowed
                if ps and any(used.get(p, 0) >= self.cap for p in ps):
                    continue
                chosen.append(int(i))
                for p in ps:
                    used[p] = used.get(p, 0) + 1
                if len(chosen) == K:
                    break
            # if cap was too tight to reach K, fill with next-best leftovers
            if len(chosen) < K:
                for i in order:
                    if int(i) not in chosen:
                        chosen.append(int(i))
                        if len(chosen) == K:
                            break
            return chosen
        if self.mode == "dpp":
            from geneal.models.selection import _greedy_map_logdet
            qb = ((q - q.min()) / (q.max() - q.min() + 1e-9) + 1e-3) ** self.beta
            Sc = np.array(S, dtype=float).copy()
            np.fill_diagonal(Sc, 1.0)
            L = (qb[:, None] * Sc) * qb[None, :]
            L = (L + L.T) / 2 + self.jitter * np.eye(len(q))
            return _greedy_map_logdet(L, K)
        raise ValueError(self.mode)
```

- [ ] **Step 4: run, confirm 2 passed.**
- [ ] **Step 5: commit** `feat: add HedgedSelect (greedy / per-pathway-cap / dpp modes)`.

---

## Task 4: Unified risk-nomination experiment + figures

**Files:** Create `scripts/run_risk_nomination.py`. Produces results + 3 figures. This is the paper's central experiment; capture honestly.

- [ ] **Step 1: write `scripts/run_risk_nomination.py`** that, for each cell line in a list:
  1. Build the **selective** dataset (Task 2) + map CORUM membership (from `corum_sim.parquet` -> for each gene, the set of complex-ids it belongs to; load the raw CORUM complex membership saved by the CORUM agent, or derive connected-components of the binary matrix as pseudo-pathways).
  2. Fit a GP surrogate on an initial random set; predict selective-lethality (UCB) for candidates = quality q.
  3. Select K targets by three methods: `greedy` (q only), `cap` (per-pathway cap, e.g. cap=2), and across a sweep of caps {1,2,3,5,∞} to trace the efficacy–concentration frontier.
  4. For each: record mean selective-lethality (efficacy), pathway_concentration, dropout_robustness, n_pathways_covered.
  - argparse knobs: --cell-lines (or auto-pick N lowest-NaN), --K, --n-initial, --cap-sweep, --lam, --thresh, --seeds.
  - Aggregate across lines+seeds (mean±CI).

- [ ] **Step 2: run it** on ~4 cell lines, 2 seeds, K=30, cap sweep {1,2,3,5,1e9}:
```bash
micromamba run -n geneal python scripts/run_risk_nomination.py \
  --n-cell-lines 4 --seeds 0 1 --K 30 --embeddings data/processed/embeddings/pubmedbert_hvg.parquet
```
Report per-method: selective-lethality, concentration, dropout-robustness.

- [ ] **Step 3: generate the 3 figures** (plotly -> html/png in the run dir):
  1. **Concentration bar**: greedy vs capped — max single-pathway fraction.
  2. **Efficacy–risk Pareto**: x=concentration (or dropout-robustness), y=mean selective-lethality, one point per cap value + greedy. The frontier.
  3. **Dropout-robustness curve**: expected surviving portfolio value vs cap level.

- [ ] **Step 4: write `data/RISK_NOMINATION_RESULTS.md`** — the numbers + an HONEST read: does capping reduce concentration / improve dropout-robustness, and at what selective-lethality cost? Is there a cap value giving meaningful hedge at small efficacy loss? Is it cell-line-dependent (per the earlier finding)? Do NOT tune to flatter the method.
- [ ] **Step 5: commit** `experiment: risk-aware target nomination — efficacy-hedge frontier + dropout robustness`.

---

## Self-Review (completed during planning)

- **Objective coverage:** efficacy (selective-lethality target, Task 2) ✓; toxicity hedge via common-essential (Task 2, resolves the no-normal-line problem) ✓; mechanism hedge / portfolio risk (Tasks 1+3+4) ✓; the efficacy–risk Pareto + dropout robustness as the deliverable figures (Task 4) ✓.
- **Honest about the β-wall:** Task 3 ships the **hard per-pathway cap** as the practical hedging lever (not the inert DPP β) — we documented β can't trace a frontier in sparse graphs, so the cap is the actionable knob. DPP mode kept for completeness/comparison, not as the headline.
- **Normal-proxy resolved:** common-essential genes as toxicity proxy (standard DepMap practice), no fabricated normal line.
- **Reuses everything:** DepMap adapter, GP surrogate, CORUM, metric-panel patterns; only 3 small new modules, all TDD-tested.
- **No placeholders:** all code complete. CORUM membership extraction in Task 4 has two stated routes (raw complex file or binary-matrix components) — implementer picks whichever the saved artifacts support.
- **Multi-line:** Task 4 runs several cell lines + reports cell-line-dependence (consistent with the established finding that hedge-cost varies by line).

## Out of scope
The full AL-loop version (this is single-shot nomination from a fitted surrogate — the cleanest test of the portfolio objective; the round-by-round version is a later extension). True normal-tissue data (DepMap has none; common-essential is the proxy). The k-DPP β-frontier (documented dead; cap is the lever).
