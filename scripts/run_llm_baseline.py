# scripts/run_llm_baseline.py
"""STANDALONE, run-once LLM baseline for the selectivity ablation.

Asks a frontier LLM to nominate K targets from the genome given the SAME observed
data the report's methods see (the assay readouts of an active-learning sample),
using the EXACT report protocol (same lines / contrast lines / tau / K, same
build_selective_dataset efficacy + toxicity_vector non-target, same evaluate()).
The cell line is NEVER revealed to the model (blocks DepMap recall); readouts are
z-scored and renamed assay_A (want high) / assay_B (want low).

Two roles (both use observations):
  llm_nom  -- LLM nominates K from the RANDOM-acquisition 120-set (no AL loop).
              Head-to-head with the report's `random` (same 120, GP vs LLM nom).
  llm_loop -- LLM also picks each round's batch (full AL), then nominates.

This does NOT touch the main report. It writes its own llm.parquet + a small
comparison report.html / printed table against the GP baselines pulled from the
sweep's ablation.parquet. Port into the main report only if it's promising.

Run (nominator, full protocol, ~96 cached calls):
  python scripts/run_llm_baseline.py --run-dir res/runs_ablation/sweep_joint3normal_20260626-190909
Add the loop (heavier, ~800 calls):  ... --roles both
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from geneal.data.depmap import load_gene_effect, parse_entrez
from geneal.data.selective import build_selective_dataset, corum_membership, toxicity_vector
from geneal.runner.ablation import run_acquisition, evaluate, _tox_threshold


# --------------------------------------------------------------------------- #
# LLM client (raw httpx to the Anthropic-style proxy; on-disk response cache)  #
# --------------------------------------------------------------------------- #
class LLM:
    def __init__(self, model, cache_dir, max_tokens=2000):
        import httpx
        self.httpx = httpx
        self.model = model
        self.max_tokens = max_tokens
        self.base = os.environ["ANTHROPIC_BASE_URL"].rstrip("/")
        tok = os.environ["ANTHROPIC_AUTH_TOKEN"]
        self.hdr = {"x-api-key": tok, "authorization": f"Bearer {tok}",
                    "anthropic-version": "2023-06-01", "content-type": "application/json"}
        for line in (os.environ.get("ANTHROPIC_CUSTOM_HEADERS") or "").splitlines():
            if ":" in line:
                k, v = line.split(":", 1); self.hdr[k.strip()] = v.strip()
        self.cache = Path(cache_dir); self.cache.mkdir(parents=True, exist_ok=True)

    def complete(self, system, prompt):
        key = hashlib.sha256(f"{self.model}|{system}|{prompt}".encode()).hexdigest()
        cf = self.cache / f"{key}.json"
        if cf.exists():
            return json.loads(cf.read_text())["text"]
        body = {"model": self.model, "max_tokens": self.max_tokens,
                "system": system, "messages": [{"role": "user", "content": prompt}]}
        last = None
        for attempt in range(4):
            try:
                r = self.httpx.post(self.base + "/v1/messages", headers=self.hdr,
                                    json=body, timeout=360)
                if r.status_code == 200:
                    blocks = r.json().get("content", []) or []
                    # opus may emit a 'thinking' block first; take all text blocks
                    txt = "".join(b.get("text", "") for b in blocks if isinstance(b, dict))
                    if txt.strip():
                        cf.write_text(json.dumps({"text": txt}))
                        return txt
                    last = f"200 but no text block: {str(blocks)[:160]}"
                else:
                    last = f"{r.status_code} {r.text[:160]}"
            except Exception as e:  # noqa: BLE001
                last = f"{type(e).__name__} {str(e)[:160]}"
            time.sleep(2 * (attempt + 1))
        # degrade gracefully: one bad call must not kill the whole run
        print(f"  WARN llm call failed after retries ({last}); returning empty")
        return ""


_SYSTEM = "You are a biology expert selecting gene knockout targets from a screen."


def _zscore_block(sym, zA, zB, idx):
    """Observed lines 'SYMBOL\\tzA\\tzB', sorted best-first by (zA - zB)."""
    order = sorted(idx, key=lambda i: zA[i] - zB[i], reverse=True)
    return "\n".join(f"{sym[i]}\t{zA[i]:+.2f}\t{zB[i]:+.2f}" for i in order)


def _parse_picks(text, sym_to_idx, candidate_set, K):
    tail = text.split("PICKS:")[-1]
    cand = re.findall(r'"([A-Za-z0-9\-\.]+)"', tail) or re.findall(r'\b[A-Z0-9\-]{2,}\b', tail)
    out = []
    for s in cand:
        i = sym_to_idx.get(s)
        if i is not None and i in candidate_set and i not in out:
            out.append(i)
    return out[:K]


def _zscores(eff, tox, ref_idx):
    a, b = eff[ref_idx], tox[ref_idx]
    zA = (eff - a.mean()) / (a.std() + 1e-9)
    zB = (tox - b.mean()) / (b.std() + 1e-9)
    return zA, zB


def llm_nominate(llm, sym, eff, tox, observed_idx, candidate_idx, K, rng, sym_to_idx,
                 target_desc=""):
    """K indices the LLM predicts are high assay_A / low assay_B, from candidate_idx."""
    zA, zB = _zscores(eff, tox, observed_idx)
    cset = set(int(i) for i in candidate_idx)
    cand_shuf = list(rng.permutation(list(candidate_idx)))
    cand_list = ", ".join(sym[i] for i in cand_shuf)        # SHUFFLED (kill ordering artifact)
    obs = _zscore_block(sym, zA, zB, observed_idx)
    ctx = f" in {target_desc} cells" if target_desc else ""
    base = (f"A gene-knockout screen{ctx} measured two readouts per gene:\n"
            f"- assay_A: strength of the knockout effect on the {target_desc or 'target'} cells "
            f"(want HIGH = strong on-target effect)\n"
            f"- assay_B: strength of the same knockout's effect on NON-CANCEROUS reference cells "
            f"(want LOW = spares normal cells)\n"
            f"Both z-scored across the observed sample. OBSERVED genes are sorted best-first "
            f"(high A, low B).\n\nOBSERVED (symbol, assay_A, assay_B):\n{obs}\n\n"
            f"From the CANDIDATES (unmeasured) pick EXACTLY {K} you predict have HIGH assay_A "
            f"and LOW assay_B. The candidate list is long and unordered — consider the WHOLE "
            f"list, not the start. Think briefly (2-3 sentences) about what distinguishes "
            f"high-A/low-B genes, then output a line 'PICKS:' followed by a JSON array of "
            f"exactly {K} symbols from the candidate list.\n\nCANDIDATES:\n{cand_list}\n")
    for att in range(3):
        prompt = base if att == 0 else base + f"\n(attempt {att}: return exactly {K} valid symbols)"
        picks = _parse_picks(llm.complete(_SYSTEM, prompt), sym_to_idx, cset, K)
        if len(picks) >= K:
            return picks[:K]
    # top up with random candidates so downstream eval has K
    extra = [int(i) for i in cand_shuf if int(i) not in set(picks)]
    return (picks + extra)[:K]


def llm_acquire(llm, sym, eff, tox, revealed_idx, candidate_idx, batch, rng, sym_to_idx, rnd,
                target_desc=""):
    """Pick `batch` new genes to assay this round (selectivity-aware acquisition)."""
    zA, zB = _zscores(eff, tox, revealed_idx)
    cset = set(int(i) for i in candidate_idx)
    cand_shuf = list(rng.permutation(list(candidate_idx)))
    cand_list = ", ".join(sym[i] for i in cand_shuf)
    obs = _zscore_block(sym, zA, zB, revealed_idx)
    ctx = f" in {target_desc} cells" if target_desc else ""
    prompt = (f"Active-learning round {rnd}. A gene-knockout screen{ctx} measures assay_A "
              f"(knockout effect on the {target_desc or 'target'} cells, want HIGH) and assay_B "
              f"(effect on NON-CANCEROUS reference cells, want LOW), z-scored across measured "
              f"genes; measured genes sorted best-first.\n\n"
              f"MEASURED (symbol, assay_A, assay_B):\n{obs}\n\n"
              f"Choose EXACTLY {batch} UNMEASURED candidates to assay next that are most likely "
              f"to be high assay_A / low assay_B (you may also probe to reduce uncertainty). "
              f"Consider the WHOLE shuffled list. Think briefly, then output 'PICKS:' and a JSON "
              f"array of exactly {batch} candidate symbols.\n\nCANDIDATES:\n{cand_list}\n")
    picks = _parse_picks(llm.complete(_SYSTEM, prompt), sym_to_idx, cset, batch)
    if len(picks) < batch:
        extra = [int(i) for i in cand_shuf if int(i) not in set(picks)]
        picks = (picks + extra)[:batch]
    return picks[:batch]


# --------------------------------------------------------------------------- #
def _factory():
    from geneal.models.surrogate import GPRSurrogate
    return GPRSurrogate(n_iters=100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="sweep dir (meta.json + ablation.parquet)")
    ap.add_argument("--gene-effect", default="data/processed/depmap/gene_effect.parquet")
    ap.add_argument("--embeddings", default="data/processed/embeddings/pubmedbert_all.parquet")
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--roles", choices=["nom", "loop", "both"], default="nom")
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="default: meta seeds")
    ap.add_argument("--thresh", type=float, default=-0.5)
    ap.add_argument("--model-csv", default="data/raw/depmap/Model.csv")
    ap.add_argument("--cache-dir", default="res/llm_cache")
    ap.add_argument("--out-root", default="res/runs_llm")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--limit-lines", type=int, default=None, help="smoke test: first N lines")
    ap.add_argument("--workers", type=int, default=8, help="concurrent LLM calls within a line")
    args = ap.parse_args()

    rd = Path(args.run_dir)
    meta = json.loads((rd / "meta.json").read_text())
    lines = meta["lines"][: args.limit_lines] if args.limit_lines else meta["lines"]
    contrasts = meta["contrast_lines"]
    seeds = args.seeds or meta["seeds"]
    K, tau, tau_mode = int(meta["K"]), float(meta["tau"]), meta.get("tau_mode", "absolute")
    n_init, n_rounds, batch = int(meta.get("n_rounds_init", meta.get("n_initial", 40))), \
        int(meta["n_rounds"]), int(meta["batch"])
    n_init = int(meta.get("n_initial", 40))
    multi = len(contrasts) > 1
    tox_specs = {(f"contrast_{i}" if multi else "contrast"): ("contrast", c)
                 for i, c in enumerate(contrasts, 1)}
    tox_specs["aggregate"] = ("aggregate", None)
    do_nom = args.roles in ("nom", "both")
    do_loop = args.roles in ("loop", "both")
    print(f"lines={len(lines)} seeds={seeds} K={K} tau={tau} roles={args.roles} model={args.model}")
    print(f"tox_sources={list(tox_specs)}  contrasts={contrasts}")

    ge = load_gene_effect(args.gene_effect)
    emb = pd.read_parquet(args.embeddings)
    llm = LLM(args.model, args.cache_dir)
    # target lineage + cancer type (NOT the exact ModelID) for prompt context
    lin = {}
    if Path(args.model_csv).exists():
        md = pd.read_csv(args.model_csv).set_index("ModelID")
        for cl in lines:
            if cl in md.index:
                a = str(md.loc[cl].get("OncotreeLineage", "")).strip()
                b = str(md.loc[cl].get("OncotreePrimaryDisease", "")).strip()
                lin[cl] = " — ".join(x for x in (a, b) if x and x != "nan")
    print("target descriptions:", {k: lin.get(k, "") for k in lines[:3]}, "...")

    rows = []
    for cl in lines:
        ds, _ = build_selective_dataset(ge, emb, cl, lam=0.0, thresh=args.thresh)
        X = StandardScaler().fit_transform(ds.embeddings)
        eff = np.asarray(ds.target, float)
        names = ds.gene_names
        sym = [n.split(" (")[0] for n in names]
        sym_to_idx = {}
        for i, s in enumerate(sym):
            sym_to_idx.setdefault(s, i)        # first occurrence wins (stable)
        mem = corum_membership(names)
        tox_by = {k: toxicity_vector(ge, names, st, target_line=cl, contrast_line=c,
                                     thresh=args.thresh) for k, (st, c) in tox_specs.items()}
        n = len(eff)
        tdesc = lin.get(cl, "")
        print(f"[{cl}] {n} genes  desc='{tdesc}'")
        # random 120-set per seed, IDENTICAL to the report's `random` acquisition
        rand_by_seed = {seed: [int(i) for i in run_acquisition(
            "random", X, eff, None, seed=seed, n_init=n_init, n_rounds=n_rounds,
            batch=batch, surr_factory=_factory, return_history=True)[1][-1]] for seed in seeds}
        src_list = list(tox_by)

        def work(job, cl=cl, n=n, tdesc=tdesc):
            seed, src = job
            toxv = tox_by[src]; ceil = _tox_threshold(toxv, tau, tau_mode)
            # per-job rng, deterministic + thread-independent (stable src index)
            rng = np.random.default_rng((seed, src_list.index(src), 4242))
            out = []
            if do_nom:
                rand120 = rand_by_seed[seed]
                cand = [i for i in range(n) if i not in set(rand120)]
                picks = llm_nominate(llm, sym, eff, toxv, rand120, cand, K, rng, sym_to_idx,
                                     target_desc=tdesc)
                m = evaluate(picks, eff, toxv, mem, tox_ceiling=ceil)
                out.append(dict(method="llm_nom", cell_line=cl, seed=seed, tox_source=src,
                                picks=[int(i) for i in picks], **m))
                print(f"  [{cl} s{seed} {src}] llm_nom useful={m['useful_efficacy']:.3f} "
                      f"n_safe={m['n_safe']} eff={m['mean_efficacy']:.3f} tox={m['mean_toxicity']:.3f}")
            if do_loop:
                revealed = [int(i) for i in np.random.default_rng(seed).permutation(n)[:n_init]]
                for rnd in range(1, n_rounds + 1):
                    rem = [i for i in range(n) if i not in set(revealed)]
                    revealed += llm_acquire(llm, sym, eff, toxv, revealed, rem, batch,
                                            rng, sym_to_idx, rnd, target_desc=tdesc)
                candL = [i for i in range(n) if i not in set(revealed)]
                picksL = llm_nominate(llm, sym, eff, toxv, revealed, candL, K, rng, sym_to_idx,
                                      target_desc=tdesc)
                mL = evaluate(picksL, eff, toxv, mem, tox_ceiling=ceil)
                out.append(dict(method="llm_loop", cell_line=cl, seed=seed, tox_source=src,
                                picks=[int(i) for i in picksL], **mL))
                print(f"  [{cl} s{seed} {src}] llm_loop useful={mL['useful_efficacy']:.3f} "
                      f"n_safe={mL['n_safe']} eff={mL['mean_efficacy']:.3f} tox={mL['mean_toxicity']:.3f}")
            return out

        jobs = [(seed, src) for seed in seeds for src in src_list]
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for r in ex.map(work, jobs):
                rows.extend(r)

    df = pd.DataFrame(rows)
    run = args.run_name or time.strftime("%Y%m%d-%H%M%S")
    out = Path(args.out_root) / run; out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "llm.parquet")
    _compare(df, rd, out, K, tau)
    print(f"\nrun dir: {out}")


def _ci(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) == 0:
        return float("nan"), float("nan")
    m = float(x.mean())
    c = 0.0 if len(x) < 2 else 1.96 * float(x.std(ddof=1)) / np.sqrt(len(x))
    return m, c


def _pool(df):
    d = df.copy()
    d["tox_source"] = d["tox_source"].astype(str).str.replace(r"^contrast_\d+$", "contrast", regex=True)
    return d


def _compare(llm_df, run_dir, out, K, tau):
    """Print + write a standalone comparison vs the sweep's GP baselines, per facet,
    mean +/- 95% CI over lines x seeds (contrast_* pooled to 'contrast')."""
    ab = pd.read_parquet(run_dir / "ablation.parquet")
    ab = ab[ab.analysis == "A"] if "analysis" in ab.columns else ab
    base_methods = ["greedy", "random", "trunc_pred", "ehvi", "ehvi_pareto", "known_safe"]
    ab = ab[ab.method.isin(base_methods)]
    keep = ["method", "cell_line", "seed", "tox_source",
            "mean_efficacy", "mean_toxicity", "useful_efficacy", "n_safe"]
    both = pd.concat([_pool(ab[keep]), _pool(llm_df[keep])], ignore_index=True)
    both["sel_ratio"] = both["mean_efficacy"] / both["mean_toxicity"].clip(lower=0.05)
    metrics = [("useful_efficacy", "Realizable↑"), ("sel_ratio", "T/NT ratio↑"),
               ("mean_efficacy", "Target eff↑"), ("mean_toxicity", "Non-target↓"),
               ("n_safe", "# safe↑")]
    order = ["known_safe", "llm_loop", "llm_nom", "trunc_pred", "ehvi_pareto",
             "ehvi", "greedy", "random"]
    label = {"known_safe": "oracle (known_safe)", "llm_nom": "LLM (nom)",
             "llm_loop": "LLM (loop)", "trunc_pred": "greedy·nom", "ehvi_pareto": "EHVI-EHVI"}
    html = ["<!DOCTYPE html><meta charset=utf-8><style>body{font-family:system-ui;margin:2rem}"
            "table{border-collapse:collapse;margin-bottom:2rem}td,th{border-bottom:1px solid #ddd;"
            "padding:.35rem .7rem;text-align:right}td:first-child,th:first-child{text-align:left}"
            "tr.llm{background:#eaf2fb}</style>",
            f"<h1>LLM baseline vs GP methods</h1><p>Same protocol (K={K}, τ={tau}); "
            f"mean ± 95% CI over lines × seeds. LLM rows highlighted.</p>"]
    for facet in [f for f in ["contrast", "aggregate"] if f in set(both.tox_source)]:
        d = both[both.tox_source == facet]
        present = [m for m in order if m in set(d.method)]
        print(f"\n=== facet: {facet} ===")
        hdr = f"{'method':22s}" + "".join(f"{lbl:>14s}" for _, lbl in metrics)
        print(hdr)
        html.append(f"<h2>{facet}</h2><table><tr><th>method</th>" +
                    "".join(f"<th>{lbl}</th>" for _, lbl in metrics) + "</tr>")
        for m in present:
            dm = d[d.method == m]
            cells = []
            for col, _ in metrics:
                mn, c = _ci(dm[col])
                cells.append(f"{mn:.3f}±{c:.3f}" if col != "n_safe" else f"{mn:.1f}±{c:.1f}")
            print(f"{label.get(m, m):22s}" + "".join(f"{v:>14s}" for v in cells))
            cls = ' class="llm"' if m.startswith("llm") else ""
            html.append(f"<tr{cls}><td>{label.get(m, m)}</td>" +
                        "".join(f"<td>{v}</td>" for v in cells) + "</tr>")
        html.append("</table>")
    (out / "report.html").write_text("\n".join(html))
    print(f"\ncomparison -> {out / 'report.html'}")


if __name__ == "__main__":
    main()
