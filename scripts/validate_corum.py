import pandas as pd, numpy as np, re, json
from itertools import combinations
np.random.seed(0)

BASE = '/cv/data/braid/andera29/projs/gene-al'
PROC = f'{BASE}/data/processed/depmap'

# ---- panel ----
with open(f'{PROC}/panel_hvg.txt') as f:
    panel = [int(x) for x in f.read().split()]
panel = list(dict.fromkeys(panel))  # dedupe preserve order
print('panel size', len(panel))

# ---- gene_effect: map entrez -> 'SYM (entrez)' label, and entrez->symbol ----
ge = pd.read_parquet(f'{PROC}/gene_effect.parquet')
ent2label = {}
ent2sym = {}
pat = re.compile(r'^(.*) \((\d+)\)$')
for lab in ge.index:
    m = pat.match(lab)
    if m:
        sym, ent = m.group(1), int(m.group(2))
        ent2label[ent] = lab
        ent2sym[ent] = sym
# symbol -> entrez (panel only) for CORUM mapping by symbol
panel_set = set(panel)
sym2ent_panel = {ent2sym[e]: e for e in panel if e in ent2sym}
print('panel genes with symbol from gene_effect:', len(sym2ent_panel))

# ---- CORUM ----
corum = pd.read_csv(f'{BASE}/data/corum_dl/humanComplexes.txt', sep='\t', dtype=str)
print('CORUM rows', len(corum), 'cols include subunits_gene_name?', 'subunits_gene_name' in corum.columns)

# parse complexes -> list of panel entrez members
complexes = []  # list of (complex_id, set(entrez panel members))
for _, row in corum.iterrows():
    raw = row.get('subunits_gene_name')
    if not isinstance(raw, str) or not raw.strip():
        continue
    syms = [s.strip() for s in raw.split(';') if s.strip()]
    ents = set()
    for s in syms:
        if s in sym2ent_panel:
            ents.add(sym2ent_panel[s])
    if len(ents) >= 1:
        complexes.append((row['complex_id'], ents))

# ---- build co-membership similarity over panel ----
N = len(panel)
idx = {e: i for i, e in enumerate(panel)}
S = np.zeros((N, N), dtype=np.float32)      # binary co-membership
W = np.zeros((N, N), dtype=np.float32)      # count shared complexes
partners = {e: set() for e in panel}        # entrez -> set of co-complex partners
covered = set()
for cid, ents in complexes:
    el = sorted(ents)
    for e in el:
        covered.add(e)
    for a, b in combinations(el, 2):
        ia, ib = idx[a], idx[b]
        S[ia, ib] = S[ib, ia] = 1.0
        W[ia, ib] += 1; W[ib, ia] += 1
        partners[a].add(b); partners[b].add(a)
np.fill_diagonal(S, 0); np.fill_diagonal(W, 0)

coverage = len(covered)
n_pairs = N * (N - 1) / 2
n_co = int(S.sum() / 2)
density = n_co / n_pairs
print(f'COVERAGE: {coverage}/{N} panel genes in >=1 CORUM complex ({coverage/N:.3f})')
print(f'DENSITY: {n_co} co-complexed pairs / {int(n_pairs)} = {density:.6f}')

# weighted normalized version: W / max(W) (simple), keep raw counts too
Wn = W / (W.max() if W.max() > 0 else 1)

# save
S_df = pd.DataFrame(S, index=panel, columns=panel)
S_df.to_parquet(f'{PROC}/corum_sim.parquet')
# also save weighted alongside (separate file to keep binary clean as requested primary)
pd.DataFrame(W, index=panel, columns=panel).to_parquet(f'{PROC}/corum_sim_weighted.parquet')
print('saved corum_sim.parquet (binary) and corum_sim_weighted.parquet (counts)')

# ===================== VALIDATION =====================
results = {}
results['corum_version'] = '5.3'
results['coverage'] = [coverage, N, coverage/N]
results['density'] = density
results['n_co_pairs'] = n_co

# lethality in ACH-000147
CL = 'ACH-000147'
eff = ge[CL]  # series indexed by label
# panel genes that have an effect value (proxy for "with embeddings" universe = panel)
lethal = {}  # entrez -> lethality (=-effect)
for e in panel:
    lab = ent2label.get(e)
    if lab is not None and lab in eff.index and pd.notna(eff[lab]):
        lethal[e] = -eff[lab]
panel_with_y = [e for e in panel if e in lethal]
print('panel genes with lethality in', CL, ':', len(panel_with_y))

# ---- HEADLINE: NN(CORUM partner) vs random lethality-diff ratio ----
rng = np.random.default_rng(42)
pwy_set = set(panel_with_y)
nn_diffs, rand_diffs = [], []
n_with_partner = 0
for e in panel_with_y:
    ps = [p for p in partners[e] if p in pwy_set]
    if not ps:
        continue
    n_with_partner += 1
    ye = lethal[e]
    nn_diffs.append(np.mean([abs(ye - lethal[p]) for p in ps]))
    # 10 random genes (from panel_with_y, excluding self)
    cands = [g for g in panel_with_y if g != e]
    rsamp = rng.choice(cands, size=min(10, len(cands)), replace=False)
    rand_diffs.append(np.mean([abs(ye - lethal[g]) for g in rsamp]))
nn_mean = float(np.mean(nn_diffs)); rand_mean = float(np.mean(rand_diffs))
ratio = nn_mean / rand_mean
results['headline'] = {'genes_with_corum_partner': n_with_partner,
                       'genes_excluded_no_partner': len(panel_with_y) - n_with_partner,
                       'nn_mean_lethdiff': nn_mean, 'rand_mean_lethdiff': rand_mean,
                       'ratio': ratio}
print(f'HEADLINE ratio (CORUM NN / random) = {ratio:.3f}  (nn={nn_mean:.4f} rand={rand_mean:.4f}); n_genes_with_partner={n_with_partner}, excluded={len(panel_with_y)-n_with_partner}')

# ---- profile cosine across OTHER cell lines ----
other_cols = [c for c in ge.columns if c != CL]
# build profile matrix for panel_with_y genes
labs = [ent2label[e] for e in panel_with_y]
prof = ge.loc[labs, other_cols].copy()
# fill NaN with row mean then center? use lethality = -effect; cosine on raw profiles fine. drop genes with all-NaN
prof = prof.apply(lambda r: r.fillna(r.mean()), axis=1)
prof_vals = prof.values.astype(np.float64)
# normalize rows for cosine
norms = np.linalg.norm(prof_vals, axis=1, keepdims=True)
norms[norms == 0] = 1
profn = prof_vals / norms
ent_order = panel_with_y
eidx = {e: i for i, e in enumerate(ent_order)}

# co-complexed pairs among panel_with_y
co_cos, rand_cos = [], []
co_pairs = []
for cid, ents in complexes:
    el = [e for e in ents if e in eidx]
    for a, b in combinations(sorted(el), 2):
        co_pairs.append((a, b))
co_pairs = list(set(co_pairs))
for a, b in co_pairs:
    co_cos.append(float(profn[eidx[a]] @ profn[eidx[b]]))
# random pairs, same count
M = len(co_pairs)
allg = ent_order
for _ in range(M):
    a, b = rng.choice(len(allg), size=2, replace=False)
    rand_cos.append(float(profn[a] @ profn[b]))
co_cos_m = float(np.mean(co_cos)); rand_cos_m = float(np.mean(rand_cos))
cos_ratio = co_cos_m / rand_cos_m if rand_cos_m != 0 else float('nan')
results['profile_cosine'] = {'n_co_pairs': M, 'co_mean': co_cos_m, 'rand_mean': rand_cos_m, 'ratio': cos_ratio}
print(f'PROFILE COSINE: co-complex mean={co_cos_m:.4f} random mean={rand_cos_m:.4f} ratio={cos_ratio:.3f} (n_pairs={M})')

# ---- mechanism-density: top-50 lethal genes ----
top50 = sorted(panel_with_y, key=lambda e: lethal[e], reverse=True)[:50]
top50_set = set(top50)
# distinct CORUM complexes spanned (complexes containing >=1 of top50; also count >=2 for co-membership)
comp_members = {}  # cid -> set of top50 entrez in it
for cid, ents in complexes:
    hit = ents & top50_set
    if hit:
        comp_members[cid] = comp_members.get(cid, set()) | hit
# group top50 by co-complex connectivity (connected components via S among top50)
# build adjacency among top50
import collections
adj = collections.defaultdict(set)
for cid, hit in comp_members.items():
    hl = sorted(hit)
    for a, b in combinations(hl, 2):
        adj[a].add(b); adj[b].add(a)
# connected components
seen = set(); comps = []
for g in top50:
    if g in seen: continue
    stack = [g]; comp = []
    while stack:
        x = stack.pop()
        if x in seen: continue
        seen.add(x); comp.append(x)
        stack.extend(adj[x] - seen)
    comps.append(comp)
comp_sizes = sorted([len(c) for c in comps], reverse=True)
n_singletons = sum(1 for c in comps if len(c) == 1)
# also: complexes with >=2 top50 members
multi_complexes = {cid: m for cid, m in comp_members.items() if len(m) >= 2}
results['mechanism_top50'] = {
    'n_top50': len(top50),
    'top50_in_any_complex': len(top50_set & covered),
    'n_connected_components': len(comps),
    'component_sizes': comp_sizes,
    'n_singletons': n_singletons,
    'n_corum_complexes_with_ge2_top50': len(multi_complexes),
    'multi_complex_sizes': sorted([len(m) for m in multi_complexes.values()], reverse=True),
}
print(f'MECHANISM top-50: {len(top50_set & covered)}/50 in >=1 complex; '
      f'{len(comps)} connected components (sizes {comp_sizes[:10]}...); '
      f'{n_singletons} singletons; {len(multi_complexes)} complexes with >=2 top50 members')

json.dump(results, open(f'{BASE}/data/corum_dl/results.json', 'w'), indent=1)
print('\nDONE. results.json written.')
