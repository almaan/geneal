"""Multi-cell-line confirmation of the embedding redundancy finding."""
import argparse, numpy as np, pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import Ridge
from geneal.data.depmap import load_gene_effect, build_cell_line_dataset
from geneal.models.surrogate import GPRSurrogate

def ratio(X, y, seed=0):
    rng=np.random.default_rng(seed); D=squareform(pdist(X)); o=np.argsort(D,axis=1); n=len(y)
    nn_=[np.abs(y[i]-y[o[i,1:11]]).mean() for i in range(n)]
    rd=[np.abs(y[i]-y[rng.choice(n,10)]).mean() for i in range(n)]
    return float(np.mean(nn_)/np.mean(rd))

def main():
    ge=load_gene_effect('data/processed/depmap/gene_effect.parquet')
    embs={'ESM2':'esm2_650m_hvg','scPRINT':'scprint_hvg','PubMedBERT':'pubmedbert_hvg','codep':'codep_hvg'}
    # pick lines: lowest-NaN among the codep panel
    emb0=pd.read_parquet('data/processed/embeddings/esm2_650m_hvg.parquet')
    from geneal.data.depmap import parse_entrez
    embset=set(emb0.index); labs=[g for g in ge.index if parse_entrez(g) in embset]
    lines=ge.loc[labs].isna().sum(0).sort_values().index[:5].tolist()
    print('lines:', lines)
    rows=[]
    for cl in lines:
        for name,f in embs.items():
            emb=pd.read_parquet(f'data/processed/embeddings/{f}.parquet')
            ds=build_cell_line_dataset(ge, emb, cl)
            X,y=ds.embeddings, ds.target
            r=ratio(X,y)
            # quick GP R2 on 70/30
            rng=np.random.default_rng(0); idx=rng.permutation(len(y)); tr=idx[:int(.7*len(y))]; te=idx[int(.7*len(y)):]
            gp=GPRSurrogate(n_iters=100).fit(X[tr],y[tr]); m,_=gp.predict(X[te])
            r2=1-np.sum((y[te]-m)**2)/np.sum((y[te]-y[te].mean())**2)
            rows.append({'cell_line':cl,'emb':name,'ratio':round(r,3),'gp_r2':round(float(r2),3)})
    df=pd.DataFrame(rows)
    print('\n=== redundancy ratio (lower=more outcome structure) ===')
    print(df.pivot(index='cell_line',columns='emb',values='ratio')[['ESM2','scPRINT','PubMedBERT','codep']].to_string())
    print('\n=== GP held-out R2 ===')
    print(df.pivot(index='cell_line',columns='emb',values='gp_r2')[['ESM2','scPRINT','PubMedBERT','codep']].to_string())
    df.to_parquet('data/processed/multiline_diag.parquet')

if __name__=='__main__': main()
