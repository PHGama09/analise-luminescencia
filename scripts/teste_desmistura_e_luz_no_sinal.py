"""Teste de duas correções candidatas (análise final de 05/11/2026). SÓ ANÁLISE: não grava nada, não muda o pipeline.

Uso (na raiz, venv ativo):  python scripts/teste_desmistura_e_luz_no_sinal.py

  1. Desmistura dos canais: inverte a matriz de vazamento de cor da câmera (medida nos pós únicos G1.5 e B1.5), quadro a quadro,
     e recalcula os descritores com descritores.calcular.
  2. Correção da luz de fundo no SINAL: desconta do sinal corrigido uma luz proporcional ao canto (inclinação da cauda contra o
     canto, dentro das preparações) e recalcula os descritores.
Critério (igual ao da etapa luz): separação entre composições = variância entre composições / (sessão + regravação), em ln, só em
A+B, CRUZADA (o que se aplica em A é calibrado em B e vice-versa). A amostra C nunca é lida. Inclui a comparação justa (mesmos
vídeos nas duas versões) e uma checagem física da desmistura (o t50 do pó puro deveria ser igual em todas as misturas;
prop_* só agrupa os vídeos nessa checagem, nunca entra em correção nenhuma).
Resultado de 05/11/2026: as duas correções foram rejeitadas (README, "Limites que não se resolvem em código", itens 2 e 9).
"""
import pandas as pd, numpy as np, yaml, io, contextlib, time
from luminescencia import descritores as D, luz as L
E='utf-8-sig'
cfg=yaml.safe_load(open('config.yaml',encoding='utf-8'))
ser=pd.read_csv('output/posprocessamento/series_pos.csv',encoding=E,low_memory=False)
vpos=pd.read_csv('output/posprocessamento/videos_pos.csv',encoding=E,low_memory=False)
ds=pd.read_csv('output/tabela/dataset.csv',encoding=E,low_memory=False)
vid=pd.read_csv('output/consolidacao/videos.csv',encoding=E,low_memory=False)
ser=ser[ser.amostra.isin(['A','B'])]; vpos=vpos[vpos.amostra.isin(['A','B'])]          # C nunca entra
def calc(s):
    with contextlib.redirect_stdout(io.StringIO()): return D.calcular(s,vpos,cfg)
t0=time.time(); base=calc(ser); print('descritores originais', base.shape, f'{time.time()-t0:.0f}s')
COLS=[f'{c}_md_{g}' for c in 'RGB' for g in ('t50','t10','r_2s','r_5s','r_10s','auc_10s')]+['razao_t50_R_G','razao_t50_G_B','razao_t50_R_B']
D4=[f'{c}_md_{g}' for c in 'RGB' for g in ('t50','r_5s','auc_10s')]+['razao_t50_R_G','razao_t50_G_B','razao_t50_R_B']
meta=ds.set_index('video')[['amostra','proporcao','excluido_saturacao']]
def separacao(desc, cols):
    x=desc.set_index('video').join(meta,rsuffix='_m'); x=x[~x.excluido_saturacao.astype(bool)]
    out={}
    for col in cols:
        if col not in x: continue
        s=x[[col,'amostra_m' if 'amostra_m' in x else 'amostra','proporcao_m' if 'proporcao_m' in x else 'proporcao']].copy(); s.columns=['v','a','p']
        s=s[s.v>0].dropna(); s=s[s.groupby('p').a.transform('nunique')==2]
        prep=s.p+'|'+s.a; s=s[prep.map(prep.value_counts())>=2]; prep=s.p+'|'+s.a
        y=np.log(s.v); mc=y.groupby(s.p).transform('mean'); mp=y.groupby(prep).transform('mean')
        out[col]=(mc.var(ddof=0)/((mp-mc).var(ddof=0)+(y-mp).var(ddof=0)), len(s))
    return out
sep0=separacao(base,COLS)
# ---------- 9. desmistura dos canais (3x3), cross-fit
def matriz(amostra):
    b=base.merge(meta[['amostra']].rename(columns={'amostra':'am'}),left_on='video',right_index=True)
    b=b[b.am==amostra]
    q=lambda prop,num,den: float((b[b.proporcao==prop][f'{num}_md_I0']/b[b.proporcao==prop][f'{den}_md_I0']).median())
    # colunas = pós (r,g,b); linhas = canais (R,G,B); pó r só aparece no R
    return np.array([[1.0, q('G1.5','R','G'), q('B1.5','R','B')],[0.0,1.0,q('B1.5','G','B')],[0.0,q('G1.5','B','G'),1.0]])
des=ser.copy()
for alvo,calib in (('A','B'),('B','A')):
    M=matriz(calib); Mi=np.linalg.inv(M); m=des.amostra==alvo
    obs=des.loc[m,['corr_R','corr_G','corr_B']].to_numpy().T; pur=Mi@obs
    des.loc[m,'corr_R'],des.loc[m,'corr_G'],des.loc[m,'corr_B']=pur
    print(f'matriz de vazamento calibrada em {calib}:',np.round(M,3).tolist())
t0=time.time(); dmix=calc(des); print('descritores desmisturados', f'{time.time()-t0:.0f}s')
sep9=separacao(dmix,COLS)
# ---------- 2. correção da luz no SINAL (aditiva), cross-fit
v=vid.set_index('video'); canto=v[[f'piso_canto_{c}' for c in 'RGB']].mean(axis=1)
x=ds[~ds.excluido_saturacao.astype(bool)&ds.amostra.isin(['A','B'])].set_index('video'); x['canto']=canto.reindex(x.index); x['prep']=x.proporcao+'|'+x.amostra
def inclin(am,c):
    s=x[(x.amostra==am)&x[f'sinal_presente_{c}'].astype(bool)].dropna(subset=[f'final_corr_{c}'])
    yw=s[f'final_corr_{c}']-s.groupby('prep')[f'final_corr_{c}'].transform('mean'); xw=s.canto-s.groupby('prep').canto.transform('mean')
    return float((xw*yw).sum()/(xw**2).sum()), float(s.canto.median())
sig=ser.copy()
for alvo,calib in (('A','B'),('B','A')):
    m=sig.amostra==alvo; cv=sig.loc[m,'video'].map(canto)
    for c in 'RGB':
        a,ref=inclin(calib,c); sig.loc[m,f'corr_{c}']=sig.loc[m,f'corr_{c}']-a*(cv-ref)
        print(f'luz no sinal: canal {c}, calibrado em {calib}: {a:+.3f} nível de sinal por nível de canto')
t0=time.time(); dsig=calc(sig); print('descritores com correção no sinal', f'{time.time()-t0:.0f}s')
sep2=separacao(dsig,COLS)
# ---------- correção atual (medidas), cross-fit, nas mesmas colunas
p=L.parametros(cfg); p={**p,'grandezas_canal':[g for g in ('md_t50','md_t10','md_r_2s','md_r_5s','md_r_10s','md_auc_10s')]}
t,cols=L.montar_tabela(ds[ds.amostra.isin(['A','B'])],base,vid,p); sl=L.separacao_cruzada(t,cols,['A','B'],p).set_index('coluna')
def ganho(sep, col): return (sep[col][0]/sep0[col][0]-1)*100 if col in sep and col in sep0 else np.nan
r=pd.DataFrame({'col':COLS})
r['separacao_original']=[sep0.get(c,(np.nan,))[0] for c in COLS]
r['desmistura_%']=[ganho(sep9,c) for c in COLS]; r['luz_no_sinal_%']=[ganho(sep2,c) for c in COLS]
r['luz_nas_medidas_%']=[sl.loc[c,'ganho_pct'] if c in sl.index else np.nan for c in COLS]
pd.set_option('display.width',200); print(r.round(2).to_string(index=False))
for nome,colsel in (('todas',COLS),('D4',D4)):
    q=r[r.col.isin(colsel)]
    print(f'{nome}: mediana do ganho  desmistura {q["desmistura_%"].median():+.1f}% (melhora em {(q["desmistura_%"]>0).sum()}/{q["desmistura_%"].notna().sum()}) | luz no sinal {q["luz_no_sinal_%"].median():+.1f}% ({(q["luz_no_sinal_%"]>0).sum()}/{q["luz_no_sinal_%"].notna().sum()}) | luz nas medidas (atual) {q["luz_nas_medidas_%"].median():+.1f}% ({(q["luz_nas_medidas_%"]>0).sum()}/{q["luz_nas_medidas_%"].notna().sum()})')
for nome,d in (('original',base),('desmistura',dmix),('luz no sinal',dsig)):
    print(f'  vídeos×canais com t50 ({nome}):',{c:int(d[f"{c}_md_t50"].notna().sum()) for c in 'RGB'})

print('\n=== COMPARAÇÃO JUSTA: só os vídeos que têm valor nas duas versões ===')
def sep_comum(d1,d2,cols):
    a=d1.set_index('video'); b=d2.set_index('video'); res={}
    for col in cols:
        if col not in a or col not in b: continue
        idx=a.index[(a[col]>0)&(b.reindex(a.index)[col]>0)]
        s1=separacao(a.loc[idx,[col]].reset_index(),[col]); s2=separacao(b.loc[idx,[col]].reset_index(),[col])
        if col in s1 and col in s2: res[col]=((s2[col][0]/s1[col][0]-1)*100, s1[col][1])
    return res
for nome,d in (('desmistura',dmix),('luz no sinal',dsig)):
    g=sep_comum(base,d,COLS); q=pd.Series({k:v[0] for k,v in g.items()})
    print(f'{nome}: mediana {q.median():+.1f}%, melhora em {(q>0).sum()}/{len(q)} | D4 {q[q.index.isin(D4)].median():+.1f}% ({(q[q.index.isin(D4)]>0).sum()}/{q.index.isin(D4).sum()})')
    print('   ',{k:(round(v[0],1),v[1]) for k,v in g.items()})

print('\n=== CHECAGEM FÍSICA: t50 por composição (mediana), original x desmisturado ===')
m=ds.set_index('video')[['proporcao','prop_R','prop_G','prop_B','excluido_saturacao']]
for c,pc in (('G','prop_G'),('R','prop_R')):
    a=base.set_index('video')[f'{c}_md_t50']; b=dmix.set_index('video')[f'{c}_md_t50']
    t=pd.DataFrame({'orig':a,'desm':b}).join(m); t=t[~t.excluido_saturacao.astype(bool)&(t[pc]>0)]
    g=t.groupby('proporcao').agg(fracao=(pc,'first'),orig=('orig','median'),desm=('desm','median'),n=('desm','count')).sort_values('fracao')
    print(f'canal {c} (só composições com o pó {c}):'); print(g.round(2).to_string())
    print(f'   correlação entre a fração do pó {c} e o t50: original {t[[pc,"orig"]].corr(method="spearman").iloc[0,1]:+.2f}, desmisturado {t[[pc,"desm"]].corr(method="spearman").iloc[0,1]:+.2f}')
