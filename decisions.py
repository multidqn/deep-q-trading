import pandas as pd
import numpy as np



# Calcola l'ensemable sulle colonne (reti) con il 100% di agreement
def full_ensemble(df):
    # Controllo quali righe hanno tutto 1
    m1 = df.eq(1).all(axis=1)

    # Controllo quali righe hanno tutto 0
    m2 = df.eq(2).all(axis=1)

    # Prevengo sovrascritture di memoria
    local_df = df.copy()
    # Creo una nuova colonna ensemble, mettendo ad 1 se tutte le colonne sono a 1, -1 se sono tutte a 0, 0 altrimenti
    local_df['ensemble'] = np.select([m1, m2], [1, -1], 0)

    # rimuovo tutte le colonne e lascio una sola colonna ensemble che contiene solamente l'operazione da fare (1, -1, 0)
    local_df = local_df.drop(local_df.columns.difference(['ensemble']), axis=1)

    return local_df

# Calcola l'ensemable sulle colonne (reti) con una % di agreement
def perc_ensemble(df, thr = 0.7):
    c1 = (df.eq(1).sum(1) / df.shape[1]).gt(thr)
    c2 = (df.eq(0).sum(1) / df.shape[1]).gt(thr)
    c2.astype(int).mul(-1).add(c1)
    m = pd.DataFrame(np.select([c1, c2], [1, -1], 0), index=df.index, columns=['ensemble'])

    return m


#res = open("risultatiEnsembleValid.csv", "w+")
#res.write("walk,reward,coverage,accuracy,positivi,negativi,euro\n")

numDel=0
df=pd.read_csv("./Output/ensamble/walk0ensamble_test.csv",index_col='Date')

fulldf=full_ensemble(df)

for j in range(0,5):
    df=pd.read_csv("./Output/ensamble/walk"+str(j)+"ensamble_test.csv",index_col='Date')

    for deleted in range(1,numDel):
        del df['iteration'+str(deleted)]

    fulldf=fulldf.append(full_ensemble(df))


fulldf.to_csv("resultEnsembleTest.csv")
fulldf=pd.read_csv("resultEnsembleTest.csv")
fulldf['Date'] = pd.to_datetime(fulldf['Date'])
fulldf = fulldf.set_index('Date')
print(fulldf.head())

fulldf.to_csv("spLong.csv")