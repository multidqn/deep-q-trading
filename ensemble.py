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
    c2 = (df.eq(2).sum(1) / df.shape[1]).gt(thr)
    c2.astype(int).mul(-1).add(c1)
    m = pd.DataFrame(np.select([c1, c2], [1, 2], 0), index=df.index, columns=['ensemble'])

    return m



# df=pd.read_csv("./Output/ensamble/walk0ensamble_test.csv",index_col='Date')
# fulldf=full_ensemble(df)
# for j in range(1,7):
#     df=pd.read_csv("./Output/ensamble/walk"+str(j)+"ensamble_test.csv",index_col='Date')
#     fulldf=fulldf.append(full_ensemble(df))

# fulldf.to_csv("resultEnsembleTest.csv")
# fulldf=pd.read_csv("resultEnsembleTest.csv")
# fulldf['Date'] = pd.to_datetime(fulldf['Date'])
# fulldf = fulldf.set_index('Date')
# print(fulldf.head())
# fulldf.to_csv("output.csv")






res = open("risultatiEnsembleTest.csv", "w+")
res.write("walk,reward,coverage,accuracy,positivi,negativi,dollari\n")

for j in range(0,7):
    df=pd.read_csv("./Output/ensamble/walk"+str(j)+"ensamble_test.csv",index_col='Date')
    df=perc_ensemble(df)

    print(df.head(10))
    dax=pd.read_csv("./dataset/sp500Day.csv",index_col='Date')
    num=0
    rew=0
    pos=0
    neg=0
    doll=0
    cov=0
    for date, i in df.iterrows():
        num+=1

        if date in dax.index:
            if (i['ensemble']==1):
                pos+= 1 if (dax.at[date,'Close']-dax.at[date,'Open'])/dax.at[date,'Open'] > 0 else 0
                
                neg+= 0 if (dax.at[date,'Close']-dax.at[date,'Open'])/dax.at[date,'Open'] > 0 else 1
                rew+=(dax.at[date,'Close']-dax.at[date,'Open'])/dax.at[date,'Open']
                doll+=(dax.at[date,'Close']-dax.at[date,'Open'])*50
                cov+=1
            elif (i['ensemble']==2):
                
                neg+= 0 if -(dax.at[date,'Close']-dax.at[date,'Open'])/dax.at[date,'Open'] > 0 else 1
                pos+= 1 if -(dax.at[date,'Close']-dax.at[date,'Open'])/dax.at[date,'Open'] > 0 else 0
                rew+=-(dax.at[date,'Close']-dax.at[date,'Open'])/dax.at[date,'Open']
                cov+=1
                doll+=-(dax.at[date,'Close']-dax.at[date,'Open'])*50


    print("reward "+ str(rew))

    print ("coverage " + str(cov/num))

    print ("accuracy " + str(pos/cov))

    print("positives "+ str(pos))

    print("negatives "+ str(neg))

    print ("doll " + str(doll) )

    res.write(str(j)+","+str(rew)+","+str(cov/num)+","+str(pos/cov)+","+str(pos)+","+str(neg)+","+str(doll)+"\n")
