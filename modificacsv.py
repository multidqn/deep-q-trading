import pandas as pd
from sys import argv


for i in range(1,8):
    nomeFile="./twoWalksVisualize"+str(i)+".csv"
    data = pd.read_csv(nomeFile)
    data.rename(columns={'date': 'iteration'}, inplace=True)
    
    data.set_index("iteration",inplace=True)
    data['Walk']=i
    data['Date']=argv[1]
    data.to_csv("./Output/csv/"+nomeFile)
    print(data)
