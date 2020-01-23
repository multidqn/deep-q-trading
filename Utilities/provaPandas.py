import numpy as np
import pandas as pd
import datetime



sp = pd.read_csv('./dataset/sp500Hour.csv')

sp['Datetime'] = pd.to_datetime(sp['Date'] + ' ' + sp['Time'])
sp = sp.set_index('Datetime')
sp = sp.drop(['Date','Time'], axis=1)

spTimeserie = pd.read_csv('./dataset/sp500Hour.csv') # opening the dataset
Date = spTimeserie.ix[:, 'Date'].tolist()
Time = spTimeserie.ix[:, 'Time'].tolist()
Open = spTimeserie.ix[:, 'Open'].tolist()
High = spTimeserie.ix[:, 'High'].tolist()
Low = spTimeserie.ix[:, 'Low'].tolist()
Close = spTimeserie.ix[:, 'Close'].tolist()
Volume = spTimeserie.ix[:, 'Volume'].tolist()

history=[]
for i in range(0,len(Open)): # organizing the dataset as a list of dictionaries
    history.append({'Date' : Date[i],'Time' : Time[i], 'Open': Open[i], 'High': High[i], 'Low': Low[i], 'Close': Close[i], 'Volume': Volume[i] })
        


print("inizio Ricerca")
start=datetime.datetime(2003,12,20,3,0,0)

index=None
while(index is None):
    try:
        index = sp.index.get_loc(start)
    except:
        start+=datetime.timedelta(0,0,0,0,0,1,0)


print(index)