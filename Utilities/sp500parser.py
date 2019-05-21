import pandas
from datetime import datetime
from datetime import timedelta

file = open("daxDay.csv", "w+")
spTimeserie = pandas.read_csv('dax.csv')
file.write("Date,Time,Open,High,Low,Close,Volume\n")

Date = spTimeserie.ix[:, 'Date'].tolist()
Time = spTimeserie.ix[:, 'Time'].tolist()
Open = spTimeserie.ix[:, 'Open'].tolist()
High = spTimeserie.ix[:, 'High'].tolist()
Low = spTimeserie.ix[:, 'Low'].tolist()
Close = spTimeserie.ix[:, 'Close'].tolist()
#Volume = spTimeserie.ix[:, 'Volume'].tolist()
records = []
limit = len(Open)
for i in range(0,limit):
    records.append({'Date' : Date[i],'Time' : Time[i], 'Open': Open[i], 'High': High[i], 'Low': Low[i], 'Close': Close[i] })

opeN = records[0]['Open']
low = records[0]['Low']
high = records[0]['High']
currentTime=records[0]['Date']
#volume=records[0]['Volume']
close=records[0]['Close']
date=records[0]['Date']

for record in records:
    #print(str(currentTime.hour) + "  " + str(datetime.strptime(record['Time'], '%H:%M').hour))
    if(currentTime!=record['Date']):
        file.write(str(date) + ',' + "00:00" + ',' + str(opeN) + ',' + str(high) + ',' + str(low) + ',' + str(close)+'\n')
        high = record['High']
        opeN = record['Open']
        low = record['Low']
        #volume=0
        currentTime=record['Date']

    #volume+=record['Volume']
    close=record['Close']
    if(record['High']>high):
        high=record['High']
    if(record['Low']<low):
        low=record['Low']
    date=record['Date']
    

