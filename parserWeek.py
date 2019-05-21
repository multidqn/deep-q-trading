import pandas
from datetime import datetime
from datetime import timedelta

file = open("daxWeek.csv", "w+")
spTimeserie = pandas.read_csv('daxDay.csv')
file.write("Date,Time,Open,High,Low,Close\n")

Date = spTimeserie.ix[:, 'Date'].tolist()
Time = spTimeserie.ix[:, 'Time'].tolist()
Open = spTimeserie.ix[:, 'Open'].tolist()
High = spTimeserie.ix[:, 'High'].tolist()
Low = spTimeserie.ix[:, 'Low'].tolist()
Close = spTimeserie.ix[:, 'Close'].tolist()
Volume = spTimeserie.ix[:, 'Volume'].tolist()
records = []
limit = len(Open)
for i in range(0,limit):
    records.append({'Date' : Date[i],'Time' : Time[i], 'Open': Open[i], 'High': High[i], 'Low': Low[i], 'Close': Close[i], 'Volume': Volume[i] })
close=0
opeN = records[0]['Open']
low = records[0]['Low']
high = records[0]['High']
currentDate=datetime.strptime(records[0]['Date'], '%m/%d/%Y')
delta = timedelta(days = 7)
volume=0
for record in records:
    
    if(record['High']>high):
        high=record['High']
    if(record['Low']<low):
        low=record['Low']
    nextDate=record['Date']
    print( datetime.strptime(nextDate, '%m/%d/%Y').strftime("%d/%m/%Y") + "  " + (currentDate + delta).strftime("%d/%m/%Y"))
    print(currentDate.strftime("%d/%m/%Y"))
    print(str(datetime.strptime(nextDate, '%m/%d/%Y') >= (currentDate + delta))  + " or " + str(datetime.strptime(nextDate, '%m/%d/%Y') < currentDate) + " = " + str(datetime.strptime(nextDate, '%m/%d/%Y') >= (currentDate + delta) or datetime.strptime(nextDate, '%m/%d/%Y') < currentDate ) )
    if(datetime.strptime(nextDate, '%m/%d/%Y') >= (currentDate + delta) or datetime.strptime(nextDate, '%m/%d/%Y') < currentDate ):
        print("writing")
        if(datetime.strptime(nextDate, '%m/%d/%Y') < currentDate):
            file.write(str(records[-1]['Date']) + ',' + "00:00" + ',' + str(opeN) + ',' + str(high) + ',' + str(low) + ',' + str(close) + '\n')
        else:
            file.write(str(nextDate) + ',' + "00:00" + ',' + str(opeN) + ',' + str(high) + ',' + str(low) + ',' + str(close) + '\n')
        high = record['High']
        opeN = record['Open']
        low = record['Low']
        volume=0
        currentDate=datetime.strptime(nextDate, '%m/%d/%Y')
    volume+=record['Volume']
    close=record['Close']
print("Done")
