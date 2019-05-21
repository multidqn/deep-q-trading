import pandas
import datetime


class MergedDataStructure():


    def __init__(self, delta=4, filename="sp500Week.csv"):
        self.delta=delta
        timeserie = pandas.read_csv(filename)

        Date = timeserie.ix[:, 'Date'].tolist()
        Time = timeserie.ix[:, 'Time'].tolist()
        Open = timeserie.ix[:, 'Open'].tolist()
        High = timeserie.ix[:, 'High'].tolist()
        Low = timeserie.ix[:, 'Low'].tolist()
        Close = timeserie.ix[:, 'Close'].tolist()
        #Volume = timeserie.ix[:, 'Volume'].tolist()

        self.list=[]
        self.dict={}
        limit = len(Date)

        for i in range(0,limit-1):
            self.list.append({'Date' : Date[i],'Time' : Time[i], 'Open': Open[i], 'High': High[i], 'Low': Low[i], 'Close': Close[i]})
            
            dateList = [datetime.datetime.strptime(Date[i+1], "%m/%d/%Y") - datetime.timedelta(days=x) for x in range(0, ( datetime.datetime.strptime(Date[i+1], "%m/%d/%Y")- datetime.datetime.strptime(Date[i], "%m/%d/%Y") ).days )]
            for date in dateList:
                dateString=date.strftime("%m/%d/%Y")
                self.dict[dateString]=i

    def get(self, date):
        dateString=str(date)
        return self.list[self.dict[dateString]-(self.delta-1):self.dict[dateString]+1]

