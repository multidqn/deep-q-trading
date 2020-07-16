#Copyright (C) 2020 Salvatore Carta, Anselmo Ferreira, Alessandro Sebastian Podda, Diego Reforgiato Recupero, Antonio Sanna. All rights reserved.

#Library used to manipulate the CSV Dataset
#organize the dataset for the Enviroment
import pandas

#Library used to manipulate dates
import datetime

class MergedDataStructure():

    def __init__(self, delta=4, filename="sp500Week.csv"):
        self.delta=delta

        #Read the CSV
        timeserie = pandas.read_csv(filename)
        
        #Transform each column into a list
        Date = timeserie.ix[:, 'Date'].tolist()
        Time = timeserie.ix[:, 'Time'].tolist()
        Open = timeserie.ix[:, 'Open'].tolist()
        High = timeserie.ix[:, 'High'].tolist()
        Low = timeserie.ix[:, 'Low'].tolist()
        Close = timeserie.ix[:, 'Close'].tolist()

        #Create empty list and dictionary
        self.list=[]
        self.dict={}

        #The limit is the number of dates
        limit = len(Date)

        #Just converting pandas data to a list
        #lets pick up the csv data and put them in the list (self.list) 
        for i in range(0,limit-1):
            self.list.append({'Date' : Date[i],'Time' : Time[i], 'Open': Open[i], 'High': High[i], 'Low': Low[i], 'Close': Close[i]})
            
            #Fill the gaps with days that do not exist 
            dateList = [datetime.datetime.strptime(Date[i+1], "%m/%d/%Y") - datetime.timedelta(days=x) for x in range(0, ( datetime.datetime.strptime(Date[i+1], "%m/%d/%Y")- datetime.datetime.strptime(Date[i], "%m/%d/%Y") ).days )]
            
            for date in dateList:
                dateString=date.strftime("%m/%d/%Y")
                #Contains dates and indexes for the list self.list
                self.dict[dateString]=i

    def get(self, date):
        #Converts the date to string
        dateString=str(date)
        #given the date, you get an interval of past days or weeks
        return self.list[self.dict[dateString]-(self.delta):self.dict[dateString]]

