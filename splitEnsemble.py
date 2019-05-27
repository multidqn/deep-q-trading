import pandas as pd


long = [[],[]]
short = [[],[]]

longs=pd.read_csv("./Output/results/spLong.csv")
shorts=pd.read_csv("./Output/results/spShort.csv")

long[0]= longs.ix[:,"Date"].tolist()
long[1]= longs.ix[:,"ensemble"].tolist()
short[0] = shorts.ix[:,"Date"].tolist()
short[1] = shorts.ix[:,"ensemble"].tolist()

output = open("finalEnsembleSP500NEW.csv", "w+")
output.write("date,ensemble\n")

for i in range(0,len(long[0])):
    if(long[0][i]==short[0][i]):
        output.write(str(long[0][i]) + "," + str(long[1][i]+short[1][i]) + "\n")
