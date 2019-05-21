import pandas as pd


long = [[],[]]
short = [[],[]]
long[0]= pd.read_csv("./spLong.csv").ix[:,"Date"].tolist()
long[1]= pd.read_csv("./spLong.csv").ix[:,"ensemble"].tolist()
short[0] = pd.read_csv("./spShort.csv").ix[:,"Date"].tolist()
short[1] = pd.read_csv("./spShort.csv").ix[:,"ensemble"].tolist()

output = open("finalEnsembleSP500.csv", "w+")
output.write("date,ensemble\n")

for i in range(0,len(long[0])):
    if(long[0][i]==short[0][i]):
        output.write(str(long[0][i]) + "," + str(long[1][i]+short[1][i]) + "\n")
