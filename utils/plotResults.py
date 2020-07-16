from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from math import floor
from ensemble import ensemble

#Call it with the name of file plus the number of walks
# python plotResults.py results 2 
numPlots=11
outputFile=str(sys.argv[1])+".pdf"
numEpochs=100
pdf=PdfPages(outputFile)
numFiles=int(sys.argv[2])
plt.figure(figsize=((numEpochs/10)*(numFiles+1),numPlots*5)) 
for i in range(1,numFiles+1):
    document = pd.read_csv("./Output/csv/walks/walks"+str(i)+".csv")
    plt.subplot(numPlots,numFiles,0*numFiles + i )
    #plt.plot(document.ix[:, 'Iteration'].tolist(),document.ix[:, 'testAccuracy'].tolist(),'r',label='Test')
    plt.plot(document.ix[:, 'Iteration'].tolist(),document.ix[:, 'trainAccuracy'].tolist(),'b',label='Train')
    plt.plot(document.ix[:, 'Iteration'].tolist(),document.ix[:, 'validationAccuracy'].tolist(),'g',label='Validation')
    plt.xticks(range(0,numEpochs,4))
    plt.yticks(np.arange(0, 1, step=0.1))
    plt.ylim(-0.05,1.05)
    plt.axhline(y=0, color='k', linestyle='-')
    plt.legend()
    plt.grid()
    plt.title('Walk'+str(i)+'\n\nAccuracy')

    plt.subplot(numPlots,numFiles,1*numFiles + i)
    #plt.plot(document.ix[:, 'Iteration'].tolist(),document.ix[:, 'testCoverage'].tolist(),'r',label='Test')
    plt.plot(document.ix[:, 'Iteration'].tolist(),document.ix[:, 'trainCoverage'].tolist(),'b',label='Train')
    plt.plot(document.ix[:, 'Iteration'].tolist(),document.ix[:, 'validationCoverage'].tolist(),'g',label='Validation')
    plt.xticks(range(0,numEpochs,4))
    plt.yticks(np.arange(0, 1, step=0.1))
    plt.ylim(-0.05,1.05)
    plt.axhline(y=0, color='k', linestyle='-')
    plt.legend()
    plt.grid()
    plt.title('Coverage')


    plt.subplot(numPlots,numFiles,2*numFiles + i )

    plt.plot(document.ix[:, 'Iteration'].tolist(),document.ix[:, 'trainReward'].tolist(),'b',label='Train')
    #plt.plot(document.ix[:, 'Iteration'].tolist(),document.ix[:, 'validationReward'].tolist(),'g',label='Validation')
    #plt.plot(document.ix[:, 'Iteration'].tolist(),document.ix[:, 'testReward'].tolist(),'r',label='Test')
    
    plt.xticks(range(0,numEpochs,4))
    plt.axhline(y=0, color='k', linestyle='-')
    plt.legend()
    plt.grid()
    plt.title('Train Reward')
    

    plt.subplot(numPlots,numFiles,3*numFiles + i )

    #plt.plot(document.ix[:, 'Iteration'].tolist(),document.ix[:, 'trainReward'].tolist(),'b',label='Train')
    plt.plot(document.ix[:, 'Iteration'].tolist(),document.ix[:, 'validationReward'].tolist(),'g',label='Validation')
    #plt.plot(document.ix[:, 'Iteration'].tolist(),document.ix[:, 'testReward'].tolist(),'r',label='Test')
    
    plt.xticks(range(0,numEpochs,4))
    plt.axhline(y=0, color='k', linestyle='-')
    plt.legend()
    plt.grid()
    plt.title('Validation Reward')
    

    plt.subplot(numPlots,numFiles,4*numFiles + i )

    plt.plot(document.ix[:, 'Iteration'].tolist(),document.ix[:, 'trainLong%'].tolist(),'b',label='Train')
    plt.plot(document.ix[:, 'Iteration'].tolist(),document.ix[:, 'validationLong%'].tolist(),'g',label='Validation')
    #plt.plot(document.ix[:, 'Iteration'].tolist(),document.ix[:, 'validationLong%'].tolist(),'r',label='Test')
    
    plt.xticks(range(0,numEpochs,4))
    plt.yticks(np.arange(0, 1, step=0.1))    
    plt.ylim(-0.05,1.05)
    plt.axhline(y=0, color='k', linestyle='-')
    plt.legend()
    plt.grid()
    plt.title('Long %')
    

    plt.subplot(numPlots,numFiles,5*numFiles + i )

    plt.plot(document.ix[:, 'Iteration'].tolist(),document.ix[:, 'trainShort%'].tolist(),'b',label='Train')
    plt.plot(document.ix[:, 'Iteration'].tolist(),document.ix[:, 'validationShort%'].tolist(),'g',label='Validation')
    #plt.plot(document.ix[:, 'Iteration'].tolist(),document.ix[:, 'validationShort%'].tolist(),'r',label='Test')
    
    plt.xticks(range(0,numEpochs,4))

    plt.yticks(np.arange(0, 1, step=0.1))
    plt.ylim(-0.05,1.05)
    plt.axhline(y=0, color='k', linestyle='-')
    plt.legend()
    plt.grid()
    plt.title('Short %')
    

    plt.subplot(numPlots,numFiles,6*numFiles + i )

    #plt.plot(document.ix[:, 'Iteration'].tolist(),list(map(lambda x: 1-x,document.ix[:, 'testCoverage'].tolist())),'r',label='Test')
    plt.plot(document.ix[:, 'Iteration'].tolist(),list(map(lambda x: 1-x,document.ix[:, 'trainCoverage'].tolist())),'b',label='Train')
    plt.plot(document.ix[:, 'Iteration'].tolist(),list(map(lambda x: 1-x,document.ix[:, 'validationCoverage'].tolist())),'g',label='Validation')
    
    plt.xticks(range(0,numEpochs,4))
    
    plt.yticks(np.arange(0, 1, step=0.1))
    plt.ylim(-0.05,1.05)
    plt.axhline(y=0, color='k', linestyle='-')
    plt.legend()
    plt.grid()
    plt.title('Hold %')
    

    plt.subplot(numPlots,numFiles,7*numFiles + i )

    plt.plot(document.ix[:, 'Iteration'].tolist(),document.ix[:, 'trainLongAcc'].tolist(),'b',label='Train')
    plt.plot(document.ix[:, 'Iteration'].tolist(),document.ix[:, 'validationLongAcc'].tolist(),'g',label='Validation')
    #plt.plot(document.ix[:, 'Iteration'].tolist(),document.ix[:, 'validationLongAcc'].tolist(),'r',label='Test')
    
    plt.xticks(range(0,numEpochs,4))
    
    plt.yticks(np.arange(0, 1, step=0.1))
    plt.ylim(-0.05,1.05)
    plt.axhline(y=0, color='k', linestyle='-')
    plt.legend()
    plt.grid()
    plt.title('Long Accuracy')

    

    plt.subplot(numPlots,numFiles,8*numFiles + i )

    plt.plot(document.ix[:, 'Iteration'].tolist(),document.ix[:, 'trainShortAcc'].tolist(),'b',label='Train')
    plt.plot(document.ix[:, 'Iteration'].tolist(),document.ix[:, 'validationShortAcc'].tolist(),'g',label='Validation')
    #plt.plot(document.ix[:, 'Iteration'].tolist(),document.ix[:, 'validationShortAcc'].tolist(),'r',label='Test')
    
    plt.xticks(range(0,numEpochs,4))
    
    plt.yticks(np.arange(0, 1, step=0.1))
    plt.ylim(-0.05,1.05)
    plt.axhline(y=0, color='k', linestyle='-')
    plt.legend()
    plt.grid()
    plt.title('Short Accuracy')

    

    plt.subplot(numPlots,numFiles,9*numFiles + i )

    plt.plot(document.ix[:, 'Iteration'].tolist(),document.ix[:, 'trainLongPrec'].tolist(),'b',label='Train')
    plt.plot(document.ix[:, 'Iteration'].tolist(),document.ix[:, 'validLongPrec'].tolist(),'g',label='Validation')
    #plt.plot(document.ix[:, 'Iteration'].tolist(),document.ix[:, 'validLongPrec'].tolist(),'r',label='Test')
    
    plt.xticks(range(0,numEpochs,4))
    
    plt.yticks(np.arange(0, 1, step=0.1))
    plt.ylim(-0.05,1.05)
    plt.axhline(y=0, color='k', linestyle='-')
    plt.legend()
    plt.grid()
    plt.title('Long Precision')

    

    plt.subplot(numPlots,numFiles,10*numFiles + i )

    plt.plot(document.ix[:, 'Iteration'].tolist(),document.ix[:, 'trainShortPrec'].tolist(),'b',label='Train')
    plt.plot(document.ix[:, 'Iteration'].tolist(),document.ix[:, 'validShortPrec'].tolist(),'g',label='Validation')
    #plt.plot(document.ix[:, 'Iteration'].tolist(),document.ix[:, 'validShortPrec'].tolist(),'r',label='Test')
    
    plt.xticks(range(0,numEpochs,4))
    
    plt.yticks(np.arange(0, 1, step=0.1))
    plt.ylim(-0.05,1.05)
    plt.axhline(y=0, color='k', linestyle='-')
    plt.legend()
    plt.grid()
    plt.title('Short Precision')


plt.suptitle("Esperimento SP500 5 (Only long):\n"
            +"Target model update: 1e-1\n"
            +"Model: 35 neurons single layer\n"
            +"Memory-Window Length: 10000-1\n"
            +"Train length: 5 Years\n"
            +"Validation length: 6 Months\n"
            +"Test lenght: 6 Months\n"
            +"Starting period: 2010-01-01\n"
            +"Other changes: Does only Long actions"
            ,size=30
            ,weight=3
            ,ha='left'
            ,x=0.1
            ,y=0.99)

pdf.savefig()



i=1

###########-------------------------------------------------------------------|Tabella Ensemble|-------------------
x=2
y=1
plt.figure(figsize=(x*5,y*5))


#for i in range(1,floor(x*y/2)+1):
plt.subplot(y,x,i)
plt.axis('off')

val,col=ensemble(numFiles,0,"valid",0)


t=plt.table(cellText=val, colLabels=col, fontsize=30, loc='center')
t.auto_set_font_size(False)
t.set_fontsize(6)
plt.title("Valid")




plt.subplot(y,x,i+1)
plt.axis('off')

val,col=ensemble(numFiles,0,"test",0)


t=plt.table(cellText=val, colLabels=col, fontsize=30, loc='center')
t.auto_set_font_size(False)
t.set_fontsize(6)
plt.title("Test")


plt.suptitle("FULL ENSEMBLE")
pdf.savefig()


###########--------------------------------------------------------------------------------------------------------------------


###########-------------------------------------------------------------------|Tabella Ensemble|-------------------
x=2
y=1
plt.figure(figsize=(x*5,y*5))


#for i in range(1,floor(x*y/2)+1):
plt.subplot(y,x,i)
plt.axis('off')

val,col=ensemble(numFiles,0.9,"valid",0)


t=plt.table(cellText=val, colLabels=col, fontsize=30, loc='center')
t.auto_set_font_size(False)
t.set_fontsize(6)
plt.title("Valid")




plt.subplot(y,x,i+1)
plt.axis('off')

val,col=ensemble(numFiles,0.9,"test",0)


t=plt.table(cellText=val, colLabels=col, fontsize=30, loc='center')
t.auto_set_font_size(False)
t.set_fontsize(6)
plt.title("Test")


plt.suptitle("90% ENSEMBLE")
pdf.savefig()


###########--------------------------------------------------------------------------------------------------------------------


###########-------------------------------------------------------------------|Tabella Ensemble|-------------------
x=2
y=1
plt.figure(figsize=(x*5,y*5))


#for i in range(1,floor(x*y/2)+1):
plt.subplot(y,x,i)
plt.axis('off')

val,col=ensemble(numFiles,0.8,"valid",0)


t=plt.table(cellText=val, colLabels=col, fontsize=30, loc='center')
t.auto_set_font_size(False)
t.set_fontsize(6)
plt.title("Valid")




plt.subplot(y,x,i+1)
plt.axis('off')

val,col=ensemble(numFiles,0.8,"test",0)


t=plt.table(cellText=val, colLabels=col, fontsize=30, loc='center')
t.auto_set_font_size(False)
t.set_fontsize(6)
plt.title("Test")


plt.suptitle('80% ENSEMBLE')
pdf.savefig()


###########--------------------------------------------------------------------------------------------------------------------

###########-------------------------------------------------------------------|Tabella Ensemble|-------------------
x=2
y=1
plt.figure(figsize=(x*5,y*5))


#for i in range(1,floor(x*y/2)+1):
plt.subplot(y,x,i)
plt.axis('off')

val,col=ensemble(numFiles,0.7,"valid",0)


t=plt.table(cellText=val, colLabels=col, fontsize=30, loc='center')
t.auto_set_font_size(False)
t.set_fontsize(6)
plt.title("Valid")




plt.subplot(y,x,i+1)
plt.axis('off')

val,col=ensemble(numFiles,0.7,"test",0)


t=plt.table(cellText=val, colLabels=col, fontsize=30, loc='center')
t.auto_set_font_size(False)
t.set_fontsize(6)
plt.title("Test")


plt.suptitle('70% ENSEMBLE')
pdf.savefig()


###########--------------------------------------------------------------------------------------------------------------------


###########-------------------------------------------------------------------|Tabella Ensemble|-------------------
x=2
y=1
plt.figure(figsize=(x*5,y*5))


#for i in range(1,floor(x*y/2)+1):
plt.subplot(y,x,i)
plt.axis('off')

val,col=ensemble(numFiles,0.6,"valid",0)


t=plt.table(cellText=val, colLabels=col, fontsize=30, loc='center')
t.auto_set_font_size(False)
t.set_fontsize(6)
plt.title("Valid")




plt.subplot(y,x,i+1)
plt.axis('off')

val,col=ensemble(numFiles,0.6,"test",0)


t=plt.table(cellText=val, colLabels=col, fontsize=30, loc='center')
t.auto_set_font_size(False)
t.set_fontsize(6)
plt.title("Test")


plt.suptitle('60% ENSEMBLE')
pdf.savefig()


###########--------------------------------------------------------------------------------------------------------------------

pdf.close()