import pandas as pd
import matplotlib.pyplot as plt
import sys


numPlots=11
outputFile=str(sys.argv[1])+".pdf"
numFiles=int(sys.argv[2])
plt.figure(figsize=(5*(numFiles+1),numPlots*5)) 
for i in range(1,numFiles+1):
    document = pd.read_csv("walks"+str(i)+".csv")
    plt.subplot(numPlots,numFiles,0*numFiles + i )
    plt.plot(document.ix[:, 'date'].tolist(),document.ix[:, 'testAccuracy'].tolist(),'r',label='Test')
    plt.plot(document.ix[:, 'date'].tolist(),document.ix[:, 'trainAccuracy'].tolist(),'b',label='Train')
    plt.plot(document.ix[:, 'date'].tolist(),document.ix[:, 'validationAccuracy'].tolist(),'g',label='Validation')
    plt.xticks(range(0,50,4))
    plt.ylim(-0.05,1.05)
    plt.axhline(y=0, color='k', linestyle='-')
    plt.legend()
    plt.grid()
    plt.title('Walk'+str(i)+'\n\nAccuracy')

    plt.subplot(numPlots,numFiles,1*numFiles + i)
    plt.plot(document.ix[:, 'date'].tolist(),document.ix[:, 'testCoverage'].tolist(),'r',label='Test')
    plt.plot(document.ix[:, 'date'].tolist(),document.ix[:, 'trainCoverage'].tolist(),'b',label='Train')
    plt.plot(document.ix[:, 'date'].tolist(),document.ix[:, 'validationCoverage'].tolist(),'g',label='Validation')
    plt.xticks(range(0,50,4))
    plt.ylim(-0.05,1.05)
    plt.axhline(y=0, color='k', linestyle='-')
    plt.legend()
    plt.grid()
    plt.title('Coverage')


    plt.subplot(numPlots,numFiles,2*numFiles + i )

    plt.plot(document.ix[:, 'date'].tolist(),document.ix[:, 'trainReward'].tolist(),'b',label='Train')
    #plt.plot(document.ix[:, 'date'].tolist(),document.ix[:, 'validationReward'].tolist(),'g',label='Validation')
    #plt.plot(document.ix[:, 'date'].tolist(),document.ix[:, 'testReward'].tolist(),'r',label='Test')
    
    plt.xticks(range(0,50,4))
    plt.axhline(y=0, color='k', linestyle='-')
    plt.legend()
    plt.grid()
    plt.title('Train Reward')
    

    plt.subplot(numPlots,numFiles,3*numFiles + i )

    #plt.plot(document.ix[:, 'date'].tolist(),document.ix[:, 'trainReward'].tolist(),'b',label='Train')
    plt.plot(document.ix[:, 'date'].tolist(),document.ix[:, 'validationReward'].tolist(),'g',label='Validation')
    plt.plot(document.ix[:, 'date'].tolist(),document.ix[:, 'testReward'].tolist(),'r',label='Test')
    
    plt.xticks(range(0,50,4))
    plt.axhline(y=0, color='k', linestyle='-')
    plt.legend()
    plt.grid()
    plt.title('Validation & Test Reward')
    

    plt.subplot(numPlots,numFiles,4*numFiles + i )

    plt.plot(document.ix[:, 'date'].tolist(),document.ix[:, 'trainLong%'].tolist(),'b',label='Train')
    plt.plot(document.ix[:, 'date'].tolist(),document.ix[:, 'testLong%'].tolist(),'g',label='Validation')
    plt.plot(document.ix[:, 'date'].tolist(),document.ix[:, 'validationLong%'].tolist(),'r',label='Test')
    
    plt.xticks(range(0,50,4))
    
    plt.ylim(-0.05,1.05)
    plt.axhline(y=0, color='k', linestyle='-')
    plt.legend()
    plt.grid()
    plt.title('Long %')
    

    plt.subplot(numPlots,numFiles,5*numFiles + i )

    plt.plot(document.ix[:, 'date'].tolist(),document.ix[:, 'trainShort%'].tolist(),'b',label='Train')
    plt.plot(document.ix[:, 'date'].tolist(),document.ix[:, 'testShort%'].tolist(),'g',label='Validation')
    plt.plot(document.ix[:, 'date'].tolist(),document.ix[:, 'validationShort%'].tolist(),'r',label='Test')
    
    plt.xticks(range(0,50,4))
    
    plt.ylim(-0.05,1.05)
    plt.axhline(y=0, color='k', linestyle='-')
    plt.legend()
    plt.grid()
    plt.title('Short %')
    

    plt.subplot(numPlots,numFiles,6*numFiles + i )

    plt.plot(document.ix[:, 'date'].tolist(),list(map(lambda x: 1-x,document.ix[:, 'testCoverage'].tolist())),'r',label='Test')
    plt.plot(document.ix[:, 'date'].tolist(),list(map(lambda x: 1-x,document.ix[:, 'trainCoverage'].tolist())),'b',label='Train')
    plt.plot(document.ix[:, 'date'].tolist(),list(map(lambda x: 1-x,document.ix[:, 'validationCoverage'].tolist())),'g',label='Validation')
    
    plt.xticks(range(0,50,4))
    
    plt.ylim(-0.05,1.05)
    plt.axhline(y=0, color='k', linestyle='-')
    plt.legend()
    plt.grid()
    plt.title('Hold %')
    

    plt.subplot(numPlots,numFiles,7*numFiles + i )

    plt.plot(document.ix[:, 'date'].tolist(),document.ix[:, 'trainLongAcc'].tolist(),'b',label='Train')
    plt.plot(document.ix[:, 'date'].tolist(),document.ix[:, 'testLongAcc'].tolist(),'g',label='Validation')
    plt.plot(document.ix[:, 'date'].tolist(),document.ix[:, 'validationLongAcc'].tolist(),'r',label='Test')
    
    plt.xticks(range(0,50,4))
    
    plt.ylim(-0.05,1.05)
    plt.axhline(y=0, color='k', linestyle='-')
    plt.legend()
    plt.grid()
    plt.title('Short Accuracy')

    

    plt.subplot(numPlots,numFiles,8*numFiles + i )

    plt.plot(document.ix[:, 'date'].tolist(),document.ix[:, 'trainShortAcc'].tolist(),'b',label='Train')
    plt.plot(document.ix[:, 'date'].tolist(),document.ix[:, 'testShortAcc'].tolist(),'g',label='Validation')
    plt.plot(document.ix[:, 'date'].tolist(),document.ix[:, 'validationShortAcc'].tolist(),'r',label='Test')
    
    plt.xticks(range(0,50,4))
    
    plt.ylim(-0.05,1.05)
    plt.axhline(y=0, color='k', linestyle='-')
    plt.legend()
    plt.grid()
    plt.title('Short Accuracy')

    

    plt.subplot(numPlots,numFiles,9*numFiles + i )

    plt.plot(document.ix[:, 'date'].tolist(),document.ix[:, 'trainLongPrec'].tolist(),'b',label='Train')
    plt.plot(document.ix[:, 'date'].tolist(),document.ix[:, 'testLongPrec'].tolist(),'g',label='Validation')
    plt.plot(document.ix[:, 'date'].tolist(),document.ix[:, 'validLongPrec'].tolist(),'r',label='Test')
    
    plt.xticks(range(0,50,4))
    
    plt.ylim(-0.05,1.05)
    plt.axhline(y=0, color='k', linestyle='-')
    plt.legend()
    plt.grid()
    plt.title('Long Precision')

    

    plt.subplot(numPlots,numFiles,10*numFiles + i )

    plt.plot(document.ix[:, 'date'].tolist(),document.ix[:, 'trainShortPrec'].tolist(),'b',label='Train')
    plt.plot(document.ix[:, 'date'].tolist(),document.ix[:, 'testShortPrec'].tolist(),'g',label='Validation')
    plt.plot(document.ix[:, 'date'].tolist(),document.ix[:, 'validShortPrec'].tolist(),'r',label='Test')
    
    plt.xticks(range(0,50,4))
    
    plt.ylim(-0.05,1.05)
    plt.axhline(y=0, color='k', linestyle='-')
    plt.legend()
    plt.grid()
    plt.title('Short Precision')


plt.suptitle("Esperimento 3: reset tasso di aggiornamento pesi; Train e validation durata 6 mesi ",size='xx-large')

plt.savefig(outputFile,dpi=700)