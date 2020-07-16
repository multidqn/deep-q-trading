# Multi-DQN: an Ensemble of Deep Q-Learning Agents for Stock Market Forecasting                                                                                                            

## Abstract 

The stock market forecasting is one of the most challenging application of machine learning, as its historical data are naturally noisy and unstable. Most of the successful approaches act in a supervised manner, labeling training data as being of positive or negative moments of the market. However, training machine learning classifiers in such a way may suffer from over-fitting, since the market behavior depends on several external factors like other markets trends, political events, etc. In this paper, we aim at minimizing such problems by proposing an ensemble of reinforcement learning approaches which do not use annotations (i.e., market goes up or down) to learn, but rather learn how to maximize a return function over the training stage. In order to achieve this goal, we exploit a Q-learning agent trained several times with the same training data and investigate its ensemble behavior in important real-world stock markets. Experimental results in intraday trading indicate better performance than the conventional Buy-and-Hold strategy, which still behaves well in our setups. We also discuss qualitative and quantitative analyses of these results.

## Authors

- Salvatore Carta
- Anselmo Ferreira
- Alessandro Sebastian Podda
- Diego Reforgiato Recupero
- Antonio Sanna

# Info 

## Description

#### These files are needed to run the main code:
* **main.py**: the entry point of the application;
* **deepQTrading.py**: used to organize our data in walks and set up the agents;
* **spEnv.py**: the environment used for the agents;
* **mergedDataStructure.py**: the data structure we use to instantiate the multi-resolution feature vector;
* **callback.py**: a module used to log and trace the results.

#### Other tools:
* **ensemble.py**: can be used to generate the threshold ensemble from the main agents;
* **splitEnsemble.py**: can be used to generate the final ensemble for the LONG+SHORT agent (after running ensemble.py).


If you want to adapt the code and use it for more markets, you can use the file **utils/parserWeek.py**, to create a weekly resolution dataset.<br>
On the other hand, the file **utils/plotResults.py** can be used to generate a .pdf containing several plots, useful to get information on the testing phase of the algorithm.


## Requirements
* Python 3
* Tensorflow (1.15): `pip install tensorflow==1.15`
* Keras: `pip install keras`
* Keras-RL: `pip install keras-rl`
* OpenAI Gym: `pip install gym`
* Pandas: `pip install pandas`

## Usage
The code needs three positional parameters to be correctly executed:<br>
`python main.py <numberOfActions> <isOnlyShort> <ensembleFolder>`<br>
<br>

* To run the **FULL** agent you need to run: `python main.py 3 0 ensembleFolder`
* To run the **ONLY LONG** agent you need to run: `python main.py 2 0 ensembleFolder`
* To run the **ONLY SHORT** agent you need to run: `python main.py 2 1 ensembleFolder`

where the paramenter **ensembleFolder** is used to set the name of the folder in which you'll get your results.
