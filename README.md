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
* **main.py**: use this to run the code
* **deepQTrading.py**: here we divide our data in walks and set up our agents
* **spEnv.py**: the environment for our agents
* **mergedDataStructure.py**: the data structure we use to create the multi resolution feature vector
* **callback.py**: a module we use to log all the results

#### Other tools:
* **ensemble.py**: use to get the threshold ensemble from the main agents
* **splitEnsemble.py**: used to get the final ensemble for the LONG+SHORT agent after running ensemble.py


If you want to adapt the code and use it for more markets, you can use the file **utils/parserWeek.py** to create a week resolution dataset.<br>
The file **utils/plotResults.py** can be used to get a pdf containing many plots, useful to get information during testing of the algorithm.


## Requirements
* Python 3
* Tensorflow (1.15): `pip install tensorflow==1.15`
* Keras: `pip install keras`
* Keras-RL: `pip install keras-rl`
* OpenAI Gym: `pip install gym`
* Pandas: `pip install pandas`

## Usage
The code needs 2 positional parameters to run:<br>
`python main.py nummberOfActions isOnlyShort`<br>
<br>

* To run the FULL agent you need to run: `python main.py 3 0`
* To run the ONLY LONG agent you need to run: `python main.py 2 0`
* To run the ONLY SHORT agent you need to run: `python main.py 2 0`
