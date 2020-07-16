# Multi-DQN: an Ensemble of Deep Q-Learning Agents for Stock Market Forecasting

                                                                                                                  
                                                                                                                  
________                                      ____        __________                   ___                        
`MMMMMMMb.                                   6MMMMb       MMMMMMMMMM                   `MM 68b                    
 MM    `Mb                                  8P    Y8      /   MM   \                    MM Y89                    
 MM     MM   ____     ____  __ ____        6M      Mb         MM ___  __    ___     ____MM ___ ___  __     __     
 MM     MM  6MMMMb   6MMMMb `M6MMMMb       MM      MM         MM `MM 6MM  6MMMMb   6MMMMMM `MM `MM 6MMb   6MMbMMM 
 MM     MM 6M'  `Mb 6M'  `Mb MM'  `Mb      MM      MM         MM  MM69 " 8M'  `Mb 6M'  `MM  MM  MMM9 `Mb 6M'`Mb   
 MM     MM MM    MM MM    MM MM    MM      MM      MM         MM  MM'        ,oMM MM    MM  MM  MM'   MM MM  MM   
 MM     MM MMMMMMMM MMMMMMMM MM    MM      MM      MM MMMMMMM MM  MM     ,6MM9'MM MM    MM  MM  MM    MM YM.,M9   
 MM     MM MM       MM       MM    MM      YM      M9         MM  MM     MM'   MM MM    MM  MM  MM    MM  YMM9    
 MM    .M9 YM    d9 YM    d9 MM.  ,M9       8b    d8          MM  MM     MM.  ,MM YM.  ,MM  MM  MM    MM (M       
_MMMMMMM9'  YMMMM9   YMMMM9  MMYMMM9         YMMMM9          _MM__MM_    `YMMM9'Yb.YMMMMMM__MM__MM_  _MM_ YMMMMb. 
                             MM                MM                                                        6M    Yb 
                             MM                YM.                                                       YM.   d9 
                            _MM_                `Mo                                                       YMMMM9  

## Abstract 

The stock market forecasting is one of the most challenging application of machine learning, as its historical data are naturally noisy and unstable. Most of the successful approaches act in a supervised manner, labeling training data as being of positive or negative moments of the market. However, training machine learning classifiers in such a way may suffer from over-fitting, since the market behavior depends on several external factors like other markets trends, political events, etc. In this paper, we aim at minimizing such problems by proposing an ensemble of reinforcement learning approaches which do not use annotations (\textit{i.e.} market goes up or down) to learn, but rather learn how to maximize a return function over the training stage. In order to achieve this goal, we exploit a Q-learning agent trained several times with the same training data and investigate its ensemble behavior in important real-world stock markets. Experimental results in intraday trading indicate better performance than the conventional Buy-and-Hold strategy, which still behaves well in our setups. We also discuss qualitative and quantitative analyses of these results.

## Authors: 

- Salvatore Carta
- Anselmo Ferreira
- Alessandro Sebastian Podda
- Diego Reforgiato Recupero
- Antonio Sanna

# Info 

README under construction. This section will be completed in the next few hours.
