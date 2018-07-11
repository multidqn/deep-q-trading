import argparse
import json

import matplotlib.pyplot as plt


def visualize_log(filename, figsize=None, output=None):
    with open(filename, 'r') as f:
        data = json.load(f)
    plt.plot(data['episode'],data['episode_reward'])
    plt.grid()
    plt.show()

visualize_log("Intraday.json")