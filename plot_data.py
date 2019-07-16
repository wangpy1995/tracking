import pandas as pd
import matplotlib.pyplot as plt


def plot_new():
    filename = 'data/new_10.csv'
    df = pd.read_csv(filename)
    for g in df.groupby('id'):
        s = g[1].sort_values('timestamp')
        plt.plot(s['x'].values, s['y'].values, 'b-')


def plot_old():
    filename = 'data/10.csv'
    df = pd.read_csv(filename)
    for g in df.groupby('id'):
        s = g[1].sort_values('timestamp')
        plt.plot(s['x'].values, s['y'].values, 'r-')


plot_old()
plot_new()
plt.show()
