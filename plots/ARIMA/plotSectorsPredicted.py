import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

file = '/Users/adelekap/Documents/capstone_algo_trading/sectorsPredicted.txt'
with open(file,'r') as pred:
    data = pred.readlines()[0]
    x = data.split(',')

x.append('Industrials')
x.append('Financials')

for i in x:
    if len(i) == 0:
        x.remove(i)

sns.countplot(y=x)
plt.tight_layout()
plt.show()