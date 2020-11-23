import matplotlib.pyplot as plt
import csv
import pandas as pd


dataset=pd.read_csv("data.csv")
x=dataset.iloc[:,0].values
y=dataset.iloc[:,1].values

plt.plot(x,y, label='Loaded from file!')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Graph')
plt.savefig('graph.png')
plt.legend()
plt.show()



