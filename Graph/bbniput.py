import pandas as pd

dataset = pd.read_csv("csvfile.csv")

current_status = dataset.iloc[-1]["5.Health Status"]

if current_status == "UnSafe Condition":
    bbninput = 'high'
else:
    bbninput = 'low'

print(bbninput)
