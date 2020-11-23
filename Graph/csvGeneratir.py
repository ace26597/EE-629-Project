
import pickle
import os.path
from time import sleep

from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.http import MediaFileUpload
import csv
from sklearn import datasets, linear_model
import warnings
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

from more_itertools import unique_everseen

import csv

#with open('data.csv','r') as f, open('csvfile.csv','w') as out_file:
#    out_file.writelines(unique_everseen(f))

def appendcsv(textline):
    with open('csvfile.csv','a') as file:
        file.write(textline)
        file.close()
        return textline


with open('newData.csv','rt') as text:
    for line in text:
        textline = line
        appendcsv(textline)
        sleep(2)

'''
def appendcsv(textline):
    with open('csvfile.csv','a') as file:
        file.write(textline)
        file.close()
        return textline


with open('data.csv','rt') as text:
    for line in text:
        textline = line
        appendcsv(textline)
        sleep(2)


def appendcsv(textline):
    with open('csvfile.csv','a') as file:
        file.write(row)
        file.close()
        return row

with open('data1.csv','rt') as text:
    seen = set()
    seentwice = set()
    rows = []
    for row in text:
        if (row[0],row[1]) in seen:
            seentwice.add((row[0],row[1]))
        seen.add((row[0],row[1]))
        rows.append(row)
    for row in rows:
        if (row[0],row[1]) not in seentwice:
            print(rows)
            textline = row
            appendcsv(textline)
            sleep(1)


with open('data.csv','r') as f, open('csvfile.csv','w') as out_file:
    out_file.writelines(unique_everseen(f))
    sleep(2)

def appendcsv(textline):
    with open('csvfile.csv','a') as out_file:
            out_file.write(line)
            out_file.close()
            return textline



data = ['1.Time','2.Signal Value','3.Status','4.Value of Shift/Drift','5.Health Status']
with open('csvfile.csv', 'a') as csvFile:
    writer = csv.writer(csvFile,delimiter = ',')
    writer.writerow(data)


data = pd.date_range('1/1/2018', periods=1438, freq='D')

print(data)

dataset=pd.read_csv("data.csv")

dataset['1.Time'] = data.astype(str)

print(dataset)
dataset.to_csv('newData.csv', index = False)

df = pd.read_csv("newData.csv", sep=",")
c = df.select_dtypes(include='1.Time').columns
df[c] = df[c].astype(str)

df.to_csv("2008_data_test.csv_tostring2.csv", index=False)

data = pd.date_range('1/1/2018', periods=1438, freq='D')

print(data)

dataset=pd.read_csv("data.csv")

dataset['6.Date'] = data.astype(str)

print(dataset)
dataset.to_csv('newData.csv', index = False)

df = pd.read_csv("newData.csv", sep=",")
c = df.select_dtypes(include='Time').columns
df[c] = df[c].astype(str)

df.to_csv("2008_data_test.csv_tostring2.csv", index=False)

'''

