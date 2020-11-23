
import csv
from sklearn import datasets, linear_model
import warnings
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
warnings.filterwarnings("ignore")


dataset=pd.read_csv("data.csv")
X=dataset.iloc[1:,1].values
y=dataset.iloc[1:,2].values
X1=dataset['1.Time']
X2=dataset['2.Signal Value']
Xf=np.array(list(zip(X1,X2)))
yf=dataset['5.Health Status']
X_train, X_test, y_train, y_test = train_test_split(Xf, yf, test_size = 0.30)

def get_data(file_name):
    data = pd.read_csv(file_name)
    x_parameter = []
    y_parameter = []
    for single_square_feet ,single_price_value in zip(data['1.Time'],data['2.Signal Value']):
        x_parameter.append([float(single_square_feet)])
        y_parameter.append(float(single_price_value))
    return x_parameter,y_parameter


def linear_model_main(X_parameters, Y_parameters, predict_value):
    # Create linear regression object
    regr = linear_model.LinearRegression()
    regr.fit(X_parameters, Y_parameters)
    predict_outcome = regr.predict(predict_value)
    predictions = {}
    Accuracy = regr.score(x, y) * 100
    predictions['intercept'] = regr.intercept_
    predictions['coefficient'] = regr.coef_
    predictions['predicted_value'] = predict_outcome
    return predictions,Accuracy

def show_linear_line(X_parameters,Y_parameters):
    # Create linear regression object
    regr = linear_model.LinearRegression()
    regr.fit(X_parameters, Y_parameters)
    plt.scatter(X_parameters,Y_parameters,color='blue')
    plt.plot(X_parameters,regr.predict(X_parameters),color='red',linewidth=4)
    plt.xticks(())
    plt.yticks(())
    plt.savefig('graph.png')
    plt.show()


x,y = get_data('data.csv')
pred_time = dataset.iloc[-1]["1.Time"]
pred_time = pred_time + 50
new_time = [[pred_time]]

result,Accuracy = linear_model_main(x,y,new_time)

#print( "Intercept value " , result['intercept'])
#print( "coefficient" , result['coefficient'])
print("Prediction for Time =  %s" %pred_time)
print( "Predicted value: ",result['predicted_value'])
print("acccuracy: ",Accuracy)

new_val= float(result['predicted_value'])
predict_val = [[pred_time,new_val]]

model= RandomForestClassifier()
model.fit(X_train, y_train)
predicted= model.predict(predict_val)
y_pred = model.predict(X_test)
print('Health Status : ', predicted)
print('accuracy random forest: ',accuracy_score(y_test, y_pred))


data = [pred_time,new_val,predicted[0]]
with open('predicted.csv', 'a') as csvFile:
    writer = csv.writer(csvFile,delimiter = ',')
    writer.writerow(data)

show_linear_line(x,y)

