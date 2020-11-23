import matplotlib.pyplot as plt
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn import datasets, linear_model

dataset=pd.read_csv("data.csv")
X=dataset.iloc[1:,1].values
y=dataset.iloc[1:,2].values

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
    plt.show()

x,y = get_data('data.csv')
predict_value = [[1450]]
result,Accuracy = linear_model_main(x,y,predict_value)

print( "Intercept value " , result['intercept'])
print( "coefficient" , result['coefficient'])
print( "Predicted value: ",result['predicted_value'])
print("acccuracy: ",Accuracy)

show_linear_line(x,y)