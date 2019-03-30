import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from exercise0 import clean_data

# load data
Data = pd.read_excel("Exercise.xlsx", sheetname="Data")
AdStock = pd.read_excel("Exercise.xlsx", sheetname="AdStock")

# 0. Inspect the data - make sure you understand all variables and the objective of this exercise. Is the data "clean"/ meaningfull?									

names = list(Data)
print(names)
# Make functions to easily controll what is executed

data = clean_data(Data)

def linear_model(x, a, b):
	return a * x + b

from scipy.optimize import curve_fit

# plot the best fit line for each media category
def linear_regression(Data,variable):

	ylabel = variable
	zero_point = np.average(Data["Sales"].nsmallest(5))
	Data = Data[Data[variable] > 0]

	plt.style.use('ggplot')
	x = Data["Sales"] - zero_point
	variable = Data[variable]

	popt, pcov = curve_fit(linear_model, x, variable,bounds=([-np.inf,0], np.inf))
	perr = np.sqrt(np.diag(pcov))
	print(popt,perr)
	
	plt.scatter(x + zero_point,variable)
	plt.plot(x + zero_point, linear_model(x, *popt), 'g', label='fit: $ per sale = %5.1f * sales + %5.0f' % tuple(popt), linestyle="--")
	plt.plot(x + zero_point, linear_model(x, *popt + perr), 'g-', linewidth=1, linestyle=":",label=r"1$\sigma$ fit err")
	plt.legend()
	plt.plot(x + zero_point, linear_model(x, *popt - perr), 'g-', linewidth=1, linestyle=":")
	#plt.grid()
	plt.xlabel("Sales")
	plt.ylabel("$ on %s" %ylabel)
	
	plt.show()
	plt.close("all")


linear_regression(Data,"Media spend")
linear_regression(Data,"TV")
linear_regression(Data,"Radio")
linear_regression(Data,"Dailies")



from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize  
from sklearn.model_selection import train_test_split 
from sklearn import metrics  

variables = ["TV","Radio","Dailies"]

def Hyperplane_fit(variables):

	Data = Data = Data[Data["Media spend"] > 0]
	zero_point = np.average(Data["Sales"].nsmallest(5))

	for i in range(0,100):
		x = Data[variables]
		y = Data[["Sales"]] - zero_point

		X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)  



		regressor = LinearRegression()  
		regressor.fit(y_train, X_train)  
		b = regressor.intercept_ 
		a = regressor.coef_ 
		

		plt.style.use('ggplot')
		colors = ["green","blue","orange"]
		for variable in range(0,len(b)):
			plt.plot(y + zero_point, linear_model(y, a[variable][0], b[variable] ),c=colors[variable], linewidth=1.5, linestyle=":",alpha=0.1,label='%s   fit: $ per sale = %5.1f * sales + %5.0f' % (variables[variable],a[variable][0],b[variable]))

		#plt.legend()
	plt.xlabel("Sales")
	plt.ylabel("$ spend")


	X_pred = regressor.predict(y_test)  
	#print(X_pred)
	#df = pd.DataFrame({'Actual': X_test, 'Predicted': X_pred})  
	#print(df)  

	print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(X_test, X_pred))) 

	plt.show()
	plt.close("all")