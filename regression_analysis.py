import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# load data
Data = pd.read_excel("Exercise.xlsx", sheetname="Data")
AdStock = pd.read_excel("Exercise.xlsx", sheetname="AdStock")

# 0. Inspect the data - make sure you understand all variables and the objective of this exercise. Is the data "clean"/ meaningfull?									

names = list(Data)
print(names)
# Make functions to easily controll what is executed



def Exercise_0(Data):
	Media_data = Data.drop(columns=["Campaign type 1", "Campaign type 2", "Campaign type 3"])#, "Competitor 1 Spend", "Competitor 2 Spend"])

	correlations = Media_data.corr()
	
	# plot correlation matrix
	fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(16, 12))
	fig1 = ax1.matshow(correlations,cmap="PuOr", vmin=-1, vmax=1)
	fig.colorbar(fig1, ax=ax1, shrink=0.45)
	ticks = np.arange(0,8,1)
	ax1.set_xticks(ticks)
	ax1.set_yticks(ticks)
	ax1.set_xticklabels(names[1:7]+names[10:],rotation='vertical')
	ax1.set_yticklabels(names[1:7]+names[10:])


	ax2.plot(Data["Week"],Data["Media spend"],color="k",linestyle="-",linewidth=2,label="Media spend")
	ax2.plot(Data["Week"],Data["TV"],color="g",linestyle="--",linewidth=1.2,label="TV")
	ax2.plot(Data["Week"],Data["Radio"],color="b",linestyle="-.",linewidth=1.2,label="Radio")
	ax2.plot(Data["Week"],Data["Dailies"],color="purple",linestyle=":",linewidth=1.2,label="Dailies")
	ax2.plot(Data["Week"],Data["Dailies"] + Data["Radio"] + Data["TV"],color="r",linestyle="--",linewidth=2,label="TV + Radio + Dailies")
	ax2.legend()
	ax2.set_ylim(0,3.5e6)
	ax2.set_ylabel("Total amount spend")
	ax2.set_xlabel("Year - Week")
	ticks = np.linspace(0,80,7)
	ax2.set_xticks(ticks)
	ax2.grid()

	plt.show()


# 1. Make a simple model that describes sales with media included and quantify the quality of the model							

def clean_data(Data):

	#remove negatives and unrealisticly small numbers
	num = Data._get_numeric_data()
	num[num < 10] = 0

	# require total spend to be the sum of its individual contributions
	for i,row in Data.iterrows():
		
		if row["Media spend"] != sum((row["TV"], row["Radio"], row["Dailies"])):
			row["Media spend"] = sum((row["TV"], row["Radio"], row["Dailies"]))

	print("Done cleaning")
	return Data

Data = clean_data(Data)
Exercise_0(Data)

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

"""
linear_regression(Data,"Media spend")
linear_regression(Data,"TV")
linear_regression(Data,"Radio")
linear_regression(Data,"Dailies")
"""


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

def adstock_chi(B, x):

	A_i = 0
	chi_2 = 0

	for i in range(0,len(x)):
		A_i = x[i] + B[1] * A_i

		
		chi_2 += abs(Data["Sales"][i] - B[0] - A_i * B[2])**2 / Data["Sales"][i]
	return chi_2

def adstock_chi2(B, x1, x2, x3):

	A_i1 = 0
	A_i2 = 0
	A_i3 = 0

	chi_2 = 0

	for i in range(0,len(x1)):
		A_i1 = x1[i] + B[0] * A_i1
		A_i2 = x2[i] + B[2] * A_i2
		A_i3 = x3[i] + B[4] * A_i3
		
		A_i_sum = (A_i1 * B[1] + A_i2 * B[3] + A_i3 * B[5] )

		chi_2 += abs(Data["Sales"][i] - 3200. - A_i_sum)**2 / Data["Sales"][i]
	return chi_2

def adstock(B, x):

	A_i = 0
	A = []

	for i in range(0,len(x)):
		A_i = (x[i] + B[1] * A_i)
		
		A.append(B[0] + A_i * B[2])
	return A

def adstock2(B, x1, x2, x3):

	A_i1 = 0
	A_i2 = 0
	A_i3 = 0
	A = []

	for i in range(0,len(x1)):
		A_i1 = x1[i] + B[0] * A_i1
		A_i2 = x2[i] + B[2] * A_i2
		A_i3 = x3[i] + B[4] * A_i3
		
		A_i_sum = (A_i1 * B[1] + A_i2 * B[3] + A_i3 * B[5] )
		
		A.append(3200 + A_i_sum)
	return A

def norm_data(data):
	return data / sum(data)

from scipy.optimize import minimize
def plot_adstock(variable,data):

	fit = minimize(adstock_chi, [3200., 0.9, 0.9], args=(variable,),bounds=((0,5000),(0,1),(0,1)))#, method="BFGS")
	params = fit.x
	print("best fit params:", params)

	plt.plot(np.linspace(0,len(data)-1,len(data)),adstock((3200,0.5,0.001),variable),label="my guess")
	plt.plot(np.linspace(0,len(data)-1,len(data)),adstock((params[0],params[1],params[2]),variable),label="fit")
	plt.plot(np.linspace(0,len(data)-1,len(data)),data,label="data")
	plt.legend()
	print(adstock_chi((3200,0.5,0.001),variable),adstock_chi((params[0],params[1],params[2]),variable))
	plt.ylim(0,10000)
	plt.show()

def plot_multivariate_adstock(variables,data):

	fit = minimize(adstock_chi2, [0.5,0.001,0.5,0.001,0.5,0.005], args=(variables[0],variables[1],variables[2]),bounds=((0,1),(0,1),(0,1),(0,1),(0,1),(0,1)))#, method="BFGS")
	params = fit.x
	print("best fit params:", params)
	weeks = np.linspace(0,len(data)-1,len(data))
	#plt.plot(np.linspace(0,len(data)-1,len(data)),adstock2((3200,0.5,0.001),variable),label="my guess")
	plt.plot(weeks,adstock2((params[0],params[1],params[2],params[3],params[4],params[5]),variables[0],variables[1],variables[2]),label="fit")
	plt.plot(weeks,data,label="data")
	plt.plot(weeks,variables[0]/500.,label="TV")
	plt.plot(weeks,variables[1]/500.,label="Radio")
	plt.plot(weeks,variables[2]/500.,label="Dailies")
	plt.grid()
	plt.legend()
	print(adstock_chi2((params[0],params[1],params[2],params[3],params[4],params[5]),variables[0],variables[1],variables[2]))
	plt.ylim(0,10000)
	plt.show()

plot_multivariate_adstock((Data["TV"],Data["Radio"],Data["Dailies"]),Data["Sales"])
plot_adstock(Data["Media spend"],Data["Sales"])

def get_decay(media):
	a=1
