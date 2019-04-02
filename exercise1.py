import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from exercise0 import clean_data
from scipy import stats

# load data
Data = pd.read_excel("Exercise.xlsx", sheetname="Data")
AdStock = pd.read_excel("Exercise.xlsx", sheetname="AdStock")

# 0. Inspect the data - make sure you understand all variables and the objective of this exercise. Is the data "clean"/ meaningfull?									

names = list(Data)
print(names)
# Make functions to easily controll what is executed

Data = clean_data(Data)

def linear_model(x, a, b):
	return a * x + b

from scipy.optimize import curve_fit

# plot the best fit line for each media category
def linear_regression(Data,variable):

	ylabel = variable
	zero_point = np.average(Data["Sales"].nsmallest(5))
	#Data = Data[Data[variable] > 0]

	x = Data["Sales"] - zero_point
	variable = Data[variable]

	slope, intercept, r_value, p_value, std_err = stats.linregress(x, variable)
	print(slope, intercept, r_value, p_value, std_err)
	popt, pcov = curve_fit(linear_model, x, variable,bounds=([-np.inf,0], np.inf))
	perr = np.sqrt(np.diag(pcov))
	print(popt,perr)

	c = "purple"

	fig, ax2 = plt.subplots(1, 1, figsize=(16, 12))
	ax2.scatter(x + zero_point,variable)
	ax2.plot(x + zero_point, linear_model(x, *popt), c, label='fit: $ per sale = %5.1f * sales + %5.0f' % tuple(popt), linestyle="--")
	ax2.plot(x + zero_point, linear_model(x, *popt + perr), c, linewidth=1, linestyle=":",label=r"1$\sigma$ fit err")
	ax2.legend(fontsize=16)
	ax2.plot(x + zero_point, linear_model(x, *popt - perr), c, linewidth=1, linestyle=":")
	ax2.grid()
	ax2.set_xlabel("Sales", fontsize=18)
	ax2.set_ylabel("$ on " + ylabel, fontsize=18)

	save_str = "figures/linreg_" + ylabel + ".png"
	print(save_str)
	plt.savefig(save_str)
	plt.close("all")

"""
linear_regression(Data,"Media spend")
linear_regression(Data,"TV")
linear_regression(Data,"Radio")
linear_regression(Data,"Dailies")
"""

# didn't work as planned
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize  
from sklearn.model_selection import train_test_split 
from sklearn import metrics  

variables = ["TV","Radio","Dailies"]

def Hyperplane_fit(variables,Data):

	#Data = Data[Data["Media spend"] > 0]
	zero_point = np.average(Data["Sales"].nsmallest(5))

	for i in range(0,1):
		x = Data[variables]
		y = Data[["Sales"]] - zero_point

		variables = ["Media spend","TV","Radio","Dailies","GRP","Competitor 1 Spend","Competitor 2 Spend"]
		#variables = ["TV","Radio","Dailies"]
		X_train = Data[variables][0:len(Data["TV"]) - 10]
		y_train = Data[["Sales"]][0:len(Data["TV"]) - 10]
		X_test = Data[variables][len(Data["TV"]) - 10:]
		y_test = Data[["Sales"]][len(Data["TV"]) - 10:]


		regressor = LinearRegression()  
		regressor.fit(y_train, X_train)  
		b = regressor.intercept_ 
		a = regressor.coef_ 
		print(b, a)

		#plt.style.use('ggplot')
		colors = ["orange","green","blue","purple","black","black","black"]
		for variable in range(0,len(b)):
			if i == 0:
				plt.plot(y + zero_point, linear_model(y, a[variable][0], b[variable] ),c=colors[variable], linewidth=1.5, linestyle=":",alpha=1,label='%s   fit: $ per sale = %5.1f * sales + %5.0f' % (variables[variable],a[variable][0],b[variable]))
			else:
				plt.plot(y + zero_point, linear_model(y, a[variable][0], b[variable] ),c=colors[variable], linewidth=1.5, linestyle=":",alpha=0.1)

		plt.legend()
	plt.xlabel("Sales")
	plt.ylabel("$ spend")


	X_pred = regressor.predict(y_test)  
	print(a[1])

	chi2 = 0
	predicted_sales = []
	for test_data in range(72,82):

		sum_d = 0
		for i, variable in enumerate(variables):
			sum_d += ((X_test[variable][test_data]- b[i]) / a[i][0])
		predicted_sales.append(sum_d)

		chi2 += (Data["Sales"][i] - predicted_sales[-1]) ** 2 / Data["Sales"][i]
	print(chi2)
	print(predicted_sales)
	print(X_pred)
	print(X_test)  
	#print(df)  

	#chi2 = (y_pred - y_test)**2 / y_pred
	#print(sum(chi2["Sales"]))  

	#print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(X_test, X_pred))) 

	plt.show()
	plt.close("all")

#linear_regression(Data,"Dailies")
#linear_regression(Data,"Radio")
#linear_regression(Data,"Dailies")
#Hyperplane_fit(variables,Data)

def line_chi2(B, x1, x2, x3):

	chi_2 = 0
	
	for i in x1.index.values:
		A = x1[i] * B[0] + x2[i] * B[1] + x3[i] * B[2] + B[3]
		

		chi_2 += (Data["Sales"][i] - A)**2 / (Data["Sales"][i])
	return chi_2

def line(B, x1, x2, x3):

	A = []
	
	for i in x1.index.values:

		A_i = x1[i] * B[0] + x2[i] * B[1] + x3[i] * B[2] + B[3]
		A.append(A_i)

	return A

from scipy.optimize import minimize
from scipy.stats import chi2

def plot_planefit(variables):

	# define train and test samples
	X_train = Data[variables][0:len(Data["TV"]) - 10]
	y_train = Data[["Sales"]][0:len(Data["TV"]) - 10]
	X_test = Data[variables][len(Data["TV"]) - 10:]
	y_test = Data[["Sales"]][len(Data["TV"]) - 10:]

	# values for fit
	init_vals = [0.001,0.001,0.001,1200]
	args = (Data[variables[0]],Data[variables[1]],Data[variables[2]])
	bounds = ((-1,1),(-1,1),(-1,1),(0,7000))
	# fit the function
	fit = minimize(line_chi2, init_vals, args=args,bounds=bounds)#, method="BFGS")
	params = fit.x
	print("best fit params:", params)

	# get chi2
	chi = (line_chi2(params,X_train[variables[0]],X_train[variables[1]],X_train[variables[2]]))
	print(len(X_train["TV"]))
	print("p-val = ", 1. - chi2.cdf(chi, len(X_train["TV"]) - 1))
	weeks = np.linspace(0,len(Data["TV"])-1,len(Data["TV"]))
	

	# plot fit
	fig, ax2 = plt.subplots(1, 1, figsize=(16, 12))
	ax2.plot(weeks[:-10],line(params,X_train[variables[0]],X_train[variables[1]],X_train[variables[2]]),label="fit",c="k")
	ax2.plot(weeks[-10:],line(params,X_test[variables[0]],X_test[variables[1]],X_test[variables[2]]),label="prediction",c="k",linestyle=':')

	ax2.errorbar(weeks,Data["Sales"],yerr=np.sqrt(Data["Sales"]),fmt='+',label="data",c="r")
	ax2.plot(weeks,Data[variables[0]]/500.,label="TV",c="green")
	ax2.plot(weeks,Data[variables[1]]/500.,label="Radio",c="blue")
	ax2.plot(weeks,Data[variables[2]]/500.,label="Dailies",c="purple")
	ax2.grid()
	ax2.legend(fontsize=16) 
	ax2.set_ylabel("Sales", fontsize=18)
	ax2.set_xlabel("Week", fontsize=18)
	print("chi2: ", line_chi2(params,X_train[variables[0]],X_train[variables[1]],X_train[variables[2]]))
	print("reduced chi2: ", line_chi2(params,X_train[variables[0]],X_train[variables[1]],X_train[variables[2]]) / (len(X_train["TV"]) - len(init_vals)))
	ax2.set_ylim(0,10000)

	print("Test data chi2: ", line_chi2(params,X_test[variables[0]],X_test[variables[1]],X_test[variables[2]]))

	plt.savefig("figures/Hyperplane_fit_3var.png")
	plt.close()

#plot_planefit(["TV","Radio","Dailies"])


def line_chi22(B, x1, x2, x3, x4, x5):

	chi_2 = 0
	
	for i in x1.index.values:
		A = x1[i] * B[0] + x2[i] * B[1] + x3[i] * B[2] + x4[i] * B[3] + x5[i] * B[4] + B[5]
		

		chi_2 += (Data["Sales"][i] - A)**2 / (Data["Sales"][i])
	return chi_2

def line2(B, x1, x2, x3, x4, x5):

	A = []
	
	for i in x1.index.values:
		A_i = x1[i] * B[0] + x2[i] * B[1] + x3[i] * B[2] + x4[i] * B[3] + x5[i] * B[4] + B[5]
		
		A.append(A_i)

	return A


def plot_planefit_all(variables):

	# define train and test samples
	X_train = Data[variables][0:len(Data["TV"]) - 10]
	y_train = Data[["Sales"]][0:len(Data["TV"]) - 10]
	X_test = Data[variables][len(Data["TV"]) - 10:]
	y_test = Data[["Sales"]][len(Data["TV"]) - 10:]

	# values for fit
	init_vals = [0.001,0.001,0.001,0.001,0.001,7000]
	args = (Data[variables[0]],Data[variables[1]],Data[variables[2]],Data[variables[3]],Data[variables[4]])
	bounds = ((-1,1),(-1,1),(-1,1),(-1,1),(-1,1),(0,7000))
	# fit the function
	fit = minimize(line_chi22, init_vals, args=args,bounds=bounds)#, method="BFGS")
	params = fit.x
	print("best fit params:", params)
	print(1. / params)
	# get chi2
	chi = (line_chi22(params,X_train[variables[0]],X_train[variables[1]],X_train[variables[2]],X_train[variables[3]],X_train[variables[4]]))
	print(len(X_train["TV"]))
	print("p-val = ", 1. - chi2.cdf(chi, len(X_train["TV"]) - 1))
	weeks = np.linspace(0,len(Data["TV"])-1,len(Data["TV"]))
	

	# plot fit
	fig, ax2 = plt.subplots(1, 1, figsize=(16, 12))
	ax2.plot(weeks[:-10],line2(params,X_train[variables[0]],X_train[variables[1]],X_train[variables[2]],X_train[variables[3]],X_train[variables[4]]),label="fit",c="k")
	ax2.plot(weeks[-10:],line2(params,X_test[variables[0]],X_test[variables[1]],X_test[variables[2]],X_test[variables[3]],X_test[variables[4]]),label="prediction",c="k",linestyle=':')

	ax2.errorbar(weeks,Data["Sales"],yerr=np.sqrt(Data["Sales"]),fmt='+',label="data",c="r")
	ax2.plot(weeks,Data[variables[0]]/500.,label="TV",c="green")
	ax2.plot(weeks,Data[variables[1]]/500.,label="Radio",c="blue")
	ax2.plot(weeks,Data[variables[2]]/500.,label="Dailies",c="purple")
	ax2.grid()
	ax2.legend(fontsize=16) 
	ax2.set_ylabel("Sales", fontsize=18)
	ax2.set_xlabel("Week", fontsize=18)
	print("chi2: ", line_chi22(params,X_train[variables[0]],X_train[variables[1]],X_train[variables[2]],X_train[variables[3]],X_train[variables[4]]))
	print("reduced chi2: ", line_chi22(params,X_train[variables[0]],X_train[variables[1]],X_train[variables[2]],X_train[variables[3]],X_train[variables[4]]) / (len(X_train["TV"]) - len(init_vals)))
	ax2.set_ylim(0,10000)

	print("Test data chi2: ", line_chi22(params,X_test[variables[0]],X_test[variables[1]],X_test[variables[2]],X_test[variables[3]],X_test[variables[4]]))

	plt.savefig("figures/Hyperplane_fit_allvar.png")
	plt.close()

plot_planefit_all(["TV","Radio","Dailies","Competitor 1 Spend","Competitor 2 Spend"])