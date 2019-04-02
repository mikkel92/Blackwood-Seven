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
Data = clean_data(Data)

def adstock_chi(B, x):

	A_i = 0
	chi_2 = 0

	for i in range(0,len(x)):
		A_i = x[i] + B[1] * A_i

		
		chi_2 += abs(Data["Sales"][i] - B[0] - A_i * B[2])**2 / (Data["Sales"][i])
	return chi_2

def adstock_chi2(B, x1, x2, x3,test_data=False):#, x6):

	if test_data:
		A_i1 = 2652288.
		A_i2 = 7144113.
		A_i3 = 617531.

	else:
		A_i1 = 0
		A_i2 = 0
		A_i3 = 0	

	chi_2 = 0

	for i in x1.index.values:
		A_i1 = x1[i] + B[1] * A_i1
		A_i2 = x2[i] + B[3] * A_i2
		A_i3 = x3[i] + B[5] * A_i3
		
		
		A_i_sum = (A_i1 * B[2] + A_i2 * B[4] + A_i3 * B[6])

		chi_2 += abs(Data["Sales"][i] - B[0] - A_i_sum)**2 / (Data["Sales"][i])
	#print(A_i1,A_i2,A_i3,A_i4,A_i5)
	return chi_2

def adstock(B, x):

	A_i = 0
	A = []

	for i in range(0,len(x)):
		A_i = (x[i] + B[1] * A_i)
		
		A.append(B[0] + A_i * B[2])
	return A

def adstock2(B, x1, x2, x3, test_data=False):#, x6):

	if test_data:
		A_i1 = 2652288.
		A_i2 = 7144113.
		A_i3 = 617531.

	else:
		A_i1 = 0
		A_i2 = 0
		A_i3 = 0

	A = []

	for i in x1.index.values:
		A_i1 = x1[i] + B[1] * A_i1
		A_i2 = x2[i] + B[3] * A_i2
		A_i3 = x3[i] + B[5] * A_i3

		
		A_i_sum = (A_i1 * B[2] + A_i2 * B[4] + A_i3 * B[6])# + A_i6 * B[11])
		A.append(B[0] + A_i_sum)
	return A

def norm_data(data):
	return data / sum(data)

from scipy.optimize import minimize
from scipy.stats import chi2

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

def plot_multivariate_adstock(variables,test_size=10):
	

	# define train and test samples
	X_train = Data[variables][0:len(Data["TV"]) - test_size]
	y_train = Data[["Sales"]][0:len(Data["TV"]) - test_size]
	X_test = Data[variables][len(Data["TV"]) - test_size:]
	y_test = Data[["Sales"]][len(Data["TV"]) - test_size:]

	# values for fit
	#init_vals = [1,1,0,1,1,1,1,1,1,1, 2* min(Data["Sales"]),1]
	init_vals = [ min(Data["Sales"]),0.5,0.001,0.5,0.001,0.5,0.001]
	args = (Data[variables[0]],Data[variables[1]],Data[variables[2]])
	bounds = ((0,7000),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1))
	# fit the function
	fit = minimize(adstock_chi2, init_vals, args=args,bounds=bounds)#, method="BFGS")
	params = fit.x
	print("best fit params:", params)
	print("sales:", 1./params[2], 1./params[4], 1./params[6])
	# get chi2
	chi = (adstock_chi2(params,X_train[variables[0]],X_train[variables[1]],X_train[variables[2]]))
	print(len(X_train["TV"]))
	print("p-val = ", 1. - chi2.cdf(chi, len(X_train["TV"]) - 1))
	weeks = np.linspace(0,len(Data["TV"])-1,len(Data["TV"]))
	

	# plot fit
	fig, ax2 = plt.subplots(1, 1, figsize=(16, 12))
	if test_size > 0:
		ax2.plot(weeks[:-test_size],adstock2(params,X_train[variables[0]],X_train[variables[1]],X_train[variables[2]]),label="fit",c="k")
		print("chi2: ", adstock_chi2(params,X_train[variables[0]],X_train[variables[1]],X_train[variables[2]]))
		print("reduced chi2: ", adstock_chi2(params,X_train[variables[0]],X_train[variables[1]],X_train[variables[2]]) / (len(X_train["TV"]) - len(init_vals)))
		test_data = True
		ax2.plot(weeks[-test_size:],adstock2(params,X_test[variables[0]],X_test[variables[1]],X_test[variables[2]],test_data),label="prediction",c="k",linestyle=':')

	else:
		ax2.plot(weeks,adstock2(params,X_train[variables[0]],X_train[variables[1]],X_train[variables[2]]),label="fit",c="k")
		test_data = False
		print("chi2: ", adstock_chi2(params,X_train[variables[0]],X_train[variables[1]],X_train[variables[2]]))
		print("reduced chi2: ", adstock_chi2(params,X_train[variables[0]],X_train[variables[1]],X_train[variables[2]]) / (len(X_train["TV"]) - len(init_vals)))
		
	ax2.errorbar(weeks,Data["Sales"],yerr=np.sqrt(Data["Sales"]),fmt='+',label="data",c="r")
	ax2.plot(weeks,Data[variables[0]]/500.,label="TV",c="green")
	ax2.plot(weeks,Data[variables[1]]/500.,label="Radio",c="blue")
	ax2.plot(weeks,Data[variables[2]]/500.,label="Dailies",c="purple")
	ax2.grid()
	ax2.legend(fontsize=16) 
	ax2.set_ylabel("Sales", fontsize=18)
	ax2.set_xlabel("Week", fontsize=18)
	ax2.set_ylim(0,10000)

	print("Test data chi2: ", adstock_chi2(params,X_test[variables[0]],X_test[variables[1]],X_test[variables[2]],test_data))
	total_sales = sum(adstock2(params,X_train[variables[0]],X_train[variables[1]],X_train[variables[2]]))
	print(total_sales,params[0]*len(Data["Sales"]))
	print(params[0]*len(Data["Sales"]) / total_sales )
	plt.savefig("figures/Hyperpoly_fit_allvar_simpler.png")
	plt.close()
	

#Data = Data[Data["Campaign type 2"] > 0]
campaign2 = Data[Data["Campaign type 2"] > 0]
campaign3 = Data[Data["Campaign type 3"] > 0]


#plot_multivariate_adstock(["TV","Radio","Dailies"],test_size=0)


def plot_adstock(B1,B2,M1,M2):

	
	A1 = []
	A2 = []
	lim = range(0,20)

	inte = [0,0]
	for i in lim:	
		inte[0] += M1
		inte[1] += M2
		A1.append(M1)
		A2.append(M2)
		M1 = M1 * B1
		M2 = M2 * B2
		print(i,inte)

	weeks = np.linspace(0,lim[-1],lim[-1]+1)
	fig, ax2 = plt.subplots(1, 1, figsize=(16, 12))
	ax2.axvline(9,linestyle=':',c="k")
	ax2.plot(weeks,A1,label="TV: B = %1.2f" %B1,c="black")
	ax2.plot(weeks,A2,label="Radio: B = %1.2f" %B2,c="red")
	ax2.grid()
	ax2.legend(fontsize=16) 
	ax2.set_ylabel("Sales per 100000 spend", fontsize=18)
	ax2.set_xlabel("Week", fontsize=18)
	#ax2.set_ylim(0,110)
	plt.savefig("figures/Adstock_example.png")
	plt.close()

plot_adstock(0.94,0.53,1e6/4728.,1e6/1353.)