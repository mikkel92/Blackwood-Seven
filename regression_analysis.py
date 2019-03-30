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
