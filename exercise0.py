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