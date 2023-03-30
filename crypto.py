import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import os
import time as t

def plot_graph(file_name):
    data = pd.read_csv(file_name)

    target = data["Close"]
    features = data[["Open", "High", "Low", "Volume"]]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

    model = LinearRegression()

    # Train the model on the training data
    model.fit(X_train, y_train)

    # Make predictions on the testing data
    predictions = model.predict(X_test)

    # Plot the data to visualize the trends
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    
    axs[0, 0].scatter(data["Open"], target)
    axs[0, 0].set_xlabel("Opening Price")
    axs[0, 0].set_ylabel("Price")
    axs[0, 0].set_title("Opening Price vs Price")

    axs[0, 1].scatter(data["High"], target)
    axs[0, 1].set_xlabel("Highest Price")
    axs[0, 1].set_ylabel("Price")
    axs[0, 1].set_title("Highest Price vs Price")

    axs[1, 0].scatter(data["Low"], target)
    axs[1, 0].set_xlabel("Lowest Price")
    axs[1, 0].set_ylabel("Price")
    axs[1, 0].set_title("Lowest Price vs Price")

    axs[1, 1].scatter(data["Volume"], target)
    axs[1, 1].set_xlabel("Trading Volume")
    axs[1, 1].set_ylabel("Price")
    axs[1, 1].set_title("Trading Volume vs Price")

    # difference in market volume vs predicted market volume
    plt.figure(figsize=(8, 6))
    plt.scatter(X_test["Volume"], predictions - y_test, color='orange', marker='x')
    plt.xlabel("Trading Volume (Predicted)")
    plt.ylabel("Price (Predicted) - Price (Actual)")
    plt.title("Trading Volume (Predicted) vs Price (Predicted) - Price (Actual)")
    plt.show()


def main():
	dir_csv=os.listdir()
	sr_no=1
	a=0
	for files in dir_csv:
		if '.csv' in files:
			print(f'{sr_no}) {files}')
			sr_no+=1
	ask_usr=int(input(">"))
	if (ask_usr=='') or (ask_usr==None) or (ask_usr>=sr_no) or (ask_usr==0):
		print("invalid selection")
	else:
		#to make it look good aise design kr diya h
		print('You selected dataset:',dir_csv[ask_usr+1])
		print('Plotting data',end='')
		while a!=5:
			print('.'*a,end='')
			t.sleep(0.2)
			a+=1
		
		plot_graph(dir_csv[ask_usr+1])
main()
