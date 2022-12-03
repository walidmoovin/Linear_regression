import csv
import matplotlib.pyplot as plt
import numpy as np
# Regression models describe relationship between a dependent variable and independent variable(s).
# We will use simple linear regression to find the price of a car based on its mileage.

# Theta's represents how important feature is in hyphotesis (regressionc coefficients or weights)
# The more important feature is, the bigger Theta is.

# Formula for simple linear regression :
# y = b + ax + e (y -> dependentVar | x -> independentVar | a -> regression coefficient = how much we expect y to change as x increases | b -> y-intercept | e -> error of the estimate.
# ----------------- PARSING -----------------
def parsing_dataset():
	mileage = []
	price = []
	# open csv file and stop program if file is not found
	try:
		with open('data.csv') as csvfile:
			reader = csv.reader(csvfile)
			for row in reader:
					mileage.append(float(row[0]))
					price.append(float(row[1]))
	except FileNotFoundError:
		print ("File not found")
		exit()
	return (mileage, price)
# ----------------- INPUT -----------------
def inputMileage(Theta0, Theta1, givenMileage, givenPrice):
	print ("Welcome to the car price estimator !")
	print ("Please enter the mileage of your car: ")
	mileage = input()
	# verify if input is a number and stop program if not and if input is negative
	try:
		mileage = float(mileage)
		if (mileage < 0):
			print ("Mileage must be positive")
			exit()
		if (estimatePrice(mileage, Theta0, Theta1) < 0):
			print("cringe, your car is broken at this point")
			exit()
		print("Your car is worth: ", estimatePrice(mileage, Theta0, Theta1))
		plotData(givenMileage, givenPrice, Theta0, Theta1)
	except ValueError:
		print ("Mileage must be a number")
		exit()
	return (mileage)
# ----------------- NORMALIZATION -----------------
# Normalization = scaling data to be between 0 and 1.
# how to normalize:
# x = (x - min(x)) / (max(x) - min(x)) (x - min(x)) = how far from min(x) is x | (max(x) - min(x)) = range of x
# https://www.youtube.com/watch?v=UqYde-LULfs
def normalizeData(givenMileage, givenPrice):
	min_mileage = min(givenMileage)
	max_mileage = max(givenMileage)
	min_price = min(givenPrice)
	max_price = max(givenPrice)
	for x in range(len(givenMileage)):
		givenMileage[x] = (givenMileage[x] - min_mileage) / (max_mileage - min_mileage)
		givenPrice[x] = (givenPrice[x] - min_price) / (max_price - min_price)
	return (givenMileage, givenPrice)
# denormalize data that was normalized
def denormalizeData(givenMileage, givenPrice, Theta0, Theta1):
	min_mileage = min(givenMileage)
	max_mileage = max(givenMileage)
	min_price = min(givenPrice)
	max_price = max(givenPrice)
	Theta0 = (Theta0 * (max_price - min_price)) + min_price
	Theta1 = Theta1 * (max_price - min_price) / (max_mileage - min_mileage)
	return (Theta0, Theta1)

# ----------------- EQUATIONS DEFINITION -----------------
def estimatePrice(givenMileage, Theta0, Theta1): # Using given hyphotesis
	return (Theta0 + (Theta1 * float(givenMileage)))
# mean squared error = average squared diff between observed vals and predicted vals.
# we use squared error to get positive values, we don't care about the value itself, it's more about magnitude.
# smal MSE = good monkey
# https://www.amherst.edu/system/files/media/1287/SLR_Leastsquares.pdf
def calculateMeanSquaredError(givenMileage, givenPrice, Theta0, Theta1):
	mean_squared_error = 0
	for x in range(len(givenMileage)):
		# mean_squared_error = (1 / len(givenMileage)) * mean_squared_error or :
		mean_squared_error += (givenPrice[x] - (estimatePrice(givenMileage[x], Theta0, Theta1))) ** 2
	return (mean_squared_error / float(len(givenMileage)))
# ----------------- TRAINING -----------------
# https://www.youtube.com/watch?v=XdM6ER7zTLk Banger
# Using gradient descent to find best Theta0 and Theta1 vals:
# Gradient descent = optimization algorithm used to find val of parameters of a function that minimizes a cost function. In this case, mean squared error.
# it doesn't directly gives us the minimum, but a direction to go to.
# Big data = more iterations
# In the equation, m = number of training examples.
def gradientDescent(givenMileage, givenPrice):
	num_iterations = 10066
	learning_rate = 0.1 # how fast our model will converge
	Theta0 = 0
	Theta1 = 0
	for i in range(num_iterations):
		Theta0_derivative = 0
		Theta1_derivative = 0
		for x in range(len(givenMileage)):
			Theta0_derivative += -(2 / float(len(givenMileage))) * (givenPrice[x] - estimatePrice(givenMileage[x], Theta0, Theta1))
			Theta1_derivative += -(2 / float(len(givenMileage))) * (givenPrice[x] - estimatePrice(givenMileage[x], Theta0, Theta1)) * givenMileage[x]
			# print ("Theta0_derivative: ", Theta0_derivative, "Theta1_derivative: ", Theta1_derivative)
		Theta0 = Theta0 - (learning_rate * Theta0_derivative)
		Theta1 = Theta1 - (learning_rate * Theta1_derivative)
	return (Theta0, Theta1)
# ----------------- PLOTTING -----------------
def plotData(givenMileage, givenPrice, Theta0, Theta1):
	# print data points
	for x in range(len(givenMileage)):
		plt.scatter(givenMileage[x], givenPrice[x])
	# print regression line
	x = np.array(givenMileage)
	y = Theta0 + Theta1 * x
	plt.plot(x, y)
	plt.xlabel('Mileage')
	plt.ylabel('Price')
	plt.show()
# ----------------- RUNNING -----------------
def main():
	mileage, price = parsing_dataset()
	mileageCopy = mileage.copy()
	priceCopy = price.copy()
	print("Mean squared error before training: ", calculateMeanSquaredError(mileage, price, 0, 0))
	mileageCopy, priceCopy = normalizeData(mileageCopy, priceCopy)
	Theta0, Theta1 = gradientDescent(mileageCopy, priceCopy)
	print ("Mean squared error: ", calculateMeanSquaredError(mileageCopy, priceCopy, Theta0, Theta1))
	Theta0, Theta1 = denormalizeData(mileage, price, Theta0, Theta1)
	print("Theta0: ", Theta0, "Theta1: ", Theta1)
	inputMileage(Theta0, Theta1, mileage, price)

if __name__ == "__main__":
	main()