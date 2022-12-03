import csv
import matplotlib.pyplot as plt
import numpy as np
# Regression models describe relationship between a dependent variable and independent variable(s).
# We will use simple linear regression to find the price of a car based on its mileage.

# Theta's represemts how important feature is in hyphotesis (regressionc coefficients or weights)
# The more important feature is, the bigger Theta is.

# Formula for simple linear regression :
# y = b + ax + e (y -> dependentVar | x -> independentVar | a -> regression coefficient = how much we expect y to change as x increases | b -> y-intercept | e -> error of the estimate.
# ----------------- PARSING -----------------
def parsing_dataset():
	mileage = []
	price = []
	i = 0
	with open('data.csv', 'r') as csvfile:
		plots = csv.reader(csvfile, delimiter=',')
		for row in plots:
			# if i == 0:
			# 	i = 1
			# 	continue
			mileage.append(float(row[0]))
			price.append(float(row[1]))
			i += 1
	return (mileage, price)
# ----------------- NORMALIZATION -----------------
# Normalization = scaling data to be between 0 and 1.
# how to normalize:
# x = (x - min(x)) / (max(x) - min(x)) -> (x - min(x)) = how far from min(x) is x | (max(x) - min(x)) = range of x
# https://www.youtube.com/watch?v=UqYde-LULfs
# use copy of data to not modify original data
def normalizeData(givenMileage, givenPrice):
	normalizedMileage = givenMileage.copy()
	normalizedPrice = givenPrice.copy()
	minMileage = min(givenMileage)
	maxMileage = max(givenMileage)
	minPrice = min(givenPrice)
	maxPrice = max(givenPrice)
	for x in range(len(givenMileage)):
		normalizedMileage[x] = (givenMileage[x] - minMileage) / (maxMileage - minMileage)
		normalizedPrice[x] = (givenPrice[x] - minPrice) / (maxPrice - minPrice)
	return (normalizedMileage, normalizedPrice)
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
	num_iterations = 100000
	learning_rate = 0.001 # how fast our model will converge
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
def run():
	mileage, price = parsing_dataset()
	mileage, price = normalizeData(mileage, price)
	print("MSE at start: ", calculateMeanSquaredError(mileage, price, 0, 0))
	Theta0, Theta1 = gradientDescent(mileage, price)
	print("Theta0: ", Theta0, "Theta1: ", Theta1)
	print("MSE: ", calculateMeanSquaredError(mileage, price, Theta0, Theta1))
	print("Price for 60000 mileage: ", estimatePrice(60000, Theta0, Theta1))
	print("Price for 70000 mileage: ", estimatePrice(70000, Theta0, Theta1))
	print("Price for 80000 mileage: ", estimatePrice(80000, Theta0, Theta1))
	print("Price for 90000 mileage: ", estimatePrice(90000, Theta0, Theta1))
	print("Price for 100000 mileage: ", estimatePrice(100000, Theta0, Theta1))
	print("Price for 200000 mileage: ", estimatePrice(200000, Theta0, Theta1))
	plotData(mileage, price, Theta0, Theta1)
run()