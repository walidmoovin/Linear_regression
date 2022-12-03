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
mileage_input = input("Enter the mileage of the car: ")
mileage = []
price = []
i = 0
with open('data.csv', 'r') as csvfile:
	plots = csv.reader(csvfile, delimiter=',')
	for row in plots:
		if i == 0:
			i = 1
			continue
		mileage.append(float(row[0]))
		price.append(float(row[1]))
		i += 1
# ----------------- EQUATIONS DEFINITION -----------------
def estimatePrice(givenMileage): # Using given hyphotesis
	return Theta0 + (Theta1 * givenMileage)
# mean squared error = average squared diff between observed vals and predicted vals.
# we use squared error to get positive values, we don't care about the value itself, it's more about magnitude.
# smal MSE = good monkey
# https://www.amherst.edu/system/files/media/1287/SLR_Leastsquares.pdf
def calculateMeanSquaredError(mileage, price):
	mean_squared_error = 0
	for x in range(len(mileage)):
		# mean_squared_error = (1 / len(mileage)) * mean_squared_error or :
		mean_squared_error += (price[x] - estimatePrice(mileage[x])) ** 2
	return mean_squared_error / float(len(mileage))
# ----------------- TRAINING -----------------
# https://www.youtube.com/watch?v=XdM6ER7zTLk Banger
# Using gradient descent to find best Theta0 and Theta1 vals:
# Gradient descent = optimization algorithm used to find val of parameters of a function that minimizes a cost function. In this case, mean squared error.
# it doesn't directly gives us the minimum, but a direction to go to.
# Big data = more iterations
# In the equation, m = number of training examples.

num_iterations = 1000
learning_rate = 0.0001 # how fast our model will converge
Theta0 = 0
Theta1 = 0
# Theta0 = learning_rate * (1 / m) * sum(estimatePrice(mileage[x]) - price[x]) for x in range(len(mileage))
# Theta1 = learning_rate * (1 / m) * sum(estimatePrice(mileage[x]) - price[x]) * mileage[x] for x in range(len(mileage))
for i in range(num_iterations):
	Theta0_derivative = 0
	Theta1_derivative = 0
	for x in range(len(mileage)):
		Theta0_derivative += -(2 / float(len(mileage))) * (price[x] - estimatePrice(mileage[x]))
		Theta1_derivative += -(2 / float(len(mileage))) * (price[x] - estimatePrice(mileage[x])) * mileage[x]
	Theta0 = Theta0 - (learning_rate * Theta0_derivative)
	Theta1 = Theta1 - (learning_rate * Theta1_derivative)
print("Theta0: ", Theta0)
print("Theta1: ", Theta1)
print("Mean squared error: ", calculateMeanSquaredError(mileage, price))
# ----------------- TESTING -----------------
print("Estimated price: ", estimatePrice(float(mileage_input)))