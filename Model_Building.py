#!/usr/bin/env python
# coding: utf-8

# # 1. Establish a Baseline
# A baseline model is a simple model used as a reference point to compare the performance of more complex models. It helps you understand if your complex models are truly adding value.
# 
# Example: Predicting House Prices
# 
# Imagine you have a dataset of house prices and you want to predict future prices. A simple baseline model might be to predict the average price for all houses.
# 
# python
# 

# In[2]:


import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# Sample data: house prices
data = {'price': [200000, 210000, 190000, 230000, 220000]}
df = pd.DataFrame(data)

# Calculate the mean (baseline prediction)
mean_price = df['price'].mean()

# Create baseline predictions (predicting the mean price for all instances)
df['baseline_prediction'] = mean_price

# Calculate Mean Squared Error (MSE) as an evaluation metric
mse_baseline = mean_squared_error(df['price'], df['baseline_prediction'])

print(f"Baseline MSE: {mse_baseline}")


# # 2. Iterate
# Iteration involves trying different models or model configurations to improve performance. This step includes training the model on your data, tweaking parameters, and trying different algorithms.
# 
# Example: Using Linear Regression
# 
# Let’s move from the baseline to a simple linear regression model.

# In[3]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Sample data: house sizes and prices
data = {'size': [1500, 1600, 1700, 1800, 1900],
        'price': [200000, 210000, 190000, 230000, 220000]}
df = pd.DataFrame(data)

# Split the data into training and testing sets
X = df[['size']]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate Mean Squared Error (MSE)
mse_model = mean_squared_error(y_test, y_pred)

print(f"Model MSE: {mse_model}")


# # 3. Evaluate
# Evaluation involves comparing the performance of your model against the baseline and using various metrics to assess its quality.
# 
# Comparison

# # Print both MSEs to compare
# print(f"Baseline MSE: {mse_baseline}")
# print(f"Model MSE: {mse_model}")

# In[6]:


improvement = mse_baseline - mse_model
print(f"Improvement in MSE: {improvement}")


# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Sample data: house sizes (in sq ft) and prices
data = {
    'size': [1500, 1600, 1700, 1800, 1900],
    'price': [200000, 210000, 190000, 230000, 220000]
}
df = pd.DataFrame(data)

# Calculate the mean price
mean_price = df['price'].mean()

# Create baseline predictions (predicting the mean price for all instances)
df['baseline_prediction'] = mean_price

# Calculate Mean Squared Error (MSE) for the baseline model
mse_baseline = mean_squared_error(df['price'], df['baseline_prediction'])
print(f"Baseline MSE: {mse_baseline}")

# Split the data into training and testing sets
X = df[['size']]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate Mean Squared Error (MSE) for the model
mse_model = mean_squared_error(y_test, y_pred)
print(f"Model MSE: {mse_model}")

# Compare Model MSE with Baseline MSE
improvement = mse_baseline - mse_model
print(f"Improvement in MSE: {improvement}")

# Plot the data and baseline prediction
plt.figure(figsize=(12, 6))

# Scatter plot of actual prices vs. sizes
plt.subplot(1, 2, 1)
plt.scatter(df['size'], df['price'], color='blue', label='Actual Prices')
plt.plot(df['size'], df['baseline_prediction'], color='red', linestyle='--', label='Baseline Prediction')
plt.xlabel('Size (sq ft)')
plt.ylabel('Price')
plt.title('Actual Prices and Baseline Prediction')
plt.legend()

# Scatter plot of actual vs. predicted prices from the model
plt.subplot(1, 2, 2)
plt.scatter(X_test, y_test, color='blue', label='Actual Prices')
plt.scatter(X_test, y_pred, color='green', label='Predicted Prices')
plt.xlabel('Size (sq ft)')
plt.ylabel('Price')
plt.title('Actual vs. Predicted Prices (Linear Regression)')
plt.legend()

plt.tight_layout()
plt.show()


# In[ ]:




