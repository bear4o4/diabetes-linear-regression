from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


#task 1
diabetes_dataset = load_diabetes(as_frame=True)

# inspect the structure
data = diabetes_dataset['data']
target = diabetes_dataset['target']
feature_names = diabetes_dataset['feature_names']

print("Features:", feature_names)
print("Target Description:\n", target.describe())

print("##############################################")


#task 2
# extract the feature matrix
X_diabetes_reg = data

# extract the disease progression score
y_diabetes_reg = target

print("Feature Matrix (X_diabetes_reg):")
print(X_diabetes_reg.head())
print("\nTarget Variable (y_diabetes_reg):")
print(y_diabetes_reg.head())

print("##############################################")

#task 3
#the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_diabetes_reg, y_diabetes_reg, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


print("##############################################")

#task 4
# train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

coefficients = model.coef_
intercept = model.intercept_

print("Coefficients:", coefficients)
print("Intercept:", intercept)

print("##############################################")

#task 5
# make predictions on the test set
y_pred = model.predict(X_test)

plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted Values')
plt.show()

print("##############################################")

#task 6
# calculate mean squared rror
mse = mean_squared_error(y_test, y_pred)

print("Mean Squared Error:", mse)