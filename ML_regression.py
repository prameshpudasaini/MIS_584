import pandas as pd

# wine quality data set
data = pd.read_csv(r"D:\GitHub\MIS_584\winequality-red.csv", sep = ';')

X = data.drop(columns = 'quality')
y = data['quality']


# =============================================================================
# Linear Regression 
# =============================================================================

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

method = 'Linear Regression'
linear_reg = LinearRegression()
linear_reg.fit(X, y)

linear_reg_pred = linear_reg.predict(X)
print("Predicted ratings for {}:".format(method))
print(linear_reg_pred)
print()

# rmse
linear_reg_rmse = mean_squared_error(y, linear_reg_pred, squared = False)
print("RMSE for {}: {:.4f}".format(method, linear_reg_rmse))

# mae
linear_reg_mae = mean_absolute_error(y, linear_reg_pred)
print("MAE for {}: {:.4f}".format(method, linear_reg_mae))
print()

# extract estimated coefficients of linear regression
linear_reg_coeff = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': linear_reg.coef_
    })

print(linear_reg_coeff)


# =============================================================================
# K-Nearest Neighbor Regression
# =============================================================================

from sklearn.neighbors import KNeighborsRegressor

method = 'K-Nearest Neighbor Regressor'
knn_reg = KNeighborsRegressor(n_neighbors = 3)
knn_reg.fit(X, y)

knn_reg_pred = knn_reg.predict(X)
print("Predicted ratings for {}:".format(method))
print(knn_reg_pred)
print()

# rmse
knn_reg_rmse = mean_squared_error(y, knn_reg_pred, squared = False)
print("RMSE for {}: {:.4f}".format(method, knn_reg_rmse))

# mae
knn_reg_mae = mean_absolute_error(y, knn_reg_pred)
print("MAE for {}: {:.4f}".format(method, knn_reg_mae))
print()


# =============================================================================
# Decision Tree Regression
# =============================================================================

from sklearn.tree import DecisionTreeRegressor

method = 'Decision Tree Regressor'
dt_reg = DecisionTreeRegressor(max_depth = 3)
dt_reg.fit(X, y)

dt_reg_pred = dt_reg.predict(X)
print("Predicted ratings for {}:".format(method))
print(dt_reg_pred)
print()

# rmse
dt_reg_rmse = mean_squared_error(y, dt_reg_pred, squared = False)
print("RMSE for {}: {:.4f}".format(method, dt_reg_rmse))

# mae
dt_reg_mae = mean_absolute_error(y, dt_reg_pred)
print("MAE for {}: {:.4f}".format(method, dt_reg_mae))

# plot tree
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize = (24, 24))
plot_tree(dt_reg, feature_names = X.columns)