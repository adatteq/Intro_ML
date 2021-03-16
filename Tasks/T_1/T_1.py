# the train.csv file needs to be in the same directory as the main file

# importing modules
from sklearn import model_selection
from sklearn import linear_model
from sklearn import metrics
import numpy as np


# importing the data from the csv file
file = np.genfromtxt('train.csv', delimiter=',')

# file[x:y, s:t] -> in rows x to y take elements s to t
y = file[1:, 0]      # Ignoring the header line, the first column is the y array
X = file[1:, 1:]     # Ignoring the header line, the rest is the X array


# lambdas array
lmb = np.array([0.1, 1, 10, 100, 200])


# result array
res = np.zeros(len(lmb))


# calculating the RSME for every hyper parameter lm
def rsme(lm):
    # define the regression model
    model = linear_model.Ridge(alpha=lm)

    # TODO: should we take cross_val_predict or cross_val_score?
    # use 10-fold CV to evaluate model
    predictions = model_selection.cross_val_predict(model, X, y, cv=10)

    # print(predictions)
    # print(len(predictions))
    # TODO: are we calculating the RSME correctly?
    return np.sqrt(metrics.mean_squared_error(y, predictions))


# calculate the RSME
for i in range(len(lmb)):
    res[i] = rsme(lmb[i])


# export the results in the csv file
# TODO: rename the output file
np.savetxt('res.csv', res, delimiter=',')
