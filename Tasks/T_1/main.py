# the train.csv file needs to be in the same directory as the main file

# importing modules
from sklearn import model_selection
from sklearn import linear_model
import numpy as np


# importing the data from the csv file
file = np.genfromtxt('train.csv', delimiter=',')

# file[x:y, s:t] -> in rows x to y take elements s to t
y = file[1:, 0]      # Ignoring the header line, the first column is the y array
X = file[1:, 1:]     # Ignoring the header line, the rest is the X array


# define the cross-validation model
# TODO: understand the parameters and make sure they are right
cv = model_selection.KFold(n_splits=10)


# lambdas array
lmb = np.array([0.1, 1, 10, 100, 200])


# result array
res = np.zeros(len(lmb))


# calculating the RSME for every hyper parameter lm
def rsme(lm):
    # define the regression model
    model = linear_model.Ridge(alpha=lm)

    # TODO: understand the parameters and make sure they are right!
    # use k-fold CV to evaluate model
    scores = model_selection.cross_val_score(model, X, y, cv=cv)

    # print(scores)
    # TODO: are we calculating the RSME correctly?
    return np.sqrt(np.mean(np.absolute(scores)))


# calculate the hyper parameters
for i in range(len(lmb)):
    res[i] = rsme(lmb[i])


# export the results in the csv file
# TODO: rename the output file
np.savetxt('res.csv', res, delimiter=',')