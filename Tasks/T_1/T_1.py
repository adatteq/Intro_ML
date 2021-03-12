# the train.csv file needs to be in the same directory as the main file

# importing modules
from sklearn import model_selection
from sklearn import linear_model
import numpy as np
import csv as csv


# importing the data from the csv file
file = np.genfromtxt('train.csv', delimiter=',')
y = file[1:, 0]      # Ignoring the header line, the first column is the y array
X = file[1:, 1:]     # Ignoring the header line, the rest is the X array


# define the cross-validation model
# TODO: understand the parameters and make sure they are right
cv = model_selection.KFold(n_splits=10, random_state=1, shuffle=True)


# lambdas array
lmb = np.array([0.1, 1, 10, 100, 200])


# result array
res = np.zeros(len(lmb))


# calculating the RSME
def rsme(x):
    return np.sqrt(np.mean(np.absolute(x)))


# calculating the hyper parameters for any lambda lm
def hyperpara(lm):
    # define the regression model
    model = linear_model.Ridge(alpha=lm)

    # TODO: understand the parameters and make sure they are right!
    # use k-fold CV to evaluate model
    scores = model_selection.cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)

    # print(rsme(scores))
    return rsme(scores)


# calculate the hyper parameters
for i in range(len(lmb)):
    res[i] = hyperpara(lmb[i])


# export the results in the csv file
# TODO: the output file name needs to be changed to the correct 'sample.csv'
np.savetxt('res.csv', res, delimiter=',')

# this code is obsolete
# with open('res.csv', 'w') as output:
#     writer = csv.writer(output)
#     writer.writerow(res)
# output.close()

