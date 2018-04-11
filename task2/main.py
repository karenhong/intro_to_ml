'''
Description
One v One = 79%
One v All = 76.5%
0.86 QDA - 0.839274141283 online 
'''

from sklearn import linear_model, metrics 
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import numpy as np
import pandas as pd

TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
SAMPLE_FILE = "sample.csv"

# Import data from csv
TRAIN = pd.read_csv(TRAIN_FILE)
x_data = TRAIN.loc[:, 'x1':'x16'].values
y_data = TRAIN.y.values

TEST = pd.read_csv(TEST_FILE)
test_id = TEST.Id.values
test_x_data = TEST.loc[:, 'x1':'x16'].values

############################################################



k = 10
kf = KFold(n_splits=k)

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA", "OneVsOne", "OneVsRest"]
models = [LinearSVC(random_state=0)]
# models =  [KNeighborsClassifier(3),
#     SVC(kernel="linear", C=0.025),
#     SVC(gamma=2, C=1),
#     GaussianProcessClassifier(1.0 * RBF(1.0)),
#     DecisionTreeClassifier(max_depth=5),
#     RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
#     MLPClassifier(alpha=1),
#     AdaBoostClassifier(),
#     GaussianNB(),
#     QuadraticDiscriminantAnalysis(),
#     OneVsOneClassifier(LinearSVC(random_state=0)), 
#     OneVsRestClassifier(LinearSVC(random_state=0))]

model_accuracy = {} #accuracy:(name,model)

i = 0
for model in models:
    print (names[i])
    cross_validation_acc = {} # accuracy:model
    for train_index, test_index in kf.split(x_data):

        X_train, X_test = x_data[train_index], x_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        acc = float(accuracy_score(y_test, prediction)) # use accuraccy score
        print (acc)
        cross_validation_acc[acc] = model

    max_accuracy_cross_validation = max(cross_validation_acc, key = float)
    model_accuracy[max_accuracy_cross_validation] = (names[i], cross_validation_acc[max_accuracy_cross_validation])
    i+=1


#want max accruacy 
max_accuracy = max(model_accuracy, key = float)
max_accuracy_model = model_accuracy[max_accuracy]

print ("RESULT - max accuracy")
print (max_accuracy, max_accuracy_model[0])



# WRITE RESULTS TO FILE
# predictions = 
# dataframe = pd.read_csv(TRAIN_FILE)
# x_data = dataframe.loc[:, 'x1':'x16'].values
# y_data = dataframe.y.values

test_predict = max_accuracy_model[1].predict(test_x_data)
result = pd.DataFrame({'Id':test_id,'y':test_predict})
# # result['y'] = result['y'].astype(np.float64)
# # print (result.dtypes)
# result.to_csv(SAMPLE_FILE, index=False)
