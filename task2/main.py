from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.svm import LinearSVC

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

import pandas as pd

# Import data from csv
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
SAMPLE_FILE = "sample.csv"

TRAIN = pd.read_csv(TRAIN_FILE)
x_data = TRAIN.loc[:, 'x1':'x16'].values
y_data = TRAIN.y.values

TEST = pd.read_csv(TEST_FILE)
test_id = TEST.Id.values
test_x_data = TEST.loc[:, 'x1':'x16'].values

############################################################

# Feature Pre-processing
pca = PCA()
x_data = pca.fit_transform(x_data)
test_x_data = pca.transform(test_x_data)

############################################################

k = 10
kf = KFold(n_splits=k)

names = [
    "LinearSVC",
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
    "OneVsOne",
    "OneVsRest"
]

models = [
    LinearSVC(random_state=0),
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    OneVsOneClassifier(LinearSVC(random_state=0)),
    OneVsRestClassifier(LinearSVC(random_state=0))
]

model_accuracy = {}

i = 0
for model in models:
    cross_validation_acc = []
    for train_index, test_index in kf.split(x_data):
        X_train, X_test = x_data[train_index], x_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        acc = cross_validation_acc.append(float(accuracy_score(y_test, prediction)))  # use accuracy score

    average = sum(cross_validation_acc)/len(cross_validation_acc)
    model_accuracy[average] = i
    print(names[i])
    print(average)
    i += 1

print(model_accuracy)
max_accuracy = max(model_accuracy, key=float)
print(max_accuracy)
max_accuracy_model = models[model_accuracy[max_accuracy]]

print("RESULT - max accuracy")
print(max_accuracy, names[model_accuracy[max_accuracy]])

max_accuracy_model.fit(x_data, y_data)
test_predict = max_accuracy_model.predict(test_x_data)
result = pd.DataFrame({'Id': test_id, 'y': test_predict})
result.to_csv(SAMPLE_FILE, index=False)
