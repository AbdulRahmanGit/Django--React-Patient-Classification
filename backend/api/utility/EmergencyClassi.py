import pandas as pd  # for data manipulation
from django.conf import settings
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pickle
path = os.path.join(settings.MEDIA_ROOT, "EmergencyDataset.csv")
df = pd.read_csv(path)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=109)
param_grid = [
    {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
]


def process_randomForest():
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    # # now you can save it to a file
    model_file = os.path.join(settings.MEDIA_ROOT, 'alexmodel.pkl')
    with open(model_file, 'wb') as f:
        pickle.dump(clf, f)
    rf_report = metrics.classification_report(y_test, y_pred, output_dict=True)
    print("Classification report for - \n{}:\n{}\n".format(clf, rf_report))
    return rf_report
    

def process_decesionTree():
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.score(X_test, y_test))
    dt_report = metrics.classification_report(y_test, y_pred, output_dict=True)
    print("Classification report for - \n{}:\n{}\n".format(clf, dt_report))
    return dt_report


def process_naiveBayes():
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.score(X_test, y_test))
    nb_report = metrics.classification_report(y_test, y_pred, output_dict=True)
    print("Classification report for - \n{}:\n{}\n".format(clf, nb_report))
    return nb_report


def process_knn():
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.score(X_test, y_test))
    gb_report = metrics.classification_report(y_test, y_pred, output_dict=True)
    print("Classification report for - \n{}:\n{}\n".format(clf, gb_report))
    return gb_report


def process_LogisticRegression():
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(solver = 'liblinear')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.score(X_test, y_test))
    lg_report = metrics.classification_report(y_test, y_pred, output_dict=True)
    print("Classification report for - \n{}:\n{}\n".format(clf, lg_report))
    return lg_report


def process_SVM():
    from sklearn.svm import SVC
    clf = SVC()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.score(X_test, y_test))
    svc_report = metrics.classification_report(y_test, y_pred, output_dict=True)
    print("Classification report for - \n{}:\n{}\n".format(clf, svc_report))
    return svc_report
