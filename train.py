from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from joblib import dump


if __name__ == "__main__":
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target_names[iris.target]


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)
    
    clf = LogisticRegression()

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print(metrics.classification_report(y_test, y_pred))

    dump(clf, 'model/clf.joblib')

    