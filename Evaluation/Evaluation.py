import numpy as np
import pandas as pd

def answer_one():
    df = pd.read_csv('fraud_data.csv')
    X = df.iloc[:,:29]
    y = df['Class']
    fraud_rate = len(df[df['Class'] == 1])/len(df.index)
    return fraud_rate

# Use X_train, X_test, y_train, y_test for all of the following questions
from sklearn.model_selection import train_test_split

df = pd.read_csv('fraud_data.csv')

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

def answer_two():
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import recall_score
    
    # Your code here
    dummy = DummyClassifier(strategy='most_frequent').fit(X_train,y_train)
    acc = dummy.score(X_test, y_test)
    # to get a recall score, we need y-predicted
    y_predicted = dummy.predict(X_test)
    rec = recall_score(y_test, y_predicted)    
    return (acc, rec)

def answer_three():
    from sklearn.metrics import recall_score, precision_score
    from sklearn.svm import SVC

    # Your code here
    svm = SVC().fit(X_train, y_train)
    acc = svm.score(X_test, y_test)
    
    # For recall and precision score, we need to svm.predict()
    y_predicted = svm.predict(X_test)
    rec = recall_score(y_test, y_predicted)
    prec = precision_score(y_test, y_predicted)
    return (acc, rec, prec)

def answer_four():
    from sklearn.metrics import confusion_matrix
    from sklearn.svm import SVC

    # Your code here
    svm = SVC(C= 1e9, gamma=1e-07).fit(X_train, y_train)
    # For confusion matrix, we need y_score
    y_score = svm.decision_function(X_test)
    # Let's set a threshold
    y_score = y_score > -220
    cm = confusion_matrix(y_test, y_score)
    return cm

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc
#import matplotlib.pyplot as plt

def answer_five():
    lr = LogisticRegression().fit(X_train, y_train)
    y_score_lr = lr.decision_function(X_test)
    precision, recall, thresholds = precision_recall_curve(y_test, y_score_lr)
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_score_lr)

#   plt.figure()
 #   plt.subplot(2,1,1)
#   plt.plot(precision,recall)
 #   plt.xlabel('Precision')
#   plt.xlabel('Recall')
    
#    plt.subplot(2,1,2)
#    plt.plot(fpr_lr, tpr_lr)
#    plt.xlabel('FPR Logistic Regression')
#    plt.ylabel('TPR Logistic Regression')
#    plt.show()
    
    return (np.interp(0.75, precision,recall), np.interp(0.16, fpr_lr, tpr_lr))

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

def answer_six():    

    # Your code here
    lr = LogisticRegression()
    grid_values = {'penalty': [11,12],'C': [0.01, 0.1, 1, 10, 100]}
    grid_clf = GridSearchCV(lr, param_grid = grid_values)
    grid_clf.fit(X_train, y_train)
    return grid_clf
