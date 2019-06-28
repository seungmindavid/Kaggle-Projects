import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


np.random.seed(0)
n = 15
x = np.linspace(0,10,n) + np.random.randn(n)/5
y = np.sin(x)+x/6 + np.random.randn(n)/10


X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)

# You can use this function to help you visualize the dataset by
# plotting a scatterplot of the data points
# in the training and test sets.
    
# NOTE: Uncomment the function below to visualize the data, but be sure 
# to **re-comment it before submitting this assignment to the autograder**.   
#part1_scatter()
x = x.reshape(-1,1)

def answer_one():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures

    # Your code here
    predicted_val = np.linspace(0,10,100).reshape(-1,1)
    result = []
    #print(x)
    #print(y)
    #print(xData)
    degrees = [1,3,6,9] 
    for i in degrees:
        poly = PolynomialFeatures(degree = i)
        X_poly = poly.fit_transform(x)
        X_train, X_test, y_train, y_test = train_test_split(X_poly, y, random_state = 0)
        linreg= LinearRegression().fit(X_train, y_train)
        pred = linreg.predict(poly.fit_transform(predicted_val))
        result.append(pred)
    return np.array(result)

def answer_two():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics.regression import r2_score

    # Your code here
    r2_train, r2_test = [], []
    
    for d in range(10):
        poly = PolynomialFeatures(degree= d)
        X_Poly = poly.fit_transform(x)
        X_train, X_test, y_train, y_test = train_test_split(X_Poly, y, random_state = 0)
        linreg= LinearRegression().fit(X_train, y_train)
        r2_train.append(linreg.score(X_train, y_train))
        r2_test.append(linreg.score(X_test, y_test))
    return (np.array(r2_train), np.array(r2_test))

def answer_three():
    
    # Your code here
    r2_train, r2_test = answer_two()
    df = pd.DataFrame({'R2_Train':r2_train, 'R2_Test': r2_test})
    df['fit'] = df['R2_Train'] - df['R2_Test']
    Underfitting = df[df['R2_Train'] == df['R2_Train'].min()].index[0]
    Good_Generalization = df[df['fit'] == df['fit'].min()].index[0]
    Overfitting = df[df['fit'] == df['fit'].max()].index[0]
    return (Underfitting, Overfitting, Good_Generalization)

def answer_four():
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Lasso, LinearRegression
    from sklearn.metrics.regression import r2_score

    # Your code here
    polyforLinear = PolynomialFeatures(degree= 12)
    X_Lin_Poly = polyforLinear.fit_transform(x)
    X_train, X_test, y_train, y_test = train_test_split(X_Lin_Poly, y, random_state = 0)
    linreg = LinearRegression().fit(X_train, y_train)
    linear_R2_test_score = linreg.score(X_test,y_test)

    polyforLasso = PolynomialFeatures(degree = 12)
    X_Lasso_Poly = polyforLasso.fit_transform(x)
    X2_train, X2_test, y2_train, y2_test = train_test_split(X_Lasso_Poly, y, random_state = 0)
    lassoreg = Lasso(alpha=0.01, max_iter=10000).fit(X2_train, y2_train)
    lasso_R2_test_score = lassoreg.score(X2_test, y2_test)
    return (linear_R2_test_score, lasso_R2_test_score)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


mush_df = pd.read_csv('mushrooms.csv')
mush_df2 = pd.get_dummies(mush_df)

X_mush = mush_df2.iloc[:,2:]
y_mush = mush_df2.iloc[:,1]

# use the variables X_train2, y_train2 for Question 5
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_mush, y_mush, random_state=0)

# For performance reasons in Questions 6 and 7, we will create a smaller version of the
# entire mushroom dataset for use in those questions.  For simplicity we'll just re-use
# the 25% test split created above as the representative subset.
#
# Use the variables X_subset, y_subset for Questions 6 and 7.
X_subset = X_test2
y_subset = y_test2
X_train2.shape

def answer_five():
    from sklearn.tree import DecisionTreeClassifier
#X_train2, X_test2, y_train2, y_test2 = train_test_split(X_mush, y_mush, random_state=0)
    clf = DecisionTreeClassifier(random_state =0).fit(X_train2, y_train2)
    important_features = pd.Series(clf.feature_importances_, index=X_train2.columns)
    return important_features.nlargest(n=5).index.tolist()
answer_five()

def answer_six():
    from sklearn.svm import SVC
    from sklearn.model_selection import validation_curve

    # Your code here
    clf = SVC(kernel = 'rbf', C=1, random_state = 0)
    gammas = np.logspace(-4,1,6)
    print(gammas)
    training_scores, test_scores = validation_curve(clf, X_subset, y_subset, param_name = 'gamma', param_range = gammas, cv=3)
    return (training_scores.mean(axis=1), test_scores.mean(axis=1))
answer_six()
