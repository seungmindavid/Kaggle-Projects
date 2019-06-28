import pandas as pd
import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier


def blight_model():
    
    # Your code here
    train_data = pd.read_csv('train.csv', encoding = "ISO-8859-1")
    test_data = pd.read_csv('test.csv', encoding = "ISO-8859-1")
    
    # Let's set index to ticket_id
    train_data.index = train_data['ticket_id']
    test_data.index = test_data['ticket_id']
    # Let's drop rows which complicance value is N/A
    train_data = train_data.dropna(subset=['compliance'])
    #print(train_data)
    #train_data.to_csv('train_datas.csv', sep='\t', encoding='ISO-8859-1')
    # Let's remove columns with string by using get_numeric_data() function
    #get_numeric_features = train_data._get_numeric_data().columns.tolist()
    #print(get_numeric_features)

    # Let's split x_data and y_data to split train and test 
    features = ['fine_amount', 'admin_fee', 'state_fee', 'late_fee', 'discount_amount', 'clean_up_cost']
    X_train_data = train_data[features]
    y_train_data = train_data['compliance']   
    # Let's use train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_train_data, y_train_data, random_state = 0)

    # Let's scale with MinMaxScaler
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)
    
    clf = RandomForestClassifier()
    clf.fit(X_train_scaled, y_train)
    # We can see that 'fine_amount', 'late_fee', 'discount_amount' has high feature importances 
    # Admin_fee, State_fee, 'Clean_up_cost has 0% of importances in this data
    features = ['fine_amount', 'late_fee', 'discount_amount']
    X_train_data = train_data[features]
    X_train, X_test, y_train, y_test = train_test_split(X_train_data, y_train_data, random_state = 0)

    # Let's scale with MinMaxScaler
    scaler = MinMaxScaler()
    X_train_scaled2 = scaler.fit_transform(X_train)
    X_test_scaled2 = scaler.fit_transform(X_test)
    
    myclf = RandomForestClassifier()
    myclf.fit(X_train_scaled2, y_train)
    
    train_score = myclf.score(X_train_scaled2, y_train)
    test_score = myclf.score(X_test_scaled2, y_test)
    # Train score = 0.93569
    # Test score = 0.93462
    # Which is very great fit
    
    # Now, let's predict test data
    prediction = myclf.predict_proba(test_data[features])[:,1]
    res = pd.Series(data= prediction, index = test_data['ticket_id'], dtype='float32')
    return res
