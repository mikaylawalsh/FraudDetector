import pandas as pd
from sklearn import linear_model, metrics

"""
LOAD CSV AND SEPARATE LABELS (WHAT WE'RE TRYING TO PREDICT) FROM FEATURES (THE DATA HELPING US PREDICT)
"""
data = pd.read_csv('creditcard.csv')
#fraud or not fraud
labels = data['Class']
#everything else
features = data[["Time","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount"]]

"""
SPLIT LABELS/FEATURES SUCH THAT WE USE 80% FOR TRAINING AND 20% FOR TESTING
"""
split_index = int(len(data)*.8) #get 80% of the data

testing_labels = labels.iloc[split_index:] #get all rows after split index (this is 20%)
testing_features =  features.iloc[split_index:]
training_labels = labels.iloc[:split_index]
training_features = features.iloc[:split_index]

"""
INITIALIZE MODEL W/ BALANCING & FIT TO TRAINING DATA
"""
lr_model = linear_model.LogisticRegression(class_weight='balanced')

lr_model = lr_model.fit(training_features, training_labels)

"""
TEST MODEL
"""
preds = lr_model.predict(testing_features) #model is predicts the labels for testing features
accuracy = metrics.accuracy_score(testing_labels, preds)#pass predications and testing labels into accuracy to see how accurate model was
print(accuracy)