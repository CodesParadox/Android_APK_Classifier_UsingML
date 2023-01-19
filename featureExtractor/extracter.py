# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import os
from androguard.core.bytecodes import apk


def extract_features(apk_dir):
    features = []
    labels = []
    for filename in os.listdir(apk_dir):
        if filename.endswith('.apk'):
            try:
                # Open the APK file
                a = apk.APK(os.path.join(apk_dir, filename))
                
                # Extract the features
                feature_dict = {}
                feature_dict['permissions'] = a.get_permissions()
                feature_dict['activities'] = len(a.get_activities())
                feature_dict['services'] = len(a.get_services())
                feature_dict['receivers'] = len(a.get_receivers())
                feature_dict['providers'] = len(a.get_providers())
                feature_dict['filters'] = len(a.get_intent_filters())
                features.append(feature_dict)
                
                # Extract the label
                label = 1 if 'malicious' in filename else 0
                labels.append(label)
                
            except:
                pass
    # Create a dataframe from the extracted features and labels
    df = pd.DataFrame(features)
    df['label'] = labels
    return df


# Extract features from APK files in the specified directory
apk_dir = '/home/kali/tool/apps/'
df = extract_features(apk_dir)

# Write the dataframe to a CSV file
df.to_csv('apks.csv', index=False)

# Load the dataset of APK features and labels
#df = pd.read_csv('apks.csv')
X = df.drop(['label'], axis=1)
y = df['label']

# Feature selection
selector = SelectKBest(chi2, k=1000)
X = selector.fit_transform(X, y)


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create an SVM classifier with a linear kernel
clf = SVC(kernel='linear', C=1.0)

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Evaluate the classifier's accuracy
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

# Generate confusion matrix
confusion_matrix(y_test, y_pred)

