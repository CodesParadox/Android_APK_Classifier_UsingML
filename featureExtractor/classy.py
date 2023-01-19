import os
import json
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from imblearn.over_sampling import RandomOverSampler
from sklearn import svm
from androguard.core.bytecodes import apk

# Extract features from an APK file
def extract_features(apk_file):
    try:
        a = apk.APK(apk_file)
        permissions = a.get_permissions()
        api_calls = a.get_all_methods()
        manifest = json.loads(a.get_android_manifest_xml())
        package_name = manifest["package"]
        version_name = a.get_androidversion_name()
        version_code = a.get_androidversion_code()
        return permissions, api_calls, package_name, version_name, version_code
    except Exception as e:
        print("Error in file: ", apk_file)
        print(e)
        return None, None, None, None, None

# Prepare the dataset
X = []
y = []

# with open(f'./dataset_{str(dataset_number)}.json') as file:
#     raw_ds = json.load(file)
# df = pd.json_normalize(raw_ds, max_level=2)

 # directory of your dataset
dataset_dir ='./home/kali/tool/apps/'

for file in os.listdir(dataset_dir):
    if file.endswith(".apk"):
        permissions, api_calls, package_name, version_name, version_code = extract_features(os.path.join(dataset_dir, file))
        if permissions is not None and api_calls is not None:
            features = permissions + api_calls + [package_name] + [version_name] + [version_code]
            X.append(features)
            if "malware" in file:
                y.append(1)
            else:
                y.append(0)

# Normalize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Handle class imbalance
ros = RandomOverSampler()
X, y = ros.fit_resample(X, y)

# Split the data into training and test sets
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

# Grid search to find the best hyper-parameters
param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': [0.1, 1]}
clf = svm.SVC()
grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='roc_auc')
grid_search.fit(X_train, y_train)

# Print the best hyper-parameters
print("Best hyper-parameters:", grid_search.best_params_)

# Train the classifier with the best hyper-parameters
clf = svm.SVC(C=grid_search.best_params_['C'], kernel=grid_search.best_params_['kernel'],
              gamma=grid_search.best_params_['gamma'])
clf.fit(X_train, y_train)

# Predict on the test dataset
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)

# Evaluate the performance of the classifier
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Confusion Matrix: ", confusion_matrix(y_test, y_pred))
print("Classification Report: ", classification_report(y_test, y_pred))
print("AUC-ROC: ", roc_auc_score(y_test, y_prob[:, 1]))

