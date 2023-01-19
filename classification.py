from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_auc_score, f1_score
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

#from sklearn.model_selection import GridSearchCV
# import xgboost as xgb # XGBoost




import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def train_model(X_train, y_train, C, epsilon, random_state_val):
    """
        This function trains a Sec-SVM classifier model on the input training data.

    Inputs:
        X_train (numpy.ndarray): The feature values for the training data.
        y_train (numpy.ndarray): The labels for the training data.
        C (float): The regularization parameter.
        epsilon (float): A small constant used to determine when to stop the training.

    Returns:
        model (sklearn.svm.LinearSVC): The trained Sec-SVM model.
    """

    # Initialize the model
    model = LinearSVC(C=C, random_state=random_state_val, tol=epsilon)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    This function evaluates a model on the test data.

    Inputs:
        model (sklearn.svm.LinearSVC): The trained model.
        X_test (numpy.ndarray): The feature values for the test data.
        y_test (numpy.ndarray): The labels for the test data.

    Returns:
        accuracy (float): The accuracy score of the model.
        precision (float): The precision score of the model.
        recall (float): The recall score of the model.
    """
    # Predict the labels on the test data using the model
    # y_pred = model.predict(X_test)
    threshold = 0.5
    y_pred = model.predict(X_test)
    # y_pred[y_pred >= threshold] = 1
    # y_pred[y_pred < threshold] = 0
    # Compute the accuracy, precision, and recall scores
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    return accuracy, precision, recall


def classify_apk(model, apk_features):
    """
        This function makes a prediction for a new APK by using the trained model to classify the APK based on its features.

    Inputs:
        model (sklearn.svm.LinearSVC): The trained model.
        apk_features (numpy.ndarray): The feature values for the new APK.

    Returns:
        label (int): The predicted label for the new APK.
    """
    return model.predict(apk_features)

def train_modelSVM(X_train, y_train, C, epsilon, random_state_val):

    model = SVC(C=C, random_state=random_state_val, tol=epsilon)
    model.fit(X_train, y_train)
    return model


def evaluate_modelSVM(model, X_test, y_test):

    # Predict the labels on the test data using the model
    y_pred = model.predict(X_test)
    # Compute the accuracy, precision, and recall scores
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    return accuracy, precision, recall


def classify_apkSVM(model, apk_features):
    """
        This function makes a prediction for a new APK by using the trained model to classify the APK based on its features.

    Inputs:
        model (sklearn.svm.SVC): The trained model.
        apk_features (numpy.ndarray): The feature values for the new APK.

    Returns:
        label (int): The predicted label for the new APK.
    """
    return model.predict(apk_features)




def train_modelRandomForest(X_train, y_train, random_state_val):
    """
        This function trains a RandomForest classifier model on the input training data.

    Args:
        X_train:
        y_train:
        random_state_val:

    Returns:

    """
    # Initialize the model
    model = RandomForestClassifier(random_state=random_state_val)
    model.fit(X_train, y_train)
    return model

def evaluate_modelRandomForest(model, X_test, y_test):
    """
    This function evaluates a model on the test data.

    Inputs:
        model (sklearn.ensemble.RandomForestClassifier): The trained model.
        X_test (numpy.ndarray): The feature values for the test data.
        y_test (numpy.ndarray): The labels for the test data.

    Returns:
        accuracy (float): The accuracy score of the model.
        precision (float): The precision score of the model.
        recall (float): The recall score of the model.
    """
    # Predict the labels on the test data using the model
    y_pred = model.predict(X_test)
    # Compute the accuracy, precision, and recall scores
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    return accuracy, precision, recall

def classify_apkRandomForest(model, apk_features):
    """
        This function makes a prediction for a new APK by using the trained model to classify the APK based on its features.

    Inputs:
        model (sklearn.ensemble.RandomForestClassifier): The trained model.
        apk_features (numpy.ndarray): The feature values for the new APK.

    Returns:
        label (int): The predicted label for the new APK.
    """
    return model.predict(apk_features)

#
# #gridSearchModel training
# def train_GridSearchCV( X_train, y_train, random_state_val):
#     param_grid = {'C': [0.1, 1, 10, 100],
#                   'kernel': ['linear', 'rbf'],
#                   'gamma': [0.1, 1, 10, 100]}
#     svm = SVC()
#     grid_search = GridSearchCV(svm, param_grid, cv=5)
#     grid_search.fit(X_train, y_train)
#     return grid_search.best_estimator_
#
# #gridSearchModel evaluation
# def evaluate_GridSearchCV(model, X_test, y_test):
#     y_pred = model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#     return accuracy, precision, recall
#
# #gridSearchModel classification
# def classify_apkGridSearchCV(model, apk_features):
#     return model.predict(apk_features)
# #
# def train_modelXGBoost(X_train, y_train, random_state_val):
#     """
#         This function trains a XGBoost classifier model on the input training data.
#
#     Args:
#         X_train:
#         y_train:
#         random_state_val:
#
#     Returns:
#
#     """
#     # Initialize the model
#     model = XGBClassifier(random_state=random_state_val)
#     model.fit(X_train, y_train)
#     return model
#
# def evaluate_modelXGBoost(model, X_test, y_test):
#     """
#     This function evaluates a model on the test data.
#
#     Inputs:
#         model (sklearn.ensemble.XGBClassifier): The trained model.
#         X_test (numpy.ndarray): The feature values for the test data.
#         y_test (numpy.ndarray): The labels for the test data.
#
#     Returns:
#         accuracy (float): The accuracy score of the model.
#         precision (float): The precision score of the model.
#         recall (float): The recall score of the model.
#     """
#     # Predict the labels on the test data using the model
#     y_pred = model.predict(X_test)
#     # Compute the accuracy, precision, and recall scores
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#
#     return accuracy, precision, recall
#
# def classify_apkXGBoost(model, apk_features):
#     """
#         This function makes a prediction for a new APK by using the trained model to classify the APK based on its features.
#
#     Inputs:
#         model (sklearn.ensemble.XGBClassifier): The trained model.
#         apk_features (numpy.ndarray): The feature values for the new APK.
#
#     Returns:
#         label (int): The predicted label for the new APK.
#     """
#     return model.predict(apk_features)
#
#
#
# def train_modelKNN(X_train, y_train, random_state_val):
#     """
#         This function trains a KNN classifier model on the input training data.
#
#     Args:
#         X_train:
#         y_train:
#         random_state_val:
#
#     Returns:
#
#     """
#     # Initialize the model
#     model = KNeighborsClassifier()
#     model.fit(X_train, y_train)
#     return model
#
# def evaluate_modelKNN(model, X_test, y_test):
#     """
#     This function evaluates a model on the test data.
#
#     Inputs:
#         model (sklearn.neighbors.KNeighborsClassifier): The trained model.
#         X_test (numpy.ndarray): The feature values for the test data.
#         y_test (numpy.ndarray): The labels for the test data.
#
#     Returns:
#         accuracy (float): The accuracy score of the model.
#         precision (float): The precision score of the model.
#         recall (float): The recall score of the model.
#     """
#     # Predict the labels on the test data using the model
#     y_pred = model.predict(X_test)
#     # Compute the accuracy, precision, and recall scores
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#
#     return accuracy, precision, recall
#
# def classify_apkKNN(model, apk_features):
#     """
#         This function makes a prediction for a new APK by using the trained model to classify the APK based on its features.
#
#     Inputs:
#         model (sklearn.neighbors.KNeighborsClassifier): The trained model.
#         apk_features (numpy.ndarray): The feature values for the new APK.
#
#     Returns:
#         label (int): The predicted label for the new APK.
#     """
#     return model.predict(apk_features)
#
# def train_modelDecisionTree(X_train, y_train, random_state_val):
#     """
#         This function trains a DecisionTree classifier model on the input training data.
#
#     Args:
#         X_train:
#         y_train:
#         random_state_val:
#
#     Returns:
#
#     """
#     # Initialize the model
#     model = DecisionTreeClassifier(random_state=random_state_val)
#     model.fit(X_train, y_train)
#     return model
#
# def evaluate_modelDecisionTree(model, X_test, y_test):
#     """
#     This function evaluates a model on the test data.
#
#     Inputs:
#         model (sklearn.tree.DecisionTreeClassifier): The trained model.
#         X_test (numpy.ndarray): The feature values for the test data.
#         y_test (numpy.ndarray): The labels for the test data.
#
#     Returns:
#         accuracy (float): The accuracy score of the model.
#         precision (float): The precision score of the model.
#         recall (float): The recall score of the model.
#     """
#     # Predict the labels on the test data using the model
#     y_pred = model.predict(X_test)
#     # Compute the accuracy, precision, and recall scores
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#
#     return accuracy, precision, recall
#
# def classify_apkDecisionTree(model, apk_features):
#     """
#         This function makes a prediction for a new APK by using the trained model to classify the APK based on its features.
#
#     Inputs:
#         model (sklearn.tree.DecisionTreeClassifier): The trained model.
#         apk_features (numpy.ndarray): The feature values for the new APK.
#
#     Returns:
#         label (int): The predicted label for the new APK.
#     """
#     return model.predict(apk_features)


