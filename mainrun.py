import warnings
from sklearn.model_selection import train_test_split
from classification import train_model, evaluate_model, train_modelSVM, evaluate_modelSVM
from classification import train_modelRandomForest, evaluate_modelRandomForest
#from classification import train_GridSearchCV, evaluate_GridSearchCV
from classification_utils import preprocess_data
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from setting import config
from utils import load_data
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
#from androguard.core.bytecodes import apk
warnings.simplefilter(action='ignore', category=FutureWarning)
# need to
global c_val, epsilon_val, test_size_val, random_state_val
global c_val_max, epsilon_val_max, random_state_val_max
global accuracy_max, precision_max, recall_max
global num_benign_apps, num_malicious_apps


def initialization(malicious_count, benign_count):
    test_size_val = 0.1  # Static 0.1
    c_val = 0.021  # 1.0
    epsilon_val = 1e-3

    random_state_val = 0
    c_val_max = 0
    epsilon_val_max = 0
    test_size_val_max = 0
    random_state_val_max = random_state_val

    accuracy_max = 0
    precision_max = 0
    recall_max = 0
    return test_size_val, c_val, epsilon_val, random_state_val,\
        c_val_max, epsilon_val_max, test_size_val_max,\
        random_state_val_max, accuracy_max, precision_max, recall_max


# def extract_features(apk_dir):
#     features = []
#     labels = []
#     for filename in os.listdir(apk_dir):
#         # function that run by 2 folders b one m
#         # each folder contains a aplication
#         # the script change the file name that we can
#         if filename.endswith('.apk'):
#             try:
#                 # Open the APK file
#                 a = apk.APK(os.path.join(apk_dir, filename))
#
#                 # Extract the features
#                 feature_dict = {}
#                 feature_dict['permissions'] = a.get_permissions()
#                 feature_dict['activities'] = len(a.get_activities())
#                 feature_dict['services'] = len(a.get_services())
#                 feature_dict['receivers'] = len(a.get_receivers())
#                 feature_dict['providers'] = len(a.get_providers())
#                 feature_dict['filters'] = len(a.get_intent_filters())
#                 features.append(feature_dict)
#
#                 # Extract the label
#                 label = 1 if 'malicious' in filename else 0
#                 labels.append(label)
#
#             except:
#                 pass
#     return features, labels

def main():
    """"
     This is the main function that executes the entire process of loading the data, preprocessing it, 
     training a model, and evaluating its performance. It also allows for the classification of a new APK.

    Inputs:
        None.
    Returns:
        None.
    """

    path = config['apksResultJsonPath']

    # Load data
    df, malicious_count, benign_count = load_data(path)

    df = df.rename(columns={'label': 'class'})

    # Split data into features and labels
    X = df.drop(columns=['class'])
    y = df['class']

    test_size_val, c_val, epsilon_val, random_state_val,\
        c_val_max, epsilon_val_max, test_size_val_max,\
        random_state_val_max, accuracy_max, precision_max, recall_max = initialization(malicious_count, benign_count)

    # Preprocess data
    X_scaled, y_encoded = preprocess_data(X, y)

    # Optimization for results
    # for x in range(1, 2):

    #     # Random c and epsilon and test_size
    #     test_size_val = get_random_number(0, 1, isInt=False)  # Static 0.1
    #     c_val = get_random_number(1, 100)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=test_size_val, shuffle=True)

    # print('X_train', X_train)
    # print('X_test', X_test)
    # print('y_train', y_train)
    # print('y_test', y_test)
    # Train model LinearSVC
    model = train_model(X_train, y_train, C=c_val,
                        epsilon=epsilon_val, random_state_val=random_state_val)

    #SVM model
    model2 = train_modelSVM(X_train, y_train, C=c_val, epsilon=epsilon_val, random_state_val=random_state_val)

    #Random Forest
    model3 = train_modelRandomForest(X_train, y_train, random_state_val=random_state_val)

    #GridSearchCV
    #model4 = train_GridSearchCV(X_train, y_train, random_state_val=random_state_val)

    # Evaluate model
    accuracy, precision, recall = evaluate_model(model, X_test, y_test)
    accuracy2, precision2, recall2 = evaluate_modelSVM(model2, X_test, y_test)
    accuracy3, precision3, recall3 = evaluate_modelRandomForest(model3, X_test, y_test)
    #accuracy4, precision4, recall4 = evaluate_GridSearchCV(model4, X_test, y_test)

    compare_models(model, model2, model3, X_test, y_test)


def compare_models(LinearSVC_model, SVM_model, RandomForestmodel, GridSearchCV, X_test, y_test):
    accuracy1, precision1, recall1 = evaluate_model(LinearSVC_model, X_test, y_test)
    accuracy2, precision2, recall2 = evaluate_modelSVM(SVM_model, X_test, y_test)
    accuracy3, precision3, recall3 = evaluate_modelRandomForest(RandomForestmodel, X_test, y_test)
    #accuracy4, precision4, recall4 = evaluate_GridSearchCV(GridSearchCV, X_test, y_test)

    best_model = ""
    best_accuracy = 0
    best_precision = 0
    best_recall = 0

    models = {"LinearSVC": (accuracy1, precision1, recall1),
              "SVM": (accuracy2, precision2, recall2),
              "RandomForest": (accuracy3, precision3, recall3)}


    for model_name, values in models.items():
        accuracy, precision, recall = values
        score = accuracy + precision + recall
        if score > best_accuracy + best_precision + best_recall:
            best_model = model_name
            best_accuracy = accuracy
            best_precision = precision
            best_recall = recall

    print(f"{best_model} model performs better with accuracy: {best_accuracy}")
    print(f"{best_model} model performs better with precision: {best_precision}")
    print(f"{best_model} model performs better with recall: {best_recall}")
    print()
    print("\nOther Models Results:")
    for model_name, values in models.items():
        if model_name == best_model:
            continue
        accuracy, precision, recall = values
        print(f"{model_name} model results:")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print()

#Compare the results of the 3 models and choose the best one
# def compare_models(LinearSVC_model, SVM_model, RandomForestmodel,  X_test, y_test):
#     accuracy1, precision1, recall1 = evaluate_model(LinearSVC_model, X_test, y_test)
#     accuracy2, precision2, recall2 = evaluate_modelSVM(SVM_model, X_test, y_test)
#     accuracy3, precision3, recall3 = evaluate_modelRandomForest(RandomForestmodel, X_test, y_test)
#
#     if (accuracy1 + precision1 + recall1 > accuracy2 + precision2 + recall2):
#         print("LinearSVC model performs better with accuracy:", accuracy1)
#         print("LinearSVC model performs better with precision:", precision1)
#         print("LinearSVC model performs better with recall:", recall1)
#     else:
#         print("SVM model performs better with accuracy:", accuracy2)
#         print("SVM model performs better with precision:", precision2)
#         print("SVM model performs better with recall:", recall2)
#


if __name__ == '__main__':
    main()

