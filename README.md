# Android_APK_Classifier_UsingML

## Classification

### Files
#### Part 1
* [**classification.py**](https://github.com/CodesParadox/Android_APK_Classifier_UsingML/blob/main/classification.py)  
  - **Choosing The Model**    
    Here we use different models in Machine Learning to classify the APK files.
    We use the following models:
  - **LinearSVC**
  - **SVM**
  - **Random Forest**
  - **KNN**
  - **GridSearch**
  - **XGBoost**

**_**Note that the GridSearch is used to find the best parameters for the model but its take a lot of time so better not use it.****_
  
#### Part 2
- **Training The Model**

     Here we train the model with the data we have. We use the data from the [**data.json**](https://github.com/CodesParadox/Android_APK_Classifier_UsingML/blob/main/data/apks/result/data.json)
    after we extract the data from the APK files, and label it -> label 1 for malicious and 0 for benign.
```train_model(x_train, y_train, model)``` is used for training a Linear Support Vector Classification (SVC) model, which is a type of supervised machine learning algorithm that can be used for classification problems.
- #### This function takes in four inputs, the training data (X_train, y_train), regularization parameter (C), a small constant used to determine when to stop the training (epsilon), and random_state_val, which is a seed used by the random number generator. 
  - The function initializes the LinearSVC model with the input parameters, then fits the model to the training data using the fit() method and returns the trained model

    -   *X_train:* a numpy array of feature values for the training data.
    -	*y_train:* a numpy array of labels for the training data.
    -	*C:* a float, the regularization parameter. In SVM, regularization parameter is used to control the trade-off between maximizing the margin and minimizing the misclassification rate. Larger values of C will result in a larger margin but a smaller misclassification rate.
    -	*epsilon:* a small constant used to determine when to stop the training. In LinearSVC, the tol parameter is used to control the stopping criterion. The optimization will stop when the difference between the objective values at two consecutive iterations is less than epsilon.
    -	*random_state_val:* an integer, a seed used by the random number generator to generate random numbers.

The function first initializes the LinearSVC (Or other models such as SVM, RandomForest, etc..)model with the input parameters, then it fits the model to the training data using the fit() method.
 The fit() method takes in the feature values and labels of the training data and learns the model parameters that best fit the training data. 
Finally, the function returns the trained model.

- **Testing The Model**

The second function ```evaluate_model(model, X_test, y_test)``` is used for evaluating a trained model on a test dataset. 
The function takes in three inputs, the trained model, the feature values for the test data (X_test), and the labels for the test data (y_test). 
Then, it predicts the labels on the test data using the model and computes the accuracy, precision, and recall scores using the accuracy_score(), precision_score(), and recall_score() functions from the sklearn.metrics library. 
Then it plots the results by calling plotting function on the first 100 samples of the test data (y_pred[:100], y_test[:100]).
 
- The function initializes the LinearSVC model with the input parameters, then fits the model to the training data using the fit() method and returns the trained model

    - *X_train:* a numpy array of feature values for the training data.
    - *y_train:* a numpy array of labels for the training data.


- #### The function first uses the trained model to predict the labels on the test data using the predict() method. Then, it uses the accuracy_score(), precision_score(), and recall_score() functions from the sklearn.metrics library to compute the accuracy, precision, and recall scores respectively. These are the three most commonly used evaluation metrics in classification tasks.
*	The accuracy score is the proportion of correct predictions made by the model out of the total number of predictions.
*	The precision score is the proportion of true positive predictions made by the model out of the total number of positive predictions.
*	The recall score is the proportion of true positive predictions made by the model out of the total number of actual positive instances.
 
-  **Prediction For A New APK**    

This function makes a prediction for a new APK by using the trained model to classify the APK based on its features.
  The function takes in two inputs, the trained model and the path to the APK file. 
  Then, it extracts the features from the APK file using the extract_features() function, and uses the trained model to predict the label of the APK file using the predict() method. 
  Finally, it returns the predicted label of the APK file.

The last function  ```classify_apk(model, apk_features)``` is used to make a prediction for a new APK by using the trained model. The inputs to this function are:
-	model: a trained LinearSVC model
-	apk_features: a numpy array of feature values for the new APK.
The function makes the prediction by using the predict() method on the model and return the predicted label for the new APK.

 #### The function takes in two inputs, the trained model and the feature values for the new APK (apk_features).
 #### The function makes the prediction by using the predict() method on the model and return the predicted label for the new APK.
    
