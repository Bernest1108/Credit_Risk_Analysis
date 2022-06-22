# Credit_Risk_Analysis
## Overview
The purpose of this study was to create and train a machine learning model in order to accurately predict high risk loans. This was done using the `imbalanced-learn` and `scikit-learn` libraries in Python so as to use the `RandomOversampler` and `SMOTE` for over-sampling, `ClusterCentoids` for under-sampling, `SMOTEEN` in order to combine over and under-sampling. Then also for Ensemble Learners we used `BalancedRandomForestClassifier` and `AdaBoost'.

## Results
### Over-sampling
#### Naive Oversampling Algorithm
* Balanced Accuracy Score: 64.64%
* Precision: 1%
* Recall: 71%

Confusion Matrix

![image](https://user-images.githubusercontent.com/100445489/175098732-03ef9828-4e10-4712-b0cd-8158da47fc0d.png)


Classification Report

![image](https://user-images.githubusercontent.com/100445489/175098847-48e81c31-7ccb-48e2-915a-5f38aeae66bd.png)


#### SMOTE Oversampling Algorithm
* Balanced Accuracy Score: 65.86%
* Precision: 1%
* Recall: 63%

Confusion Matrix

![image](https://user-images.githubusercontent.com/100445489/175098945-a47efb10-816a-4e37-b624-967d3ea94728.png)


Classification Report

![image](https://user-images.githubusercontent.com/100445489/175099008-37eb5988-8cfa-40c7-857c-041ed713e13c.png)


### Under-sampling
#### ClusterCentroids Algorithm
* Balanced Accuracy Score: 54.47%
* Precision: 1%
* Recall: 69%

Confusion Matrix

![undersampling_cm](https://github.com/aKnownSaltMine/Credit_Risk_Analysis/blob/main/Results/undersampling_cm.PNG)

Classification Report

![undersampling_class_report](https://github.com/aKnownSaltMine/Credit_Risk_Analysis/blob/main/Results/undersampling_class_report.PNG)

### Combination Over- and Under-Sampling
#### SMOTEEN Algorithm
* Balanced Accuracy Score: 64.80%
* Precision: 1%
* Recall: 72%

Confusion Matrix

![smoteen_cm](https://github.com/aKnownSaltMine/Credit_Risk_Analysis/blob/main/Results/smoteen_cm.PNG)

Classification Report

![image](https://user-images.githubusercontent.com/100445489/175099124-c093d52d-e3d1-463f-902c-099820900971.png)


### Ensemble Learners
#### Balanced Random Forest Classifier Algorithm
* Balanced Accuracy Score: 78.85%
* Precision: 3%
* Recall: 70%

Confusion Matrix

![image](https://user-images.githubusercontent.com/100445489/175099319-3db20eff-8803-4f76-93f2-5dea21d85916.png)


Classification Report

![image](https://user-images.githubusercontent.com/100445489/175099422-fc80897a-c6a2-4b3c-ad80-c11f9bdfb575.png)


#### Easy Ensemble AdaBoost Classifier
* Balanced Accuracy Score: 93.17%
* Precision: 9%
* Recall: 92%

Confusion Matrix

![adaboost_cm](https://github.com/aKnownSaltMine/Credit_Risk_Analysis/blob/main/Results/adaboost_cm.PNG)

Classification Report

![image](https://user-images.githubusercontent.com/100445489/175099493-b6d1e8a9-fbf3-4da0-bf68-e99fd4bda45f.png)


## Summary
The balanced accuracy scores for each of the models in descending order are: 
* Easy Ensemble AdaBoost Classifier: 93.17%
* Balanced Random Forest Classifier Algorithm: 78.85%
* SMOTE Oversampling Algorithm: 65.86%
* SMOTEEN Algorithm: 64.80%
* Naive Oversampling Algorithm: 64.64%
* ClusterCentroids Algorithm: 54.47%

When taking loans into consideration, most of the time, it is better to be more cautious than to be more willing to give out loans to people who will not pay back. Most of the models were actually very good at predicting low risk loans, which could have value in its own right, but the goal was to build a model in order to predict high risk loans. By far the model with the largest Balanced Accuracy score was the Easy Ensemble Adaboost Classifier. Taking a look at the model's confusion matrix, we can see that it only missed 8 instances where it considered a high risk loan low risk while the rest of the models hovered around 30 that were missed. It also was able to reduce the false positives by quite a bit as well. The Easy Ensemble AdaBoost model would be the one that I recommend using moving forward.
