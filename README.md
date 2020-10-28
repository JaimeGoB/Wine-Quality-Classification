# Wine-Quality-Classification

The dataset contains 11 physiochemical characteristics of as predictors for White Wine. The dataset has change a little. For example, the response variables contains a range from 0-9 on the quality of wine. I have transformed this response variable, wines with quality of wine higher than 7 are considered good and lower than 7 are not.

Using this dataset I will use different classifiers to find the best one to use:

- Linear Discriminant Analysis
- Quadratic Discriminant Analysis
- Logistic Regression 
- KNN


## Measures of Perfomance of Different Classifiers

### 10 Fold Cross Validation

**I did not use the R libraries for this, instead I MANUALLY PARTIONED THE DATASET INTO 10 FOLDS AND CALCULATED THE TEST ERROR FOR UNUSED FOLD.** 

Below is a diagram of how the dataset was partitioned and error was calculated:


## AUC

Another meausure was to plot the area under the curve(AUC) for each classifier. This can be done by ploting the specificity and sensititivy. We want a plot that "elbows" top right corner and has a AUC closer to 1.

### ROC KNN

### ROC Logistic Regression

### ROC LDA

### ROC QDA

## Results

After carefully examing ROC plots and test errors, I am inclined to make the decision that using QDA as a classifier for this dataset. Quadratic Discrimant Analysis has lowest test error using 10-fold-cv and highest area under the curve(AUC).

<img src="https://github.com/JaimeGoB/Model-Building-Variable-Selection-Methods/blob/main/data/full.png" width="600" height="300" />





