library(class)
library(cvAUC)
library(pROC)
#lda
library(MASS)
library(caret)

################################################
# 3.  Dichotomize the quality variable as good, which takes the value 1 if quality ≥ 7 and the value 0, otherwise. 
# We will take good as response and all the 11 physiochemical characteristics of the wines in the data as predictors.
# Use 10-fold cross-validation for estimating the test error rates below and compute the estimates using caret package 
# with seed set to 1234 before each computation.
################################################

#reading in dataset
wine <- read.csv("winequality-white.csv", header = T, sep = ";")
wine$quality <- ifelse(wine$quality >= 7, 1, 0)
names(wine)[12] <- "good"

#This is to fit knn
wine2 <- read.csv("winequality-white.csv", header = T, sep = ";")
wine2$quality <- ifelse(wine2$quality >= 7, 1, 0)
wine2$quality = as.factor(wine2$quality)
names(wine2)[12] <- "good"

#Splitting train and test set
train_Y = wine[,12]
train_X = wine[,-12]

# (a) Fit a KNN with K chosen optimally using test error rate. 
#Also, report its estimated test error rate.
set.seed(1234)

#FINDING OPTIMAL K IN KNN
train_control <- trainControl(method="cv", number=10)
knn.fit <- train(good ~ ., 
                 method = "knn", 
                 tuneGrid = expand.grid(k = 2:100), 
                 trControl  = train_control,
                 metric = "Accuracy", 
                 data = wine2)
results_from_all_k <- data.frame(knn.fit$results)

optimal_k_accuracry <- results_from_all_k[results_from_all_k$Accuracy == max(results_from_all_k$Accuracy),]
optimal_k_accuracry
optimal_k = optimal_k_accuracry[1] #3
optimal_k

#Measures of performance for optimal K in KNN based on the training data. 
specificity_vector_knn <- c()
sensitivity_vector_knn <- c()
train_error_vector_knn <- c()
test_error_vector_knn <- c()
auc_vector_knn <- c()

#creating a 10 fold partition
folds <- cut(seq(1,nrow(wine)), breaks = 10, labels = F)
for(fold in 1:10) {
  
  current_fold <- which(folds == fold, arr.ind = T)
  test = wine[current_fold, ]
  train = wine[-current_fold, ]
  train_X = train[, -12]
  train_Y = train[, 12]
  test_X = test[, -12]
  test_Y = test[, 12]
  
  set.seed(1234)
  model_train <- knn(train_X, train_X, train_Y, k = optimal_k)
  set.seed(1234)
  model_test <- knn(train_X, test_X, train_Y, k = optimal_k)
  
  confusion_matrix_knn = table(model_train, train_Y)
  confusion_matrix_knn_test = table(model_test, test_Y)
  
  #sensitivity 
  specificity_vector_knn[fold] <- confusion_matrix_knn[2, 2] / (confusion_matrix_knn[1, 2] + confusion_matrix_knn[2, 2])
  #specificity, 
  sensitivity_vector_knn[fold] <- confusion_matrix_knn[1, 1] / (confusion_matrix_knn[1, 1] + confusion_matrix_knn[2, 1])
  #train error
  train_error_vector_knn[fold] = mean(model_train!=train_Y)
  #train error
  test_error_vector_knn[fold] = mean(model_test!=test_Y)
  #and AUC for the optimal KNN 
  auc_vector_knn[fold] <- auc(roc(model_train, train_Y))
}

specificity_knn = mean(specificity_vector_knn)
sensitivity_knn = mean(sensitivity_vector_knn)
train_error_knn = mean(train_error_vector_knn)
test_error_knn = mean(test_error_vector_knn)
auc_knn = round( mean(auc_vector_knn) , 3)

knn_list <- c(specificity_knn, sensitivity_knn, train_error_knn, test_error_knn,auc_knn)

#Plotting ROC curve for knn
knn_pred_prob <- predict(knn.fit, wine, type = "prob")
roc_curve_knn <- roc(train_Y, knn_pred_prob[,1])
plot(roc_curve_knn, col = 'black', main = paste("ROC Curve KNN \nAUC ", auc_knn) )
legend("bottomright",
       legend=c("ROC Curve - KNN"),
       col=c("black"),
       lty=c(1))

# (b) Repeat (a) using logistic regression.
set.seed(1234)
train_error_vector_glm <- c()
test_error_vector_glm <- c()
specificity_vector_glm <- c()
sensitivity_vector_glm <- c()
auc_vector_glm <- c()

#creating a 10 fold partition
folds <- cut(seq(1,nrow(wine)), breaks = 10, labels = F)
for(fold in 1:10) {
  #setting train and test
  current_fold <- which(folds == fold, arr.ind = T)
  test = wine[current_fold, ]
  train = wine[-current_fold, ]
  train_X = train[, -12]
  train_Y = train[, 12]
  test_X = test[, -12]
  test_Y = test[, 12]
  
  # fitting glm
  set.seed(1234)
  model_train <- glm(good ~ ., family = binomial, data = train)
  set.seed(1234)
  model_test<- glm(good ~ ., family = binomial, data = test)
  
  #prediction for glm
  prediction_train <- predict(model_train, train_X, type = "response")
  prediction_glm_train <- ifelse(prediction_train >= 0.5, "1", "0")
  prediction_test <- predict(model_test, test_X, type = "response")
  prediction_glm_test <- ifelse(prediction_test >= 0.5, "1", "0")
  
  #confusion matrix
  confusion_matrix_train = table(prediction_glm_train, train_Y)
  confusion_matrix_test = table(prediction_glm_test, test_Y)
  
  #sensitivity
  specificity_vector_glm[fold] <- confusion_matrix_train [2, 2] / (confusion_matrix_train [1, 2] + confusion_matrix_train [2, 2])
  #specificity,
  sensitivity_vector_glm[fold] <- confusion_matrix_train [1, 1] / (confusion_matrix_train [1, 1] + confusion_matrix_train [2, 1])
  # training error 
  train_error_vector_glm[fold] = mean(prediction_glm_train!=train_Y)
  #testing error
  test_error_vector_glm[fold] = mean(prediction_glm_test!=test_Y)
  
  #AUC
  auc_vector_glm[fold] <- auc(roc(prediction_glm_train, train_Y))[1]
}

specificity_glm = mean(specificity_vector_glm)
sensitivity_glm = mean(sensitivity_vector_glm)
train_error_glm = mean(train_error_vector_glm)
test_error_glm = mean(test_error_vector_glm)
auc_glm = round( mean(auc_vector_glm) , 3)

glm_list <- c(specificity_glm, sensitivity_glm, train_error_glm, test_error_glm,auc_glm)

#Plotting ROC curve
logistic_regression_fit <- glm(good ~ ., family = binomial,data = wine)
logistic_regression_pred <- predict(logistic_regression_fit, type = 'response')

roc_curve_glm <- roc(good ~ logistic_regression_pred, data = wine)

plot(roc_curve_glm, col = 'blue', main=paste("ROC Curve Logistic \nAUC ", auc_glm))
legend("bottomright",
       legend=c("ROC Curve - LDA"),
       col=c("blue"),
       lty=c(1))

# (c) Repeat (a) using LDA.
set.seed(1234)
train_error_vector_lda <- c()
test_error_vector_lda <- c()
specificity_vector_lda <- c()
sensitivity_vector_lda <- c()
auc_vector_lda <- c()

#creating a 10 fold partition
folds <- cut(seq(1,nrow(wine)), breaks = 10, labels = F)
for(fold in 1:10) {
  #setting train and test
  current_fold <- which(folds == fold, arr.ind = T)
  test = wine[current_fold, ]
  train = wine[-current_fold, ]
  train_X = train[, -12]
  train_Y = train[, 12]
  test_X = test[, -12]
  test_Y = test[, 12]
  
  # fitting lda
  set.seed(1234)
  model_train <- lda(good ~ ., data = train)
  set.seed(1234)
  model_test<- lda(good ~ ., data = test)
  #prediction for lda
  prediction_lda_train <- predict(model_train, train_X)
  prediction_lda_test <- predict(model_test, test_X)

  #confusion matrix
  confusion_matrix_train = table(prediction_lda_train$class, train_Y)
  confusion_matrix_test = table(prediction_lda_test$class, test_Y)
  
  #sensitivity
  specificity_vector_lda[fold] <- confusion_matrix_train [2, 2] / (confusion_matrix_train [1, 2] + confusion_matrix_train [2, 2])
  #specificity,
  sensitivity_vector_lda[fold] <- confusion_matrix_train [1, 1] / (confusion_matrix_train [1, 1] + confusion_matrix_train [2, 1])
  # training error 
  train_error_vector_lda[fold] = mean(prediction_lda_train$class!=train_Y)
  #testing error
  test_error_vector_lda[fold] = mean(prediction_lda_test$class!=test_Y)
  
  #AUC
  auc_vector_lda[fold] <- auc(roc(prediction_lda_train$class, train_Y))[1]
}

specificity_lda = mean(specificity_vector_lda)
sensitivity_lda = mean(sensitivity_vector_lda)
train_error_lda = mean(train_error_vector_lda)
test_error_lda = mean(test_error_vector_lda)
auc_lda = round( mean(auc_vector_lda) , 3)

lda_list <- c(specificity_lda, sensitivity_lda, train_error_lda, test_error_lda,auc_lda)

#Plotting ROC curve  model_train 
lda_fit <- lda(good ~ ., data = wine)
lda_pred <- predict(lda_fit, train_X)
roc_curve_lda <- roc(train_Y, lda_pred$posterior[,"1"], levels = c("0", "1"))

plot(roc_curve_lda, col = 'blue', main=paste("ROC Curve LDA \nAUC ", auc_lda))
legend("bottomright",
       legend=c("ROC Curve - LDA"),
       col=c("blue"),
       lty=c(1))

# (d) Repeat (a) using QDA.
set.seed(1234)
train_error_vector_qda <- c()
test_error_vector_qda <- c()
specificity_vector_qda <- c()
sensitivity_vector_qda <- c()
auc_vector_qda <- c()


#creating a 10 fold partition
folds <- cut(seq(1,nrow(wine)), breaks = 10, labels = F)
for(fold in 1:10) {
  #setting train and test
  current_fold <- which(folds == fold, arr.ind = T)
  test = wine[current_fold, ]
  train = wine[-current_fold, ]
  train_X = train[, -12]
  train_Y = train[, 12]
  test_X = test[, -12]
  test_Y = test[, 12]
  
  # fitting qda
  set.seed(1234)
  model_train <- qda(good ~ ., data = train)
  set.seed(1234)
  model_test<- qda(good ~ ., data = test)
  #prediction for qda
  prediction_qda_train <- predict(model_train, train_X)
  prediction_qda_test <- predict(model_test, test_X)
  
  #confusion matrix
  confusion_matrix_train = table(prediction_qda_train$class, train_Y)
  confusion_matrix_test = table(prediction_qda_test$class, test_Y)
  
  #sensitivity
  specificity_vector_qda[fold] <- confusion_matrix_train [2, 2] / (confusion_matrix_train [1, 2] + confusion_matrix_train [2, 2])

  #specificity,
  sensitivity_vector_qda[fold] <- confusion_matrix_train [1, 1] / (confusion_matrix_train [1, 1] + confusion_matrix_train [2, 1])

  # training error 
  train_error_vector_qda[fold] = mean(prediction_qda_train$class!=train_Y)
  #testing error
  test_error_vector_qda[fold] = mean(prediction_qda_test$class!=test_Y)
  
  #AUC
  auc_vector_qda[fold] <- auc(roc(prediction_qda_train$class, train_Y))[1]
}

specificity_qda = mean(specificity_vector_qda)
sensitivity_qda = mean(sensitivity_vector_qda)
train_error_qda = mean(train_error_vector_qda)
test_error_qda = mean(test_error_vector_qda)
auc_qda = round( mean(auc_vector_qda) ,3)

qda_list <- c(specificity_qda, sensitivity_qda, train_error_qda, test_error_qda,auc_qda)

#Plotting ROC curve
qda_fit <- qda(good ~ ., data = wine)
qda_pred <- predict(qda_fit, train_X)

roc_curve_qda <- roc(train_Y, qda_pred$posterior[,"1"], levels = c("0", "1"))

plot(roc_curve_qda, col = 'red', main=paste("ROC Curve QDA \nAUC ", auc_qda))
legend("bottomright",
       legend=c("ROC Curve - QDA"),
       col=c("red"),
       lty=c(1))

# (e) Compare the results in (a)-(d). Which classifier would you recommend? Justify your answer.

all_models <- rbind(knn_list, glm_list, lda_list, qda_list)
colnames(all_models) <- c("Specificity", "Sensitivity", "Training Error", "Testing Error", "AUC")
rownames(all_models) <- c("knn","logistic","lda", "qda")
all_models


#Comparing all classifiers
plot(roc_curve_lda, col = 'blue', main = "ROC Curve Binary Classifcations")
lines(roc_curve_qda, col = 'red')
lines(roc_curve_glm, col = 'green')
lines(roc_curve_knn, col = 'black')
legend("bottomright",
       legend=c("Linear Discriminant Analysis\nAUC ", "Quadratic Discriminant Analysis", "Logistic Regression", "KNN"),
       col=c("blue", "red", "green", "black"),
       lty=c(1))

