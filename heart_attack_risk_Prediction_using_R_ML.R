library(ROSE)
library(tidyr)
library(naniar)
library(tidyverse)
library(PerformanceAnalytics)
library(corrplot)
library(rpart)
library(rpart.plot)
library(party)
library(e1071)
library(ROSE)
library(xgboost)
library(pROC)
library(ggplot2)
library(DMwR2)
library(gbm)
library(DataExplorer)
library(caret)

setwd("C:/Users/elangovan.paramasiva/Downloads/R")
Heartriskdata <- read.csv(file = "ht_risk.csv", header = TRUE, stringsAsFactors = FALSE)

#------Data cleaning--------------
# Checking rows and col and their types
str(Heartriskdata)

# Converting all categorical value into factor
Heartriskdata$Sex <- as.factor(Heartriskdata$Sex)
Heartriskdata$Diet <- as.factor(Heartriskdata$Diet)
Heartriskdata$Country <- as.factor(Heartriskdata$Country)
Heartriskdata$Continent <- as.factor(Heartriskdata$Continent)
Heartriskdata$Hemisphere <- as.factor(Heartriskdata$Hemisphere)

# Splitting blood pressure into systolic and diastolic values
Heartriskdata <- separate(Heartriskdata, col = Blood.Pressure, into = c("sysBP", "diasBP"), sep = "/")
Heartriskdata$sysBP <- as.numeric(Heartriskdata$sysBP)
Heartriskdata$diasBP <- as.numeric(Heartriskdata$diasBP)

# Check for missing values
vis_miss(Heartriskdata)

# remove duplicates
Heartriskdata <- Heartriskdata %>% distinct()



#calculating skewness, kurtosis, moct only for continuous or discrete variables. herre all variables showing nesr to 0 means no skewness
Heartriskdata_numeric <- Heartriskdata %>%
  select(where(is.numeric))
skewness_values <- apply(Heartriskdata_numeric, 2, skewness)
print(skewness_values)
kurtosis_values <- apply(Heartriskdata_numeric, 2, kurtosis)
print(kurtosis_values)

#understand the data distribution across numerical features
par(mfrow = c(2, 3))
# Plot histograms for the original variables without x-axis labels
hist(Heartriskdata$Exercise.Hours.Per.Week, main = "Exercise Hours Per Week", col = "lightblue", xaxt = "n")
hist(Heartriskdata$Sedentary.Hours.Per.Day, main = "Sedentary Hours Per Day", col = "lightgreen", xaxt = "n")
hist(Heartriskdata$Income, main = "Income", col = "lightcoral", xaxt = "n")
hist(Heartriskdata$BMI, main = "BMI", col = "lightgray", xaxt = "n")
hist(Heartriskdata$Age, main = "Age", col = "lightyellow", xaxt = "n")
hist(Heartriskdata$Sleep.Hours.Per.Day, main = "Sleep Hour", col = "lightpink", xaxt = "n")

# Reset the layout to default
par(mfrow = c(1, 1))

# Create a 2x3 layout for the plots
par(mfrow = c(2, 3))
hist(Heartriskdata$Triglycerides, main = "Triglycerides", col = "lightpink", xaxt = "n")
hist(Heartriskdata$Heart.Rate, main = "Heart Rate", col = "lightblue", xaxt = "n")
hist(Heartriskdata$Cholesterol, main = "Cholesterol", col = "lightgreen", xaxt = "n")
hist(Heartriskdata$sysBP, main = "SysBP", col = "lightcoral", xaxt = "n")
hist(Heartriskdata$diasBP, main = "DiasBP", col = "lightyellow", xaxt = "n")
# Reset the layout to default
par(mfrow = c(1, 1))

ggplot(data = Heartriskdata ) + geom_bar(mapping = aes(x = Diet)) # almost equal dist
ggplot(data = Heartriskdata ) + geom_bar(mapping = aes(x = Continent))
ggplot(data = Heartriskdata ) + geom_bar(mapping = aes(x = Hemisphere)) # more asia ansd EU data there
ggplot(data = Heartriskdata ) + geom_bar(mapping = aes(x = Sex)) # more male data are there
ggplot(data = Heartriskdata ) + geom_bar(mapping = aes(x = Heart.Attack.Risk)) 

#outlier detection for continuous numerical value features. No outlier on any columns
ggplot(Heartriskdata)+geom_boxplot(mapping = aes(x = Exercise.Hours.Per.Week))
ggplot(Heartriskdata)+geom_boxplot(mapping = aes(x = Sedentary.Hours.Per.Day))
ggplot(Heartriskdata)+geom_boxplot(mapping = aes(x = Cholesterol))
ggplot(Heartriskdata)+geom_boxplot(mapping = aes(x = Triglycerides))
ggplot(Heartriskdata)+geom_boxplot(mapping = aes(x = BMI))

library(ggplot2)
library(reshape2)

# Melt the data into long format
melted_data <- melt(Heartriskdata, measure.vars = c("Exercise.Hours.Per.Week", "Sedentary.Hours.Per.Day", "Cholesterol", "Triglycerides", "BMI"))

# Create a combined boxplot with rotated x-axis labels
combined_boxplot <- ggplot(melted_data, aes(x = variable, y = value, fill = variable)) +
  geom_boxplot(position = position_dodge(width = 0.8), alpha = 0.7) +
  ggtitle("Combined Boxplot to find Outlier") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Print the combined boxplot
print(combined_boxplot)

#correlation of dependentant variables with other independent 
#Seperate numeric column 
Heartriskdata_numeric <- Heartriskdata %>%
  select(where(is.numeric))

#correlation table generated
correlation_matrixtable <- cor(Heartriskdata_numeric, method = "pearson")
print(correlation_matrixtable)

library(corrplot)
par(mfrow = c(1, 1))  # Set the number of rows and columns in the layout. srt is rotation and cex is tex size
# Adjust rotation angle and size of text labels
corrplot(correlation_matrixtable,type='lower', 
         tl.col = "black", tl.srt = 90, tl.cex = 0.78)

# circle based corelation and variance explained by how many features
library(factoextra)
pca_hr <- princomp(Heartriskdata_numeric, cor = TRUE, scores = TRUE)	

fviz_eig(pca_hr)

fviz_pca_var(pca_hr, col.var = "contrib", repel = TRUE)
fviz_pca_biplot(pca_hr, repel = TRUE)



# domain knowledge 
Heartriskdata$mbp <- (2/3 * Heartriskdata$diasBP) + (1/3 * Heartriskdata$sysBP)
Heartriskdata <- Heartriskdata[-4]
Heartriskdata <- Heartriskdata[-4]
# Removing obesity as in BMI it will be covered
Heartriskdata<-Heartriskdata[-8]

#country is having 20 different categorical value so when we do one hot encoding 20+ new columns are introduced. training and performance will gte impact
#label encoding also not possible as it may negatively impact the model in diff way as there is no high/low value across region.
# frequency encoding also not possible as there is repeated values and frequency doesnt denotes proper representation for country
#we are doing target encoding for country column to see how much its with target variable

# Convert ordinal to numeric value
Heartriskdata$Diet <- ifelse(Heartriskdata$Diet == "Healthy", 3, ifelse(Heartriskdata$Diet == "Average", 2, 1))
# Convert column gender to separate it
Heartriskdata$Sex.Male <- ifelse(Heartriskdata$Sex == "Male", 1, 0)
Heartriskdata$Sex.Female <- ifelse(Heartriskdata$Sex == "Female", 1, 0)
Heartriskdata <- Heartriskdata[-2]
# Convert column Hemisphere to separate it
Heartriskdata$Hemisphere.North <- ifelse(Heartriskdata$Hemisphere == "Northern Hemisphere", 1, 0)
Heartriskdata$Hemisphere.South <- ifelse(Heartriskdata$Hemisphere == "Southern Hemisphere", 1, 0)
Heartriskdata <- Heartriskdata[-21]

# Target Encoding for Continent
Heartriskdata <- Heartriskdata %>%
  group_by(Continent) %>%
  mutate(Continent_mean = mean(Heart.Attack.Risk)) %>%
  ungroup()
Heartriskdata <- Heartriskdata %>% select(-Continent)

# Feature Removal obesity and country
Heartriskdata <- Heartriskdata[-19]
ggplot(data = Heartriskdata) + geom_count(mapping = aes(x = BMI, y = Obesity))

#scaling. for age,heart rate, BMI not needed.inc and triglyceride optional. exer hours and sedantary hours data are in broder so we will try toscale it.
#we have 2 types of scaling. lets see the distribution of data and decide which is better

#Scaling Min max scaler 
rs_function <- function(x) (x - min(x)) / (max(x) - min(x))
Heartriskdata$Exercise.Hours.Per.Week <- rs_function(Heartriskdata$Exercise.Hours.Per.Week)
#Heartriskdata <- Heartriskdata[-11]
Heartriskdata$Sedentary.Hours.Per.Day <- rs_function(Heartriskdata$Sedentary.Hours.Per.Day)
#Heartriskdata <- Heartriskdata[-15]
#--------------------------------------------------------------------
# Spliiting the data in 80/20 ratio.
set.seed(123)
Heartriskdata[, "train"] <- ifelse(runif(nrow(Heartriskdata)) < 0.8, 1, 0)
trainset <- Heartriskdata[Heartriskdata$train == 1, ]
testset <- Heartriskdata[Heartriskdata$train == 0, ]
trainset <- trainset[-26]
testset <- testset[-26]


#----------------------------Model training 
# Check for class imbalance
table(Heartriskdata$Heart.Attack.Risk)
table(trainset$Heart.Attack.Risk)
class_distribution <- table(trainset$Heart.Attack.Risk)

# Create a bar plot
barplot(class_distribution, main = "Current Class Distribution",
        xlab = "Heart Attack Risk", ylab = "Frequency",
        col = c("lightblue", "lightcoral"), legend = TRUE)


# Oversample the minority class using ROSE package
trainset_balanced <- ROSE(Heart.Attack.Risk ~ ., data = trainset, seed = 1)$data


# Check for class imbalance
class_distributionbal <- table(trainset_balanced$Heart.Attack.Risk)
print(class_distributionbal)
# Create a bar plot
barplot(class_distributionbal, main = "Class Distribution after Balanced",
        xlab = "Heart Attack Risk", ylab = "Frequency",
        col = c("lightblue", "lightcoral"), legend = TRUE)

#------creating a fucntion to generate metrics--------------
calculate_metrics <- function(y_true, y_pred_prob) {
  confusion_matrix <- table(Actual = y_true, Predicted = y_pred_prob)
  
  true_positive <- confusion_matrix[2, 2]
  false_positive <- confusion_matrix[1, 2]
  false_negative <- confusion_matrix[2, 1]
  true_negative <- confusion_matrix[1, 1]
  
  accuracy <- (true_positive + true_negative) / sum(confusion_matrix)
  precision <- true_positive / (true_positive + false_positive)
  recall <- true_positive / (true_positive + false_negative)
  f1_score <- 2 * precision * recall / (precision + recall)
  
  roc_curve <- roc(y_true, y_pred_prob)
  auc <- auc(roc_curve)
  
  mcc <- ((true_positive * true_negative) - (false_positive * false_negative)) /
    (sqrt(true_positive + false_positive) * sqrt(true_positive + false_negative) *sqrt
     (true_negative + false_positive) * sqrt(true_negative + false_negative))
  
  
  result <- c(Accuracy = accuracy, Precision = precision, Recall = recall,
              F1_Score = f1_score,AUC=auc, MCC = mcc)
  return(result)
}


#--------------------------------------------------------- Decision Tree
Htrsk_dcstree <- rpart(Heart.Attack.Risk ~ ., data = trainset_balanced, method = 'class')
rpart.plot(Htrsk_dcstree, extra = 106)
#removing the train feature as splitting is done otherwise it will inhibit effectiveness f classification

test_data <- testset[-19]
Htrsk_dcstreeprd <- predict(Htrsk_dcstree, newdata = test_data, type = "class")

summary(Htrsk_dcstree)
summary(Htrsk_dcstreeprd)

#decision tree accuracy for seed 123 60%
DT_metrics <- calculate_metrics(testset$Heart.Attack.Risk, as.numeric(Htrsk_dcstreeprd))
print(DT_metrics)

# Assuming you have a model with caret compatible structure
Htrsk_dcstree_caret <- train(Heart.Attack.Risk ~ ., data = trainset_balanced, method = 'rpart', trControl = trainControl(method = "cv"))

# Get variable importance
var_importance <- varImp(Htrsk_dcstree)

# Print the variable importance
print(var_importance)

#------------------------------------------------------Random forest select 
formula_rf <- Heart.Attack.Risk ~ .

# Convert Heart.Attack.Risk to factor with two levels
trainset$Heart.Attack.Risk <- as.factor(trainset$Heart.Attack.Risk)
testset$Heart.Attack.Risk <- as.factor(testset$Heart.Attack.Risk)

crf_model <- cforest(Heart.Attack.Risk ~ ., data = trainset, control = cforest_unbiased(mtry = 24, ntree = 500))
crf_predictions <- predict(crf_model, newdata = test_data)
#rf_pred <- ifelse(crf_predictions>0.5, 1, 0)
crf_predictions <- as.factor(crf_predictions)
#accuracy for  24 feat 200 tree is 63.4%  

RF_metrics <- calculate_metrics(testset$Heart.Attack.Risk, as.numeric(crf_predictions))
print(RF_metrics)

CRF_VarImp <- varimp(crf_model) 
barplot(CRF_VarImp)

#------------------------------------------SVM select
svm_trainset <- trainset_balanced
svm_trainset$Heart.Attack.Risk <- as.factor(svm_trainset$Heart.Attack.Risk)
SVMtr <- svm(Heart.Attack.Risk ~ ., data = svm_trainset)
SVMpred <- predict(SVMtr, newdata = test_data, type = "response")
# accuracy is 54.7% 
SVM_metrics <- calculate_metrics(testset$Heart.Attack.Risk, as.numeric(SVMpred))
print(SVM_metrics)

summary(SVMtr)

#-----------------------------------------Naive Bayes
NBtr <- naiveBayes(Heart.Attack.Risk ~ ., data = trainset_balanced)
NBpred <- predict(NBtr, newdata = test_data)
#accuracy is 58%
NB_metrics <- calculate_metrics(testset$Heart.Attack.Risk, as.numeric(NBpred))
print(NB_metrics)

#-----------------------------------------GBM
# Train a Gradient Boosting model
GBMtrain <- gbm(Heart.Attack.Risk ~ ., data = trainset_balanced, distribution = "bernoulli", n.trees = 200, interaction.depth = 15, shrinkage = 0.1)

# Predictions on the test set
GBMpred <- predict(GBMtrain, newdata = test_data, type = "response", n.trees = 200)

# Predictions on the test set
#gbm_pred <- predict(GBMtrain, newdata = test_data_numeric[, -16], type = "response", n.trees = 100)

# Convert predictions to binary (0 or 1)
gbm_pred_binary <- ifelse(GBMpred > 0.5, 1, 0)
#accuracy is 57.5 % 
# Evaluate metrics
GBM_metrics <- calculate_metrics(testset$Heart.Attack.Risk, gbm_pred_binary)
print(GBM_metrics)

#---------------------------------------XG BOOST

library(xgboost)

# Convert factors to numeric as xgboost doesn't handle factors
trainset_balanced_numeric <- as.data.frame(lapply(trainset_balanced, as.numeric))
test_data_numeric <- as.data.frame(lapply(test_data, as.numeric))

# Train XGBoost model
xgb_model <- xgboost(data = as.matrix(trainset_balanced_numeric[, -19]), 
                     label = trainset_balanced$Heart.Attack.Risk, 
                     nrounds = 100, objective = "binary:logistic", eval_metric = "logloss")

# Predictions on the test set
xgb_pred <- predict(xgb_model, as.matrix(test_data_numeric))

# Convert predictions to binary (0 or 1)
xgb_pred_binary <- ifelse(xgb_pred > 0.5, 1, 0)
# accuracy is 55.8%
# Calculate metrics
XGB_metrics <- calculate_metrics(testset$Heart.Attack.Risk, xgb_pred_binary)
print(XGB_metrics)

#---------------------------------------------------------------------

# Assuming crf_model is your trained Conditional Random Forest model
library(caret)

# Extract feature importance
CRF_VarImp <- varImp(crf_model) 

# Plot feature importance
plot(CRF_VarImp, main = "Conditional Random Forest Feature Importance")


# Assuming SVMtr is your trained SVM model
library(e1071)

# Extract feature weights
SVM_VarImp <- abs(coef(SVMtr))

# Plot feature importance
barplot(SVM_VarImp, names.arg = colnames(SVM_VarImp), main = "SVM Feature Importance")


# Assuming xgb_model is your trained XGBoost model
library(xgboost)

# Plot feature importance
xgb.plot.importance(model = xgb_model, importance_type = "weight", main = "XGBoost Feature Importance")



library(gbm)
# Plot feature importance
plot(GBMtrain, main = "Gradient Boosting Feature Importance")






#------------------------------------------------------------ensemble method ----------


#-------------------------------Ensemble method

# Assuming you have SVM, Naive Bayes, GBM, and XGBoost models already trained and predictions available
# Assuming you have SVM, Naive Bayes, GBM, and XGBoost models already trained and predictions available
# Example for SVM predictions
length(svm3_pred)

# Example for Naive Bayes predictions
length(nb_predict)

# Example for GBM predictions
length(gbm_pred_binary)

# Example for Random Forest predictions
length(rf_pred_binary)

# Example for XGBoost predictions
length(xgb_pred_binary)


# Combine predictions into a data frame
ensemble_predictions <- data.frame(SVM = as.numeric(svm3_pred),
                                   NB = as.numeric(nb_predict),
                                   DT = as.numeric(tree3m_pred),
                                   GBM = gbm_pred_binary,
                                   RF = rf_pred_binary,
                                   XGB = xgb_pred_binary)
ensemble_predictions <- data.frame(SVM = as.numeric(SVMpred), 
                                   RF = as.numeric(crf_predictions),
                                   XGB = xgb_pred_binary)

sapply(ensemble_predictions, class)

# Assuming majority voting for binary classification
final_predictions <- ifelse(rowMeans(ensemble_predictions) > 0.5, 1, 0)

# Calculate metrics for the ensemble model
Ensemble_metrics <- calculate_metrics(testset$Heart.Attack.Risk, final_predictions)
print(Ensemble_metrics)

#------------------------------------------------------
# Assuming crf_model is your trained Conditional Random Forest model
library(caret)
library(ggplot2)
install.packages("viridis")
library(viridis)
# Extract feature importance
CRF_VarImp <- varImp(crf_model)

# Assuming crf_model is your trained Conditional Random Forest model
library(caret)
library(ggplot2)
library(viridis)  # You can use any other color palette library

# Extract feature importance
CRF_VarImp <- varImp(crf_model)

# Create a data frame for plotting
plot_data <- data.frame(
  Feature = rownames(CRF_VarImp),
  Importance = CRF_VarImp$Overall
)

# Sort the data frame by importance for better visualization
plot_data <- plot_data[order(-plot_data$Importance), ]

# Create a bar plot using ggplot2 with a color palette
ggplot(plot_data, aes(x = Feature, y = Importance, fill = Feature)) +
  geom_bar(stat = "identity") +
  scale_fill_viridis(discrete = TRUE) +  # You can use any other color palette
  labs(title = "Conditional Random Forest Feature Importance") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1) )

# feat imp for SVM

# Assuming SVMtr is your trained SVM model
library(e1071)

# Extract feature weights
SVM_VarImp <- abs(coef(SVMtr))

# Plot feature importance
barplot(SVM_VarImp, names.arg = colnames(SVM_VarImp), main = "SVM Feature Importance")

#             XG Boost
# Assuming xgb_model is your trained XGBoost model
library(xgboost)

# Plot feature importance
xgb.plot.importance(model = xgb_model, importance_type = "weight", main = "XGBoost Feature Importance")


# Gradient boost
