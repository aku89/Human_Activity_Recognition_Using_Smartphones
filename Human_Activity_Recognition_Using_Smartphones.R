adress <- "http://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
temp <-  "C:/Users/Artur/Rprojects/Human_Activity_Recognition_Using_Smartphones/data/HAR.zip"
download.file(adress, temp)

unziped_path <- "C:/Users/Artur/Rprojects/Human_Activity_Recognition_Using_Smartphones/data/unziped"
ziped_data_path <- "C:/Users/Artur/Rprojects/Human_Activity_Recognition_Using_Smartphones/data/HAR.zip"
unzip(ziped_data_path, exdir=unziped_path)

if(!require(tidyverse)) install.packages("tidyverse"); library(tidyverse)
if(!require(caret)) install.packages("caret"); library(caret)
if(!require(stats)) install.packages("stats"); library(stats)
if(!require(gbm)) install.packages("gbm"); library(gbm)


#the orinators split the data into 4 sets (train with response and predictors, test with response and predictors)
X_train <- read.table("C:/Users/Artur/Rprojects/Human_Activity_Recognition_Using_Smartphones/data/unziped/UCI HAR Dataset/train/X_train.txt", sep="")
y_train <- read.table("C:/Users/Artur/Rprojects/Human_Activity_Recognition_Using_Smartphones/data/unziped/UCI HAR Dataset/train/y_train.txt", sep="")
X_test <- read.table("C:/Users/Artur/Rprojects/Human_Activity_Recognition_Using_Smartphones/data/unziped/UCI HAR Dataset/test/X_test.txt", sep="")
y_test <- read.table("C:/Users/Artur/Rprojects/Human_Activity_Recognition_Using_Smartphones/data/unziped/UCI HAR Dataset/test/y_test.txt", sep="")

columns <- read.table("C:/Users/Artur/Rprojects/Human_Activity_Recognition_Using_Smartphones/data/unziped/UCI HAR Dataset/features.txt", sep= "")

#i dislike the partition
#a validation set and a small test set would be preferrable
# so i make my own:
train <- cbind(X_train, y_train)
test <- cbind(X_test, y_test)
df <- data.frame(rbind(train, test))
names <- append(seq(1:561), "outcome")
colnames(df) <- names

#first, lets clear up some memory
rm(temp, adress, unziped_path, ziped_data_path, X_train, X_test, y_train, y_test, train, test, columns, aproach_1)
gc()

#I will use a larger training, a smaller testing and a final validation set
validation_index <- createDataPartition(y = df[, 562], times=1, p=0.15, list=FALSE)
validation <- df[validation_index, ]
temp <- df[-validation_index, ]


test_index <- createDataPartition(y = temp[, 562], times=1, p=0.15, list=FALSE)
testing <- temp[test_index, ]
training <- temp[-test_index, ]

training <- training%>%
  mutate(outcome =as.factor(outcome))

rm(test_index, temp)
gc()

#looking at data
dim(df) #more than 100k instances and 561 features

unique(df$outcome) #our outcome variables are coded in integers and represent different classes of human activity (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING)
#I am not really sure whether it makes sense to look at the distribution of the features since it 1. is a classification problem and
#2. these are 561 features

#what could be useful is the distribution of the outcome
df%>%
  group_by(outcome)%>%
  summarize(n=n())%>%
  ggplot(aes(x=outcome, y=n))+
  geom_bar(stat="identity")
#we can see here that activity 2 and definetly 3 are underrepresented
#in my experience this might lead to some bias, but I can not say to what extent

system.time({
  gbm_fit <- train(outcome ~ ., data=training, method="gbm", metric="Accuracy")
})
#took my notebook 15561sec till it was finished while doing some other stuff
#but the default parameters are more than sufficient, as I could see in with the testing set

summary(gbm_fit)%>%head(10)
#here we can see the 10 most important features from the perspective of the gbm-algorithm
#if we wanted to make the algorithm faster, we could limit the number of features based on this info

gbm_prediciton <- predict(gbm_fit, testing)
confMa <- confusionMatrix(as.factor(testing$outcome), gbm_prediciton)
#pretty good overall accuracy of 97.87% and for each category even without further tuning


#lets see how long it takes for KNN and how well it performs
system.time({
    knn_fit <- train(outcome ~., data=training, method="knn", metric="Accuracy", tuneGrid=data.frame(k=seq(1,5,1)))
})# took  4274.59 sec on my notebook while doing other stuff
#knn_fit$bestTune gave me k=1
#normally I would think this is due to overtraining, but since I tested it on the testing data I was curious.
#https://stats.stackexchange.com/questions/107870/does-k-nn-with-k-1-always-implies-overfitting here it is said that large space..
# ..between the different groups of outcomes, K=1 does not lead to overfitting and with that many features in my dataset,
# there is much space spanned by the 561 features for the datapoints 
# it also seems that I am very lucky with my neat dataset which has not many/large outliers

knn_prediction <- predict(knn_fit, testing)
confMa_knn <- confusionMatrix(as.factor(testing$outcome), knn_prediction)
#overall accuracy of about 96.12% and for each category

#for PCA, I remove the outcome-variable
pca.training <- training[, -562]
pcompon <- prcomp(pca.training, center=TRUE, scale=TRUE)
#since prcomp uses SVD centering and scaling is neccessary
#also since features can have different measuring units the variance can differ much and this will effect large loadings

pca.var <- pcompon$sdev^2
pca.var.perc <- round(pca.var/sum(pca.var), 3)

#making an elbow (or scree) plot:
elbow <- cumsum(pca.var.perc)
length(elbow[elbow <= 0.950]) #gives me the value of 104. This is the number (n-1) of PC´s which explain 95% of the variance
#this is considerable less than 561 variables 
#this will give us a boost in time needed to fit a gbm-model, but will we be able to predict as good as before?

new_training <- as.data.frame(pcompon$x[, seq(1,105)])%>%
  cbind(training$outcome)%>%
  rename(outcome = `training$outcome`) #the hidden quote signs from cbind-func drove me nuts xD

pca.testing <- testing[, -562]
#projecting the testing set onto the pca-space by centering(with train-mean),scaling(with train-sd) and multiplying with the loadings
new_testing <- data.frame(scale(pca.testing, pcompon$scale, pcompon$center)%*%pcompon$rotation)
new_testing <- new_testing[, seq(1,105)]

system.time({
  gbm_pca <- train(outcome ~., data=new_training, method="gbm", metric="Accuracy")
})#waaaay faster with 1378 seconds

gbm_pca_prediciton <- predict(gbm_pca, new_testing)
confMa2 <- confusionMatrix(as.factor(testing$outcome), gbm_pca_prediciton)
#dadum.. the overall accuracy droped down to ca. 23%


#lets train and test knn with help of pca
system.time({
  knn_pca <- train(outcome ~ ., data=new_training, method="knn", metric="Accuracy", tuneGrid=data.frame(k=seq(1,5,1)))
})#398 sec, didnt think it would be faster than gbm!

knn_pca_prediction <- predict(knn_pca, new_testing)
confMa_knn_pca <- confusionMatrix(as.factor(testing$outcome), knn_pca_prediction)
#overall accuracy droped down to ca 18%

#lets clear some memory again
rm(pca.training, pca.testing, pcompon, elbow, pca.var, pca.var.perc, confMa, confMa2, confMa_knn, confMa_knn_pca, new_training, new_testing, gbm_pca_prediciton, knn_pca_prediction, gbm_pca, knn_pca)
gc()


#Final predictions:
final_gbm <- predict(gbm_fit, validation[, -562])
confMa_val_gbm <- confusionMatrix(as.factor(validation$outcome), final_gbm)
#overall Accuracy of 98,86% <- slightly overtrained!

final_knn <- predict(knn_fit, validation[, -562])
confMa_val_gbm <- confusionMatrix(as.factor(validation$outcome), final_knn)
#overall Accuracy of 97,41% <- slightly overtrained!
