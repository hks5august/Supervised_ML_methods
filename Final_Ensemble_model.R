#usage: nohup Rscript Final_Ensemble_model.R sel_train sel_test sel_ext Ens_Train_result Recurrent Ensem_test1_res Ensem_test2_res Ensem_test1_pred Ensem_test2_pred &
#Load required libraries
library(car)
library(caret)
library(dplyr)
library(ggfortify)
library(ggplot2)
library(kernlab)
library(e1071)
library(randomForest)
library(DataExplorer)
library(ROSE)
library(skimr)
library(RANN)
library(fastAdaboost)
library(xgboost)
library(caretEnsemble)
library(C50)
library(earth)
library(gbm)
library(rpart)
library(bnclassify)
library(RSNNS)
library(MASS);
args <- commandArgs(TRUE)
set.seed(7)

#load data
train <- read.table(args[1], header = TRUE, sep = "\t", row.names = 1, check.names = FALSE)
#train <- read.table("sel_train", header = TRUE, sep = "\t", row.names = 1, check.names = FALSE)
test1 <- read.table(args[2], header = TRUE, sep = "\t", row.names = 1, check.names = FALSE)
#test1 <- read.table("sel_test", header = TRUE, sep = "\t", row.names = 1, check.names = FALSE)
test2 <- read.table(args[3], header = TRUE, sep = "\t", row.names = 1, check.names = FALSE)
#test2 <- read.table("sel_ext", header = TRUE, sep = "\t", row.names = 1, check.names = FALSE)
train_control <- trainControl(method="repeatedcv", number=5, repeats=2,  summaryFunction = twoClassSummary, classProbs = TRUE, sampling = "up")

algorithmList <- c('rf', 'svmRadial', "svmLinear", "glm", "LogitBoost",  "glmnet", "knn", "C5.0", "lda", "mlpWeightDecay", "xgbDART", "xgbTree", "nnet")

models <- caretList(as.factor(Class) ~ ., data=train, metric="ROC", trControl = train_control, methodList=algorithmList)


results_models <- resamples(models)
summary(results_models)
tr_results <- as.data.frame(capture.output(summary(results_models)))

#write.table(tr_results,file=args[4],row.names=T,col.names=T,sep = '\t');


# Box plots to compare models
scales <- list(x=list(relation="free"), y=list(relation="free"))
jpeg(file="Ensemble_train_models.jpeg", units="in", width=10, height=10, res=300)
bwplot(results_models, scales=scales, cex.axis= 0.5)
dev.off()


# Ensemble the predictions of `models` to form a new combined prediction based on glm
stack.glm <- caretStack(models, method="glm", metric="ROC", trControl=train_control)
print(stack.glm)

save(stack.glm, file= "Ensemble_model.RData")

ensem_tr <- as.data.frame(stack.glm$ens_model$results)

write.table(ensem_tr,file=args[4],row.names=T,col.names=T,sep = '\t');


# Prediction on testData
stack_predictions_test1 <- predict(stack.glm, newdata=test1[2:ncol(test1)])
stack_predictions_test2 <- predict(stack.glm, newdata=test2[2:ncol(test2)])

#Create confusion matrix
stack_test1_cm <- caret::confusionMatrix(as.factor(stack_predictions_test1), as.factor(test1[,1]), mode='everything', positive=args[5])
stack_test2_cm <- caret::confusionMatrix(as.factor(stack_predictions_test2), as.factor(test2[,1]), mode='everything', positive=args[5])


cm_stack_test1 <- as.data.frame(as.matrix(stack_test1_cm, what = "classes"))
cm_stack_test2 <- as.data.frame(as.matrix(stack_test2_cm, what = "classes"))


#accuracy
test1_acc <- stack_test1_cm$overall[1]
test2_acc <- stack_test2_cm$overall[1]

#cobine accuracy with results
cm_stack1 <- rbind(cm_stack_test1 , test1_acc )
cm_stack2 <- rbind(cm_stack_test2 , test2_acc )

#write into file
write.table(cm_stack1,file=args[6],row.names=T,col.names=T,sep = '\t');
write.table(cm_stack2,file=args[7],row.names=T,col.names=T,sep = '\t');


#Test predictions
test1_pred <- cbind(as.data.frame(test1[1]),as.data.frame(stack_predictions_test1))
colnames (test1_pred) <- c("Actual","Ens_pred")
test2_pred <- cbind(as.data.frame(test2[1]),as.data.frame(stack_predictions_test2))
colnames (test2_pred) <- c("Actual","Ens_pred")

write.table(test1_pred,file=args[8],row.names=T,col.names=T,sep = '\t');
write.table(test2_pred,file=args[9],row.names=T,col.names=T,sep = '\t');
