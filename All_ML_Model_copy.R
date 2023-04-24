#usage: Rscript  All_ML_Model.Rtrain sel_test1  Recurrent Tr_out_res test_out_res

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

library(MASS);
args <- commandArgs(TRUE)
set.seed(7)

#load data
train <- read.table(args[1], header = TRUE, sep = "\t", row.names = 1, check.names = FALSE)
test <- read.table(args[2], header = TRUE, sep = "\t", row.names = 1, check.names = FALSE)
#ext_val <- read.table(args[3], header = TRUE, sep = "\t", row.names = 1, check.names = FALSE)
#meth <- args[3]

train_control <- trainControl(method="repeatedcv", number=5, repeats=2,  summaryFunction = twoClassSummary, classProbs = TRUE, sampling = "up")

### For tuning parameters
#tunegrid_dt <- expand.grid(.cp=c(0.01,0.02,0.03,0.1,0.2,0.3))#DT
tunegrid_rf <- expand.grid(.mtry=c(1:50))#Rf
tunegrid_svmR <- expand.grid(sigma= 2^c(0,-1,-2,-3,-4,-5,-6,-7,-8,-9,-10), C= 2^c(0:0.2:5)) #svmR
tunegrid_svmL <- expand.grid(C = seq(0, 2, length = 20))#svmL
tunegrid_knn <- expand.grid(.k=c(seq(2,7,11)))#knn
tunegrid_nnet <- expand.grid(size = seq(from = 1, to = 10, by = 1),decay = seq(from = 0.1, to = 0.5, by = 0.1)) #nnet
tunegrid_glm <- expand.grid(parameter=c(0.001, 0.01, 0.1, 1,10,100, 1000))
tunegrid_LR_B <- expand.grid(nIter = seq(11, 51, by = 10)) 
tunegrid_glmnet <-  expand.grid(alpha = 1,lambda = seq(0.001,0.1,by = 0.001))
tunegrid_gbm <- expand.grid(interaction.depth = c(1, 5, 9),  n.trees = (1:30)*50,  shrinkage = 0.1, n.minobsinnode = 20)
tunegrid_C5 <- expand.grid( .winnow = c(TRUE,FALSE), .trials=c(1,5,10,15,20), .model="tree" )
#tunegrid_lda #no tuning parameter
tunegrid_mlpwd <- expand.grid(.decay = c(0, .01), .size = 1:3)
tunegrid_vgam<- expand.grid(parallel = TRUE, link = c("loglink", "probit"))
tunegrid_xgbD<- expand.grid(nrounds = seq(from = 100, to = 1000, by = 50),max_depth = c(2, 3, 4, 5, 6, 8), eta = c(0.025, 0.05, 0.1, 0.2), gamma = c(0,1,2,3,4), subsample = 0.8, colsample_bytree = 1, rate_drop=seq(from = 0.1, to = 1, by = 0.1), min_child_weight = 1, skip_drop=0.1)#rate_drop
tunegrid_xgbT <- expand.grid(nrounds = seq(from = 100, to = 1000, by = 50),eta = c(0.025, 0.05, 0.1, 0.2),max_depth = c(2, 3, 4, 5, 6, 8),gamma = c(0,1,2,3,4),colsample_bytree = 1,min_child_weight = 1,subsample = 0.8)#XGB
tunegrid_mlp <- expand.grid(layer1 = c(0:4),layer2 = c(0:4),layer3 = c(0:4))#mlp

# tunegrid_xgb <- expand.grid(nrounds = seq(from = 100, to = 1000, by = 50),eta = c(0.025, 0.05, 0.1, 0.2),max_depth = c(2, 3, 4, 5, 6, 8),gamma = c(0,1,2,3,),colsample_bytree = 1,min_child_weight = 1,subsample = 0.8)#XGB
tunegrid_ridge <- expand.grid(alpha = 0,lambda = 10^seq(-3, 3, length = 100))#alpha=0 for ridge
tunegrid_lasso <- expand.grid(alpha = 1,lambda = 10^seq(-3, 3, length = 100))#alpha=1 for lasso





#Adjacent Categories Probability Model for Ordinal Data


#xgbDART model
model_xgbDART <- train(as.factor(Class) ~ ., data=train, metric="ROC", trControl = train_control, method="xgbDART", tuneGrid = tunegrid_xgbD)
model_xgbDART1<- train(as.factor(Class) ~ ., data=train, metric="ROC", trControl = train_control, method="xgbDART", .nrounds=model_xgb$bestTune$nrounds, .max_depth= model_xgb$bestTune$max_depth, .eta=model_xgb$bestTune$eta, .gamma=model_xgb$bestTune$gamma, .colsample_bytree= model_xgb$bestTune$colsample_bytree, .min_child_weight= model_xgb$bestTune$min_child_weight, .subsample= model_xgb$bestTune$subsample)
res_xgbDART <- model_xgbDART1$results
save(model_xgbDART1, file= "xgbDART_model.RData")
#Test prediction
test_pred_xgbDART <- predict(model_xgbDART1, newdata=test[2:ncol(test)])
cm_xgbDART <- confusionMatrix(as.factor(test_pred_xgbDART), as.factor(test[,1]), mode='everything', positive=args[3])
cm_xgbDART1 <- as.data.frame(as.matrix(cm_xgbDART, what = "classes"))

#xgbTree model
model_xgbTree <- train(as.factor(Class) ~ ., data=train, metric="ROC", trControl = train_control, method="xgbTree", tuneGrid = tunegrid_xgbT)
model_xgbTree1 <- train(as.factor(Class) ~ ., data=train, metric="ROC", trControl = train_control, method="xgbTree", .nrounds=model_xgb$bestTune$nrounds, .max_depth= model_xgb$bestTune$max_depth, .eta=model_xgb$bestTune$eta, .gamma=model_xgb$bestTune$gamma, .colsample_bytree= model_xgb$bestTune$colsample_bytree, .min_child_weight= model_xgb$bestTune$min_child_weight, .subsample= model_xgb$bestTune$subsample)
res_xgbT<- model_xgbTree1$results
save(model_xgbTree1, file= "xgbTree_model.RData")
#Test prediction
test_pred_xgbTree <- predict(model_xgbTree1, newdata=test[2:ncol(test)])
cm_xgbTree <- confusionMatrix(as.factor(test_pred_xgbTree), as.factor(test[,1]), mode='everything', positive=args[3])
cm_xgbTree1 <- as.data.frame(as.matrix(cm_xgbTree, what = "classes"))


model_Nnet<- train(as.factor(Class) ~ ., data=train, metric="ROC", trControl = train_control, method = "nnet", tuneGrid = tunegrid_nnet)
model_Nnet1<- train(as.factor(Class) ~ ., data=train, metric="ROC", trControl = train_control, method = "nnet",.size= model_Nnet$bestTune$size, .decay= model_Nnet$bestTune$decay)
res_nnet <- model_Nnet1$results
save(model_Nnet1, file= "Nnet_model.RData")
#Test predictions
test_pred_nnet <- predict(model_nnet1, newdata=test[2:ncol(test)])
cm_nnet <- confusionMatrix(as.factor(test_pred_nnet), as.factor(test[,1]), mode='everything', positive=args[3])
cm_nnet1 <- as.data.frame(as.matrix(cm_nnet, what = "classes"))


Final_all_models_tr_res <- rbind (res_rf, res_svr, res_svmL, res_glm, res_LRB, res_glmnet, res_knn, res_stc_grd, res_C5, res_lda, res_MLP, res_vgam, res_xgbDART, res_xgbT, res_nnet)

rownames(Final_all_models_tr_res) <- c("RF", "SVM-RBF", "SVM-L", "GLM", "Boosted_LR", "GlmNet", "KNN", "Stochastic_Gradient", "C5-DT", "LDA", "VGAM", "xgbDART", "xgbTree", "Neural_NET" )

write.table(Final_all_models_tr_res,file=args[4],row.names=T,col.names=T,sep = '\t');

#Final_all_models_test_CM <- cbind(cm_rf1, cm_svr1, cm_svl1, cm_glm1, cm_LRB1, cm_glmnet1, cm_knn1, cm_stoch_grd1, cm_C5_1, cm_lda1, cm_MLP1, cm_vgam1, cm_xgbDART1, cm_xgbTree1, cm_nnet1)

#colnames(Final_all_models_test_CM) <- c("RF", "SVM-RBF", "SVM-L", "GLM", "Boosted_LR", "GlmNet", "KNN", "Stochastic_Gradient", "C5-DT", "LDA", "VGAM", "xgbDART", "xgbTree", "Neural_NET" )

write.table(Final_all_models_test_CM,file=args[5],row.names=T,col.names=T,sep = '\t');

