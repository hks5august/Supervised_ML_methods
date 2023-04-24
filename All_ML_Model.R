#usage: nohup Rscript All_ML_Model.R sel_train1 sel_test1  Recurrent Tr_out_res test_out_res test_predictions &
# sel_train1 - Training Data
# sel_test1  - Test data
# Recurrent - Positive class Label present in data
# Tr_out_res -  Test result output file name
# test_predictions - test prediction output file name

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
tunegrid_vgam<- expand.grid(parallel = TRUE, link = c("logit", "probit"))
tunegrid_xgbD<- expand.grid(nrounds = seq(from = 100, to = 1000, by = 50),eta = c(0.025, 0.05, 0.1, 0.2),max_depth = c(2, 3, 4, 5, 6, 8),gamma = c(0,1,2,3,4),colsample_bytree = 1,min_child_weight = 1,subsample = 0.8, rate_drop=seq(from = 0.1, to = 1, by = 0.1), skip_drop=seq(from = 0.1, to = 1, by = 0.1))#rate_drop
tunegrid_xgbT <- expand.grid(nrounds = seq(from = 100, to = 1000, by = 50),eta = c(0.025, 0.05, 0.1, 0.2),max_depth = c(2, 3, 4, 5, 6, 8),gamma = c(0,1,2,3,4),colsample_bytree = 1,min_child_weight = 1,subsample = 0.8)#XGB
tunegrid_mlp <- expand.grid(layer1 = c(0:4),layer2 = c(0:4),layer3 = c(0:4))#mlp

# tunegrid_xgb <- expand.grid(nrounds = seq(from = 100, to = 1000, by = 50),eta = c(0.025, 0.05, 0.1, 0.2),max_depth = c(2, 3, 4, 5, 6, 8),gamma = c(0,1,2,3,),colsample_bytree = 1,min_child_weight = 1,subsample = 0.8)#XGB
tunegrid_ridge <- expand.grid(alpha = 0,lambda = 10^seq(-3, 3, length = 100))#alpha=0 for ridge
tunegrid_lasso <- expand.grid(alpha = 1,lambda = 10^seq(-3, 3, length = 100))#alpha=1 for lasso

#parameters <- tunegrid_rf$bestTune, tunegrid_svmR$bestTune, tunegrid_svmL$bestTune, tunegrid_knn$bestTune, tunegrid_nnet$bestTune, tunegrid_glm$bestTune, tunegrid_LR_B$bestTune , tunegrid_gbm$bestTune , tunegrid_C5$bestTune, tunegrid_mlpwd$bestTune, tunegrid_vgam$bestTune,tunegrid_xgbD$bestTune, tunegrid_xgbT$bestTune, tunegrid_mlp$bestTune  


#Random Forest
model_rf <- train(as.factor(Class) ~ ., data=train, metric="ROC", trControl = train_control, method="rf", tuneGrid=tunegrid_rf)
model_rf1 <- train(as.factor(Class) ~ ., data=train, metric="ROC", trControl = train_control, method="rf", .mtry= model_rf$bestTune$mtry)
save(model_rf1, file= "RF_model.RData")
#res_rf <- model_rf1$results
res_rf <- head(as.data.frame(model_rf1$results[order(model_rf1$results[,"ROC"], decreasing =  TRUE),]),1)
#Test prediction
test_pred_rf <- predict(model_rf1, newdata=test[2:ncol(test)])
cm_rf <- confusionMatrix(as.factor(test_pred_rf), as.factor(test[,1]), mode='everything', positive=args[3])
cm_rf1 <- as.data.frame(as.matrix(cm_rf, what = "classes"))

#SVM-RBF
model_svmR <- train(as.factor(Class) ~ ., data=train, metric="ROC", trControl = train_control, method="svmRadial", tuneGrid=tunegrid_svmR)
model_svmR
model_svmR1 <- train(as.factor(Class) ~ ., data=train, metric="ROC", trControl = train_control, method="svmRadial", .sigma= model_svmR$bestTune$sigma,.C= model_svmR$bestTune$C)
#res_svr <- model_svmR1$results
res_svr <- head(as.data.frame(model_svmR1$results[order(model_svmR1$results[,"ROC"], decreasing =  TRUE),]),1)
save(model_svmR1, file= "SVR_model.RData")
#Test prediction
test_pred_svr <- predict(model_svmR1, newdata=test[2:ncol(test)])
cm_svr <- confusionMatrix(as.factor(test_pred_svr), as.factor(test[,1]), mode='everything', positive=args[3])
cm_svr1 <- as.data.frame(as.matrix(cm_svr, what = "classes"))

#Linear SVM
model_svmL <- train(as.factor(Class) ~ ., data=train, metric="ROC", trControl = train_control, method="svmLinear", tuneGrid = expand.grid(C = seq(0, 2, length = 20)))
model_svmL1 <- train(as.factor(Class) ~ ., data=train, metric="ROC", trControl = train_control, method="svmLinear", .C=model_svmL$bestTune$C)
#res_svmL<- model_svmL1$results
res_svmL<- head(as.data.frame(model_svmL1$results[order(model_svmL1$results[,"ROC"], decreasing =  TRUE),]),1)
save(model_svmL1, file=  "svmL_model.RData")
#Test prediction
test_pred_svl <- predict(model_svmL1, newdata=test[2:ncol(test)])
cm_svl <- confusionMatrix(as.factor(test_pred_svl), as.factor(test[,1]), mode='everything', positive=args[3])
cm_svl1 <- as.data.frame(as.matrix(cm_svl, what = "classes"))


#LR model
model_glm <- train(as.factor(Class) ~ ., data=train, metric="ROC", trControl = train_control, method="glm", tuneGrid = tunegrid_glm)
model_glm1 <- train(as.factor(Class) ~ ., data=train, metric="ROC", trControl = train_control, method="glm")
#res_glm <- model_glm1$results
res_glm <- head(as.data.frame(model_glm1$results[order(model_glm1$results[,"ROC"], decreasing =  TRUE),]),1)
save(model_glm1, file= "Glm_model.RData")
#Test prediction
test_pred_glm <- predict(model_glm1, newdata=test[2:ncol(test)])
cm_glm <- confusionMatrix(as.factor(test_pred_glm), as.factor(test[,1]), mode='everything', positive=args[3])
cm_glm1 <- as.data.frame(as.matrix(cm_glm, what = "classes"))


#Boosted Logistic Regression
model_LR_B<- train(as.factor(Class) ~ ., data=train, metric="ROC", trControl = train_control, method = "LogitBoost", tuneGrid =tunegrid_LR_B)
model_LR_B1<- train(as.factor(Class) ~ ., data=train, metric="ROC", trControl = train_control, method = "LogitBoost",.nIter=model_LR_B$bestTune$nIter)
#res_LRB<- model_LR_B1$results
res_LRB<- head(as.data.frame(model_LR_B1$results[order(model_LR_B1$results[,"ROC"], decreasing =  TRUE),]),1)
save(model_LR_B1, file= "LR_B_model.RData")
#Test prediction
test_pred_LRB <- predict(model_LR_B1, newdata=test[2:ncol(test)])
cm_LRB <- confusionMatrix(as.factor(test_pred_LRB), as.factor(test[,1]), mode='everything', positive=args[3])
cm_LRB1 <- as.data.frame(as.matrix(cm_LRB, what = "classes"))


#GLMnet model
model_glmnet <- train(as.factor(Class) ~ ., data=train, metric="ROC", trControl = train_control, method="glmnet", tuneGrid = tunegrid_glmnet)
model_glmnet1 <- train(as.factor(Class) ~ ., data=train, metric="ROC", trControl = train_control, method="glmnet", .alpha=model_glmnet$bestTune$alpha, .lambda= model_glmnet$bestTune$lambda)
#res_glmnet<- model_glmnet1$results
res_glmnet<- head(as.data.frame(model_glmnet1$results[order(model_glmnet1$results[,"ROC"], decreasing =  TRUE),]),1)
save(model_glmnet1, file= "glmnet_model.RData")
#Test prediction
test_pred_glmnet <- predict(model_glmnet1, newdata=test[2:ncol(test)])
cm_glmnet <- confusionMatrix(as.factor(test_pred_glmnet), as.factor(test[,1]), mode='everything', positive=args[3])
cm_glmnet1 <- as.data.frame(as.matrix(cm_glmnet, what = "classes"))


#KNN model
model_knn1 <- train(as.factor(Class) ~ ., data=train, metric="ROC", trControl = train_control, method="knn", tuneGrid = tunegrid_knn)
#model_knn1 <- train(as.factor(Class) ~ ., data=train, metric="ROC", trControl = train_control, method="knn",k= model_knn$bestTune$k)
#res_knn<- model_knn1$results
res_knn<- head(as.data.frame(model_knn1$results[order(model_knn1$results[,"ROC"], decreasing =  TRUE),]),1)
save(model_knn1, file= "KNN_model.RData")
#Test prediction
test_pred_knn <- predict(model_knn1, newdata=test[2:ncol(test)])
cm_knn <- confusionMatrix(as.factor(test_pred_knn), as.factor(test[,1]), mode='everything', positive=args[3])
cm_knn1 <- as.data.frame(as.matrix(cm_knn, what = "classes"))


#Stochastic Gradient Boosting 
model_stoc_grd1<- train(as.factor(Class) ~ ., data=train, metric="ROC", trControl = train_control, method="gbm", tuneGrid = tunegrid_gbm)
#model_stoc_grd1<- train(as.factor(Class) ~ ., data=train, metric="ROC", trControl = train_control, method="gbm",.n.trees=model_stoc_grd$bestTune$n.trees, .interaction.depth= model_stoc_grd$bestTune$interaction.depth, .shrinkage= model_stoc_grd$bestTune$shrinkage, .n.minobsinnode= model_stoc_grd$n.minobsinnode)
#res_stc_grd<- model_stoc_grd1$results
res_stc_grd<- head(as.data.frame(model_stoc_grd1$results[order(model_stoc_grd1$results[,"ROC"], decreasing =  TRUE),]),1)
save(model_stoc_grd1, file= "Stoc_Grad_model.RData")
#Test prediction
test_pred_stoch_grd <- predict(model_stoc_grd1, newdata=test[2:ncol(test)])
cm_stoch_grd  <- confusionMatrix(as.factor(test_pred_stoch_grd), as.factor(test[,1]), mode='everything', positive=args[3])
cm_stoch_grd1 <- as.data.frame(as.matrix(cm_stoch_grd , what = "classes"))

#C5 Decision Tree
model_DT<- train(as.factor(Class) ~ ., data=train, metric="ROC", trControl = train_control, method = "C5.0", tuneGrid = tunegrid_C5)
model_DT1<- train(as.factor(Class) ~ ., data=train, metric="ROC", trControl = train_control, method = "C5.0", .trials=model_DT$bestTune$trials, .model=model_DT$bestTune$model, .winnow=model_DT$bestTune$winnow)
#res_C5 <- model_DT1$results
res_C5 <- head(as.data.frame(model_DT1$results[order(model_DT1$results[,"ROC"], decreasing =  TRUE),]),1)
save(model_DT1, file= "C5_model.RData")
#Test prediction
test_pred_C5 <- predict(model_DT1, newdata=test[2:ncol(test)])
cm_C5 <- confusionMatrix(as.factor(test_pred_C5), as.factor(test[,1]), mode='everything', positive=args[3])
cm_C5_1 <- as.data.frame(as.matrix(cm_C5, what = "classes"))

#Linear Discriminant analysis
model_lda1<- train(as.factor(Class) ~ ., data=train, metric="ROC", trControl = train_control, method = "lda")
#res_lda <- model_lda1$results
res_lda <- head(as.data.frame(model_lda1$results[order(model_lda1$results[,"ROC"], decreasing =  TRUE),]),1)
save(model_lda1, file= "LDA_model.RData")
#Test prediction
test_pred_lda <- predict(model_lda1, newdata=test[2:ncol(test)])
cm_lda <- confusionMatrix(as.factor(test_pred_lda), as.factor(test[,1]), mode='everything', positive=args[3])
cm_lda1 <- as.data.frame(as.matrix(cm_lda, what = "classes"))


#Multilayer Perceptron
model_MLP_WD<- train(as.factor(Class) ~ ., data=train, metric="ROC", trControl = train_control, method = "mlpWeightDecay", tuneGrid = tunegrid_mlpwd)
model_MLP_WD1 <- train(as.factor(Class) ~ ., data=train, metric="ROC", trControl = train_control, method = "mlpWeightDecay",.size=model_MLP_WD$bestTune$size, .decay=model_MLP_WD$bestTune$decay)
#res_MLP <- model_MLP_WD1$results
res_MLP <- head(as.data.frame(model_MLP_WD1$results[order(model_MLP_WD1$results[,"ROC"], decreasing =  TRUE),]),1)
save(model_MLP_WD1, file= "MLP_model.RData")
#Test prediction
test_pred_MLP <- predict(model_MLP_WD1, newdata=test[2:ncol(test)])
cm_MLP <- confusionMatrix(as.factor(test_pred_MLP), as.factor(test[,1]), mode='everything', positive=args[3])
cm_MLP1 <- as.data.frame(as.matrix(cm_MLP, what = "classes"))


#Adjacent Categories Probability Model for Ordinal Data
model_vgam1<- train(as.factor(Class) ~ ., data=train, metric="ROC", trControl = train_control, method = "vglmAdjCat")
#model_vgam<- train(as.factor(Class) ~ ., data=train, metric="ROC", trControl = train_control, method = "vglmAdjCat", tuneGrid = tunegrid_vgam)
#model_vgam1 <- train(as.factor(Class) ~ ., data=train, metric="ROC", trControl = train_control, method = "vglmAdjCat", .parallel= model_vgam$bestTune$parallel, .link=model_vgam$bestTune$link) 
#res_vgam <- model_vgam1$results
res_vgam <- head(as.data.frame(model_vgam1$results[order(model_vgam1$results[,"ROC"], decreasing =  TRUE),]),1)
save(model_vgam1, file= "vgam_model.RData")
#Test prediction
test_pred_vgam <- predict(model_vgam1, newdata=test[2:ncol(test)])
cm_vgam <- confusionMatrix(as.factor(test_pred_vgam), as.factor(test[,1]), mode='everything', positive=args[3])
cm_vgam1 <- as.data.frame(as.matrix(cm_vgam, what = "classes"))


#xgbDART model
#model_xgbDART <- train(as.factor(Class) ~ ., data=train, metric="ROC", trControl = train_control, method="xgbDART", tuneGrid = tunegrid_xgbD)
#model_xgbDART1<- train(as.factor(Class) ~ ., data=train, metric="ROC", trControl = train_control, method="xgbDART", .nrounds=model_xgb$bestTune$nrounds, .max_depth= model_xgb$bestTune$max_depth, .eta=model_xgb$bestTune$eta, .gamma=model_xgb$bestTune$gamma, .colsample_bytree= model_xgb$bestTune$colsample_bytree, .min_child_weight= model_xgb$bestTune$min_child_weight, .subsample= model_xgb$bestTune$subsample)
model_xgbDART1<- train(as.factor(Class) ~ ., data=train, metric="ROC", trControl = train_control, method="xgbDART")
#res_xgbDART <- model_xgbDART1$results
res_xgbDART <- head(as.data.frame(model_xgbDART1$results[order(model_xgbDART1$results[,"ROC"], decreasing =  TRUE),]),1)
save(model_xgbDART1, file= "xgbDART_model.RData")
#Test prediction
test_pred_xgbDART <- predict(model_xgbDART1, newdata=test[2:ncol(test)])
cm_xgbDART <- confusionMatrix(as.factor(test_pred_xgbDART), as.factor(test[,1]), mode='everything', positive=args[3])
cm_xgbDART1 <- as.data.frame(as.matrix(cm_xgbDART, what = "classes"))

#xgbTree model
#model_xgbTree <- train(as.factor(Class) ~ ., data=train, metric="ROC", trControl = train_control, method="xgbTree", tuneGrid = tunegrid_xgbT)
#model_xgbTree1 <- train(as.factor(Class) ~ ., data=train, metric="ROC", trControl = train_control, method="xgbTree", .nrounds=model_xgb$bestTune$nrounds, .max_depth= model_xgb$bestTune$max_depth, .eta=model_xgb$bestTune$eta, .gamma=model_xgb$bestTune$gamma, .colsample_bytree= model_xgb$bestTune$colsample_bytree, .min_child_weight= model_xgb$bestTune$min_child_weight, .subsample= model_xgb$bestTune$subsample)
model_xgbTree1 <- train(as.factor(Class) ~ ., data=train, metric="ROC", trControl = train_control, method="xgbTree")
#res_xgbT<- model_xgbTree1$results
res_xgbT<- head(as.data.frame(model_xgbTree1$results[order(model_xgbTree1$results[,"ROC"], decreasing =  TRUE),]),1)
save(model_xgbTree1, file= "xgbTree_model.RData")
#Test prediction
test_pred_xgbTree <- predict(model_xgbTree1, newdata=test[2:ncol(test)])
cm_xgbTree <- confusionMatrix(as.factor(test_pred_xgbTree), as.factor(test[,1]), mode='everything', positive=args[3])
cm_xgbTree1 <- as.data.frame(as.matrix(cm_xgbTree, what = "classes"))


model_Nnet<- train(as.factor(Class) ~ ., data=train, metric="ROC", trControl = train_control, method = "nnet", tuneGrid = tunegrid_nnet)
model_Nnet1<- train(as.factor(Class) ~ ., data=train, metric="ROC", trControl = train_control, method = "nnet",.size= model_Nnet$bestTune$size, .decay= model_Nnet$bestTune$decay)
#res_nnet <- model_Nnet1$results
res_nnet <- head(as.data.frame(model_Nnet1$results[order(model_Nnet1$results[,"ROC"], decreasing =  TRUE),]),1)
save(model_Nnet1, file= "Nnet_model.RData")
#Test predictions
test_pred_nnet <- predict(model_Nnet1, newdata=test[2:ncol(test)])
cm_nnet <- confusionMatrix(as.factor(test_pred_nnet), as.factor(test[,1]), mode='everything', positive=args[3])
cm_nnet1 <- as.data.frame(as.matrix(cm_nnet, what = "classes"))


#Training Results
Final_all_models_tr_res <- bind_rows(res_rf, res_svr, res_svmL, res_glm, res_LRB, res_glmnet, res_knn, res_stc_grd, res_C5, res_lda, res_MLP, res_vgam, res_xgbDART, res_xgbT, res_nnet)

rownames(Final_all_models_tr_res) <- c("RF", "SVM-RBF", "SVM-L", "GLM", "Boosted_LR", "GlmNet", "KNN", "Stochastic_Gradient", "C5-DT", "LDA", "MLP", "VGAM", "xgbDART", "xgbTree", "Neural_NET" )

write.table(Final_all_models_tr_res,file=args[4],row.names=T,col.names=T,sep = '\t');

#Test results
Final_all_models_test_CM <- cbind(cm_rf1, cm_svr1, cm_svl1, cm_glm1, cm_LRB1, cm_glmnet1, cm_knn1, cm_stoch_grd1, cm_C5_1, cm_lda1, cm_MLP1, cm_vgam1, cm_xgbDART1, cm_xgbTree1, cm_nnet1)

colnames(Final_all_models_test_CM) <- c("RF", "SVM-RBF", "SVM-L", "GLM", "Boosted_LR", "GlmNet", "KNN", "Stochastic_Gradient", "C5-DT", "LDA", "MLP", "VGAM", "xgbDART", "xgbTree", "Neural_NET" )

write.table(Final_all_models_test_CM,file=args[5],row.names=T,col.names=T,sep = '\t');


#Test predictions
test_pred_all <- cbind(as.data.frame(test[1]),as.data.frame(test_pred_rf), as.data.frame(test_pred_svr), as.data.frame(test_pred_svl), as.data.frame(test_pred_glm), as.data.frame(test_pred_LRB), as.data.frame(test_pred_glmnet), as.data.frame(test_pred_knn), as.data.frame(test_pred_stoch_grd) , as.data.frame(test_pred_C5) , as.data.frame(test_pred_lda), as.data.frame(test_pred_MLP), as.data.frame(test_pred_vgam), as.data.frame(test_pred_xgbDART), as.data.frame(test_pred_xgbTree), as.data.frame(test_pred_nnet))

colnames(test_pred_all) <- c("Actual_class", "RF", "SVM-RBF", "SVM-L", "GLM", "Boosted_LR", "GlmNet", "KNN", "Stochastic_Gradient", "C5-DT", "LDA", "MLP", "VGAM", "xgbDART", "xgbTree", "Neural_NET" )

write.table(test_pred_all, file=args[6],row.names=T,col.names=T,sep = '\t')
