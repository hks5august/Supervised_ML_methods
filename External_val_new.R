#usage: nohup Rscript external_val_new.R sel_test  Recurrent  New_test_res New_test_pred New_test_prob_pred New_test_acc  &
# sel_test1  - Test data
# Recurrent - Positive class Label present in data
# New_test_res -  Test result output file name
# val_pred - trst prediction output file name


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

test <- read.table(args[1], header = TRUE, sep = "\t", row.names = 1, check.names = FALSE)


#RF model
model_rf1 <- get(load("RF_model.RData"))
print("RF model");
model_rf1
#Test prediction
test_pred_rf <- predict(model_rf1, newdata=test[2:ncol(test)])
test_pred_rf
test_pred_rf1 <- predict(model_rf1, newdata=test[2:ncol(test)], type = "prob")
cm_rf <- confusionMatrix(as.factor(test_pred_rf), as.factor(test[,1]), mode='everything', positive=args[2])
cm_rf1 <- as.data.frame(as.matrix(cm_rf, what = "classes"))
print("RF Done");

#SVM-RBF model
model_svmR1  <- get(load("SVR_model.RData"))
print("svmR model");
#Test prediction
test_pred_svr <- predict(model_svmR1, newdata=test[2:ncol(test)])
test_pred_svr1 <- predict(model_svmR1, newdata=test[2:ncol(test)], type = "prob")
cm_svr <- confusionMatrix(as.factor(test_pred_svr), as.factor(test[,1]), mode='everything', positive=args[2])
cm_svr1 <- as.data.frame(as.matrix(cm_svr, what = "classes"))
print("svmR model Done")


#svm linear model
model_svmL1  <- get(load("svmL_model.RData"))
#Test prediction
test_pred_svl <- predict(model_svmL1, newdata=test[2:ncol(test)])
test_pred_svl1 <- predict(model_svmL1, newdata=test[2:ncol(test)], type = "prob")
cm_svl <- confusionMatrix(as.factor(test_pred_svl), as.factor(test[,1]), mode='everything', positive=args[2])
cm_svl1 <- as.data.frame(as.matrix(cm_svl, what = "classes"))
print("svmL model Done")

#LR model
model_glm1  <- get(load("Glm_model.RData"))
#Test prediction
test_pred_glm <- predict(model_glm1, newdata=test[2:ncol(test)])
test_pred_glm1 <- predict(model_glm1, newdata=test[2:ncol(test)], type = "prob")
cm_glm <- confusionMatrix(as.factor(test_pred_glm), as.factor(test[,1]), mode='everything', positive=args[2])
cm_glm1 <- as.data.frame(as.matrix(cm_glm, what = "classes"))
print("LR model Done")

#boosted LR model
model_LR_B1  <- get(load( "LR_B_model.RData"))

#Test prediction
test_pred_LRB <- predict(model_LR_B1, newdata=test[2:ncol(test)])
test_pred_LRB1 <- predict(model_LR_B1, newdata=test[2:ncol(test)], type = "prob")
cm_LRB <- confusionMatrix(as.factor(test_pred_LRB), as.factor(test[,1]), mode='everything', positive=args[2])
cm_LRB1 <- as.data.frame(as.matrix(cm_LRB, what = "classes"))
print(" Boosted LR model Done")


#glmNet model
model_glmnet1  <- get(load("glmnet_model.RData"))
#Test prediction
test_pred_glmnet <- predict(model_glmnet1, newdata=test[2:ncol(test)])
test_pred_glmnet1 <- predict(model_glmnet1, newdata=test[2:ncol(test)],type = "prob")
cm_glmnet <- confusionMatrix(as.factor(test_pred_glmnet), as.factor(test[,1]), mode='everything', positive=args[2])
cm_glmnet1 <- as.data.frame(as.matrix(cm_glmnet, what = "classes"))
print("glmnet model Done")

#knn model
model_knn1  <- get(load("KNN_model.RData"))
#Test prediction
test_pred_knn <- predict(model_knn1, newdata=test[2:ncol(test)])
test_pred_knn1 <- predict(model_knn1, newdata=test[2:ncol(test)], type = "prob")
cm_knn <- confusionMatrix(as.factor(test_pred_knn), as.factor(test[,1]), mode='everything', positive=args[2])
cm_knn1 <- as.data.frame(as.matrix(cm_knn, what = "classes"))
print("knn model Done")

#Stochastic Gradient Boosting
#model_stoc_grd1  <- get(load("Stoc_Grad_model.RData"))
#Test prediction
#test_pred_stoch_grd <- predict(model_stoc_grd1, newdata=test[2:ncol(test)])
#test_pred_stoch_grd1 <- predict(model_stoc_grd1, newdata=test[2:ncol(test)], type = "prob")
#cm_stoch_grd  <- confusionMatrix(as.factor(test_pred_stoch_grd), as.factor(test[,1]), mode='everything', positive=args[2])
#cm_stoch_grd1 <- as.data.frame(as.matrix(cm_stoch_grd , what = "classes"))
#print("Stoch Grad model Done")

#C5 DT model
model_DT1  <- get(load("C5_model.RData"))
#Test prediction
test_pred_C5 <- predict(model_DT1, newdata=test[2:ncol(test)])
test_pred_C5_1 <- predict(model_DT1, newdata=test[2:ncol(test)], type = "prob")
cm_C5 <- confusionMatrix(as.factor(test_pred_C5), as.factor(test[,1]), mode='everything', positive=args[2])
cm_C5_1 <- as.data.frame(as.matrix(cm_C5, what = "classes"))
print("C5-DT model Done")

#LDA model
model_lda1  <- get(load("LDA_model.RData"))
#Test prediction
test_pred_lda <- predict(model_lda1, newdata=test[2:ncol(test)])
test_pred_lda1 <- predict(model_lda1, newdata=test[2:ncol(test)],  type = "prob")
cm_lda <- confusionMatrix(as.factor(test_pred_lda), as.factor(test[,1]), mode='everything', positive=args[2])
cm_lda1 <- as.data.frame(as.matrix(cm_lda, what = "classes"))
print("LDA model Done")

#MLP model
model_MLP_WD1  <- get(load("MLP_model.RData"))
#Test prediction
test_pred_MLP <- predict(model_MLP_WD1, newdata=test[2:ncol(test)])
test_pred_MLP1 <- predict(model_MLP_WD1, newdata=test[2:ncol(test)], type = "prob")
cm_MLP <- confusionMatrix(as.factor(test_pred_MLP), as.factor(test[,1]), mode='everything', positive=args[2])
cm_MLP1 <- as.data.frame(as.matrix(cm_MLP, what = "classes"))
print("MLP model Done")

#Adjacent Categories Probability Model for Ordinal Data
model_vgam1  <- get(load("vgam_model.RData"))
#Test prediction
test_pred_vgam <- predict(model_vgam1, newdata=test[2:ncol(test)])
test_pred_vgam1 <- predict(model_vgam1, newdata=test[2:ncol(test)], type = "prob")
cm_vgam <- confusionMatrix(as.factor(test_pred_vgam), as.factor(test[,1]), mode='everything', positive=args[2])
cm_vgam1 <- as.data.frame(as.matrix(cm_vgam, what = "classes"))
print("VGAM model Done")


#xgbDART model
model_xgbDART1  <- get(load("xgbDART_model.RData"))
#Test prediction
test_pred_xgbDART <- predict(model_xgbDART1, newdata=test[2:ncol(test)])
test_pred_xgbDART1 <- predict(model_xgbDART1, newdata=test[2:ncol(test)], type = "prob")
cm_xgbDART <- confusionMatrix(as.factor(test_pred_xgbDART), as.factor(test[,1]), mode='everything', positive=args[2])
cm_xgbDART1 <- as.data.frame(as.matrix(cm_xgbDART, what = "classes"))
print("xgbDART model Done")


#xgbTree model
model_xgbTree1  <- get(load("xgbTree_model.RData"))
#Test prediction
test_pred_xgbTree <- predict(model_xgbTree1, newdata=test[2:ncol(test)])
test_pred_xgbTree1 <- predict(model_xgbTree1, newdata=test[2:ncol(test)],type = "prob")
cm_xgbTree <- confusionMatrix(as.factor(test_pred_xgbTree), as.factor(test[,1]), mode='everything', positive=args[2])
cm_xgbTree1 <- as.data.frame(as.matrix(cm_xgbTree, what = "classes"))
print("xgbTree model Done")


#NNET MODEL
model_Nnet1  <- get(load("Nnet_model.RData"))
#Test predictions
test_pred_nnet <- predict(model_Nnet1, newdata=test[2:ncol(test)])
test_pred_nnet1 <- predict(model_Nnet1, newdata=test[2:ncol(test)], type = "prob")
cm_nnet <- confusionMatrix(as.factor(test_pred_nnet), as.factor(test[,1]), mode='everything', positive=args[2])
cm_nnet1 <- as.data.frame(as.matrix(cm_nnet, what = "classes"))
print("NNet model Done")


#Test results
#Final_all_models_test_CM <- cbind(cm_rf1, cm_svr1, cm_svl1, cm_glm1, cm_LRB1, cm_glmnet1, cm_knn1, cm_stoch_grd1, cm_C5_1, cm_lda1, cm_MLP1, cm_vgam1, cm_xgbDART1, cm_xgbTree1, cm_nnet1)
Final_all_models_test_CM <- cbind(cm_rf1, cm_svr1, cm_svl1, cm_glm1, cm_LRB1, cm_glmnet1, cm_knn1, cm_C5_1, cm_lda1, cm_MLP1, cm_vgam1, cm_xgbDART1, cm_xgbTree1, cm_nnet1)


#colnames(Final_all_models_test_CM) <- c("RF", "SVM-RBF", "SVM-L", "GLM", "Boosted_LR", "GlmNet", "KNN", "Stochastic_Gradient", "C5-DT", "LDA", "MLP", "VGAM", "xgbDART", "xgbTree", "Neural_NET" )

colnames(Final_all_models_test_CM) <- c("RF", "SVM-RBF", "SVM-L", "GLM", "Boosted_LR", "GlmNet", "KNN", "C5-DT", "LDA", "MLP", "VGAM", "xgbDART", "xgbTree", "Neural_NET" )
write.table(Final_all_models_test_CM,file=args[3],row.names=T,col.names=T,sep = '\t');


#Test predictions
#test_pred_all <- cbind(as.data.frame(test[1]),as.data.frame(test_pred_rf), as.data.frame(test_pred_svr), as.data.frame(test_pred_svl), as.data.frame(test_pred_glm), as.data.frame(test_pred_LRB), as.data.frame(test_pred_glmnet), as.data.frame(test_pred_knn), as.data.frame(test_pred_stoch_grd) , as.data.frame(test_pred_C5) , as.data.frame(test_pred_lda), as.data.frame(test_pred_MLP), as.data.frame(test_pred_vgam), as.data.frame(test_pred_xgbDART), as.data.frame(test_pred_xgbTree), as.data.frame(test_pred_nnet))

test_pred_all <- cbind(as.data.frame(test[1]),as.data.frame(test_pred_rf), as.data.frame(test_pred_svr), as.data.frame(test_pred_svl), as.data.frame(test_pred_glm), as.data.frame(test_pred_LRB), as.data.frame(test_pred_glmnet), as.data.frame(test_pred_knn) , as.data.frame(test_pred_C5) , as.data.frame(test_pred_lda), as.data.frame(test_pred_MLP), as.data.frame(test_pred_vgam), as.data.frame(test_pred_xgbDART), as.data.frame(test_pred_xgbTree), as.data.frame(test_pred_nnet))

#colnames(test_pred_all) <- c("Actual_class", "RF", "SVM-RBF", "SVM-L", "GLM", "Boosted_LR", "GlmNet", "KNN", "Stochastic_Gradient", "C5-DT", "LDA", "MLP", "VGAM", "xgbDART", "xgbTree", "Neural_NET" )


colnames(test_pred_all) <- c("Actual_class", "RF", "SVM-RBF", "SVM-L", "GLM", "Boosted_LR", "GlmNet", "KNN","C5-DT", "LDA", "MLP", "VGAM", "xgbDART", "xgbTree", "Neural_NET" )
write.table(test_pred_all, file=args[4],row.names=T,col.names=T,sep = '\t')


#Test predictions prob
#test_pred_all1 <- cbind(as.data.frame(test[1]),as.data.frame(test_pred_rf), as.data.frame(test_pred_svr), as.data.frame(test_pred_svl), as.data.frame(test_pred_glm), as.data.frame(test_pred_LRB), as.data.frame(test_pred_glmnet), as.data.frame(test_pred_knn), as.data.frame(test_pred_stoch_grd) , as.data.frame(test_pred_C5) , as.data.frame(test_pred_lda), as.data.frame(test_pred_MLP), as.data.frame(test_pred_vgam), as.data.frame(test_pred_xgbDART), as.data.frame(test_pred_xgbTree), as.data.frame(test_pred_nnet))

test_pred_all1 <- cbind(as.data.frame(test[1]),as.data.frame(test_pred_rf1), as.data.frame(test_pred_svr1), as.data.frame(test_pred_svl1), as.data.frame(test_pred_glm1), as.data.frame(test_pred_LRB1), as.data.frame(test_pred_glmnet1), as.data.frame(test_pred_knn1) , as.data.frame(test_pred_C5_1) , as.data.frame(test_pred_lda1), as.data.frame(test_pred_MLP1), as.data.frame(test_pred_vgam1), as.data.frame(test_pred_xgbDART1), as.data.frame(test_pred_xgbTree1), as.data.frame(test_pred_nnet1))

#colnames(test_pred_all) <- c("Actual_class", "RF", "SVM-RBF", "SVM-L", "GLM", "Boosted_LR", "GlmNet", "KNN", "Stochastic_Gradient", "C5-DT", "LDA", "MLP", "VGAM", "xgbDART", "xgbTree", "Neural_NET" )


colnames(test_pred_all1) <- c("Actual_class", "RF", "SVM-RBF", "SVM-L", "GLM", "Boosted_LR", "GlmNet", "KNN","C5-DT", "LDA", "MLP", "VGAM", "xgbDART", "xgbTree", "Neural_NET" )
write.table(test_pred_all, file=args[5],row.names=T,col.names=T,sep = '\t')

#Test i accuracy results
Final_all_models_test_accuracy <- cbind(cm_rf$overall[1], cm_svr$overall[1], cm_svl$overall[1], cm_glm$overall[1], cm_LRB$overall[1], cm_glmnet$overall[1], cm_knn$overall[1], cm_C5$overall[1], cm_lda$overall[1], cm_MLP$overall[1], cm_vgam$overall[1], cm_xgbDART$overall[1], cm_xgbTree$overall[1], cm_nnet$overall[1])
#Final_all_models_test_CM <- cbind(cm_rf1, cm_svr1, cm_svl1, cm_glm1, cm_LRB1, cm_glmnet1, cm_knn1, cm_C5_1, cm_lda1, cm_MLP1, cm_vgam1, cm_xgbDART1, cm_xgbTree1, cm_nnet1)


#colnames(Final_all_models_test_CM) <- c("RF", "SVM-RBF", "SVM-L", "GLM", "Boosted_LR", "GlmNet", "KNN", "Stochastic_Gradient", "C5-DT", "LDA", "MLP", "VGAM", "xgbDART", "xgbTree", "Neural_NET" )

colnames(Final_all_models_test_accuracy) <- c("RF", "SVM-RBF", "SVM-L", "GLM", "Boosted_LR", "GlmNet", "KNN", "C5-DT", "LDA", "MLP", "VGAM", "xgbDART", "xgbTree", "Neural_NET" )
write.table(Final_all_models_test_accuracy,file=args[6],row.names=T,col.names=T,sep = '\t');


