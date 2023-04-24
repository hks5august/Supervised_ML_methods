#usage:nohup Rscript ROC_Value.R  input_file output &
#input file: prediction file, where 1st column containing samples IDs, 2nd column actual label and 3rd column onward contain predicted labels
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
library(pROC)
library(MASS);
args <- commandArgs(TRUE)
set.seed(7)

test1_pred <- read.table(args[1], header = TRUE, sep = "\t", row.names = 1, check.names = FALSE)
#test2_pred <- read.table("test2_pred", header = TRUE, sep = "\t", row.names = 1, check.names = FALSE)

#out <- args[1]

roc_data <- test1_pred
E= length(roc_data )
for(i in seq(from=2, to=E ,by=1))
{
  #roc1 = roc(as.factor(test1_pred[,1]), as.factor(test1_pred[,i]), main = paste0("ROC Plot for" + colnames(roc_data[i)))
  #ROC curve
   roc<- roc.curve(as.factor(roc_data[,1]), as.factor(roc_data[,i]), plotit = T,  main = paste0("ROC Plot: ", colnames(roc_data[i])))
   roc_val <- round(roc$auc,2)
   #print(roc_val )
   write.table(cbind(colnames(roc_data[i]),roc_val), file= args[2], row.names=F,col.names=F,sep = '\t',append = T)
}

