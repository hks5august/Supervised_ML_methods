
nohup Rscript All_ML_Model.R sel_train1 sel_test1  Recurrent Train_out_res test_out_res test_predictions &

nohup Rscript external_val_new.R sel_test1  Recurrent  New_test1_res New_test1_pred New_test1_prob_pred New_test1_acc  &

nohup Rscript external_val_new.R sel_test2  Recurrent  New_test2_res New_test2_pred New_test2_prob_pred New_test2_acc  &

nohup Rscript ROC_Value.R New_test1_pred test1_roc &
mv Rplots.pdf test1_ROC_plot.pdf

nohup Rscript ROC_Value.R New_test2_pred test2_roc &
 mv Rplots.pdf test2_ROC_plot.pdf

