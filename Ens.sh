while read line
do


cd $line
cp ~/scripts/Final_Ensemble_model.R .

nohup Rscript Final_Ensemble_model.R sel_train sel_test sel_ext Ens_Train_result Recurrent Ensem_test1_res Ensem_test2_res Ensem_test1_pred Ensem_test2_pred &
wait


nohup Rscript ROC_Value.R  Ensem_test1_pred Ens_test1_roc &

wait

mv Rplots.pdf Ensem_test1_ROC_plot.pdf

wait 
nohup Rscript ROC_Value.R  Ensem_test2_pred Ens_test2_roc &

wait

mv Rplots.pdf Ensem_test2_ROC_plot.pdf

cd ../
done<list

