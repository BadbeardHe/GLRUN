version="debug"
modelpath="models/$version/checkpoint_best.pth"
iq_out="results/$version/iq"
mu_out="results/$version/mu"
d_out="results/$version/depth"
metric_save="results/result_metrics/$version"

python predict.py \
    -in "/home/canyon/Documents/DepthCompletion/datasets/FLAT/noise" \
    -ls "/home/canyon/Documents/DepthCompletion/datasets/FLAT/list/test.txt" \
    -out $iq_out \
    -out_mu $mu_out \
    -out_d $d_out \
    -m $modelpath \

python eval.py \
   -in $d_out \
   -gt "/home/canyon/Documents/DepthCompletion/datasets/FLAT/ideal_depth" \
   -out $metric_save \
   -v "1.0"
