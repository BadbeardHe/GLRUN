version="debug"

python -u train.py \
    -lr 2.5e-3 \
    --weight_decay 1e-5 \
    -in '/home/canyon/Documents/DepthCompletion/datasets/FLAT' \
    -out "./models/$version" \
    -d "./results/$version/train" \
    -e 200