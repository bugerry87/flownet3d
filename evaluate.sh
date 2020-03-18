


python3 evaluate.py \
    --model model_concat_upsa \
    --data data/kitti_rm_ground \
    --log_dir log_evaluate \
    --model_path model/model.ckpt \
    --num_point 2048 \
    --batch_size 16 \
    --shuffle $1 \
    --randomize 0
