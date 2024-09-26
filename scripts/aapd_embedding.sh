python source/train_embedding.py \
    --output_dir embeddings \
    --dataset AAPD \
    --label_name first \
    --strategy single \
    --stage 1 \
    --margin 0.5 \
    --num_epochs 2 \
    --batch_size 64 \
    --exclude_duplicate_negative