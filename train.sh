for EPOCHS in 10 5; do
  for BATCH_SIZE in 64 32 16; do
    for LAYERS in 4 2; do
      for HEADS in 8 4 2; do
        for LR in 1e-5; do

          CONFIG="ep${EPOCHS}-bs${BATCH_SIZE}-ly${LAYERS}-hd${HEADS}-lr${LR}"
          mkdir -p "./result/${CONFIG}"

          python train.py \
            --save_path="./result/${CONFIG}" \
            --n_layers=${LAYERS} \
            --n_heads=${HEADS} \
            --lr=${LR} \
            --epochs=${EPOCHS} \
            --batch_size=${BATCH_SIZE}

        done
      done
    done
  done
done
