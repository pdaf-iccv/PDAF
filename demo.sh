SOURCE=cityscapes
TARGET=bdd100k
ROOT='./checkpoints'
HYP="${ROOT}/deeplab_res50.yaml"
OUTPUT_DIR="outs"

LPE_WEIGHT="${ROOT}/best-lpe-${TARGET}.pt"
DCM_WEIGHT="${ROOT}/best-dcm-${TARGET}.pt"
DPE_WEIGHT="${ROOT}/best-dpe-${TARGET}.pt"
accelerate launch --gpu_ids '1' --main_process_port 29500 run.py \
        --num_train_epochs 300 \
        --train_batch_size 4 \
        --val_batch_size 4 \
        --config_yaml $HYP \
        --output_dir $OUTPUT_DIR \
        --dataset $SOURCE \
        --val_dataset $TARGET \
        --lpe_weight $LPE_WEIGHT \
        --dcm_weight $DCM_WEIGHT \
        --dpe_weight $DPE_WEIGHT \
        --test_only
