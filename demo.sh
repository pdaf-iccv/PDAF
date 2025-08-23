SOURCE=cityscapes
TARGETS=(bdd100k)
ROOT='/mnt/187/b/edd/workspace/RobustNet/exclude/all_runs/runs_iccv/ablation/city/stage-73_6'
HYP="${ROOT}/deeplab_res50.yaml"
OUTPUT_DIR="outs"



for TARGET in "${TARGETS[@]}"
do
        CPEN_WEIGHT="${ROOT}/best-cpen_s1-${TARGET}.pt"
        TASKMODEL_WEIGHT="${ROOT}/best-task_model-${TARGET}.pt"
        DIFFUSION_WEIGHT="${ROOT}/best-diffusion-${TARGET}.pt"
        accelerate launch --gpu_ids '1' --main_process_port 29500 run.py \
                --num_train_epochs 300 \
                --train_batch_size 4 \
                --val_batch_size 4 \
                --config_yaml $HYP \
                --output_dir $OUTPUT_DIR \
                --dataset $SOURCE \
                --val_dataset $TARGET \
                --lpe_weight $CPEN_WEIGHT \
                --dcm_weight $TASKMODEL_WEIGHT \
                --dpe_weight $DIFFUSION_WEIGHT \
                --test_only --scales 1.0 
done