      
#!/bin/bash

# ---------------------------
# Experiments
# ---------------------------

BASE="../"
TEMPLATE_YAML="${BASE}/LLaMA-Factory/examples/train_lora/llama3_lora_sft_ds3.yaml"

bash /mnt/public/wangshaobo/conda_init.sh
export CONDA_ENVS_DIRS=/mnt/public/gpfs-jd/data/wangshaobo/conda_envs/:$CONDA_ENVS_DIRS

CHECKPOINTS_DIR="${BASE}/checkpoints"
SAVE_DIR="${BASE}/saves"
DATASET="alpaca"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_WORKER_MULTIPROC_METHOD=spawn
NPROC_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
if [ "$NPROC_PER_NODE" -ge 8 ]; then
    HF_NUM_GPUS=8
elif [ "$NPROC_PER_NODE" -ge 4 ]; then
    HF_NUM_GPUS=4
elif [ "$NPROC_PER_NODE" -ge 2 ]; then
    HF_NUM_GPUS=2
else
    HF_NUM_GPUS=1
fi

EXPERIMENTS=(
    # "random random 100 100 Llama-3.2-3B Llama-3.2-3B 4 3 5e-5"
    # "random random 100 100 Meta-Llama-3-8B Llama-3.2-3B 4 3 5e-5"

    "random random 10 10 Meta-Llama-3-8B Llama-3.2-3B 4 3 5e-5"
    "random random 10 30 Meta-Llama-3-8B Llama-3.2-3B 4 3 5e-5"
    "random random 10 50 Meta-Llama-3-8B Llama-3.2-3B 4 3 5e-5"
    # "random random 10 100 Meta-Llama-3-8B Llama-3.2-3B 4 3 5e-5"
    "random random 30 10 Meta-Llama-3-8B Llama-3.2-3B 4 3 5e-5"
    "random random 30 30 Meta-Llama-3-8B Llama-3.2-3B 4 3 5e-5"
    "random random 30 50 Meta-Llama-3-8B Llama-3.2-3B 4 3 5e-5"
    "random random 30 100 Meta-Llama-3-8B Llama-3.2-3B 4 3 5e-5"
    "random random 50 10 Meta-Llama-3-8B Llama-3.2-3B 4 3 5e-5"
    "random random 50 30 Meta-Llama-3-8B Llama-3.2-3B 4 3 5e-5"
    "random random 50 50 Meta-Llama-3-8B Llama-3.2-3B 4 3 5e-5"
    # "random random 50 100 Meta-Llama-3-8B Llama-3.2-3B 4 3 5e-5"
    "random random 100 10 Meta-Llama-3-8B Llama-3.2-3B 4 3 5e-5"
    "random random 100 30 Meta-Llama-3-8B Llama-3.2-3B 4 3 5e-5"
    # "random random 100 50 Meta-Llama-3-8B Llama-3.2-3B 4 3 5e-5"
    # "random random 100 100 Meta-Llama-3-8B Llama-3.2-3B 4 3 5e-5"

    "infobatch random 10 10 Meta-Llama-3-8B Llama-3.2-3B 4 3 5e-5"
    "infobatch random 10 30 Meta-Llama-3-8B Llama-3.2-3B 4 3 5e-5"
    "infobatch random 10 50 Meta-Llama-3-8B Llama-3.2-3B 4 3 5e-5"
    "infobatch random 10 100 Meta-Llama-3-8B Llama-3.2-3B 4 3 5e-5"
    "infobatch random 30 10 Meta-Llama-3-8B Llama-3.2-3B 4 3 5e-5"
    "infobatch random 30 30 Meta-Llama-3-8B Llama-3.2-3B 4 3 5e-5"
    "infobatch random 30 50 Meta-Llama-3-8B Llama-3.2-3B 4 3 5e-5"
    "infobatch random 30 100 Meta-Llama-3-8B Llama-3.2-3B 4 3 5e-5"
    "infobatch random 50 10 Meta-Llama-3-8B Llama-3.2-3B 4 3 5e-5"
    "infobatch random 50 30 Meta-Llama-3-8B Llama-3.2-3B 4 3 5e-5"
    "infobatch random 50 50 Meta-Llama-3-8B Llama-3.2-3B 4 3 5e-5"
    # "infobatch random 50 100 Meta-Llama-3-8B Llama-3.2-3B 4 3 5e-5"
    # "infobatch random 100 10 Meta-Llama-3-8B Llama-3.2-3B 4 3 5e-5"
    # "infobatch random 100 30 Meta-Llama-3-8B Llama-3.2-3B 4 3 5e-5"
    # "infobatch random 100 50 Meta-Llama-3-8B Llama-3.2-3B 4 3 5e-5"
    # "infobatch random 100 100 Meta-Llama-3-8B Llama-3.2-3B 4 3 5e-5"

    "random rho 10 10 Meta-Llama-3-8B Llama-3.2-3B 4 3 5e-5"
    "random rho 10 30 Meta-Llama-3-8B Llama-3.2-3B 4 3 5e-5"
    "random rho 10 50 Meta-Llama-3-8B Llama-3.2-3B 4 3 5e-5"
    # "random rho 10 100 Meta-Llama-3-8B Llama-3.2-3B 4 3 5e-5"
    "random rho 30 10 Meta-Llama-3-8B Llama-3.2-3B 4 3 5e-5"
    "random rho 30 30 Meta-Llama-3-8B Llama-3.2-3B 4 3 5e-5"
    "random rho 30 50 Meta-Llama-3-8B Llama-3.2-3B 4 3 5e-5"
    # "random rho 30 100 Meta-Llama-3-8B Llama-3.2-3B 4 3 5e-5"
    "random rho 50 10 Meta-Llama-3-8B Llama-3.2-3B 4 3 5e-5"
    "random rho 50 30 Meta-Llama-3-8B Llama-3.2-3B 4 3 5e-5"
    "random rho 50 50 Meta-Llama-3-8B Llama-3.2-3B 4 3 5e-5"
    # "random rho 50 100 Meta-Llama-3-8B Llama-3.2-3B 4 3 5e-5"
    "random rho 100 10 Meta-Llama-3-8B Llama-3.2-3B 4 3 5e-5"
    "random rho 100 30 Meta-Llama-3-8B Llama-3.2-3B 4 3 5e-5"
    # "random rho 100 50 Meta-Llama-3-8B Llama-3.2-3B 4 3 5e-5"
    # "random rho 100 100 Meta-Llama-3-8B Llama-3.2-3B 4 3 5e-5"

    "infobatch rho 10 10 Meta-Llama-3-8B Llama-3.2-3B 4 3 5e-5"
    "infobatch rho 10 30 Meta-Llama-3-8B Llama-3.2-3B 4 3 5e-5"
    "infobatch rho 10 50 Meta-Llama-3-8B Llama-3.2-3B 4 3 5e-5"
    # "infobatch rho 10 100 Meta-Llama-3-8B Llama-3.2-3B 4 3 5e-5"
    "infobatch rho 30 10 Meta-Llama-3-8B Llama-3.2-3B 4 3 5e-5"
    "infobatch rho 30 30 Meta-Llama-3-8B Llama-3.2-3B 4 3 5e-5"
    "infobatch rho 30 50 Meta-Llama-3-8B Llama-3.2-3B 4 3 5e-5"
    # "infobatch rho 30 100 Meta-Llama-3-8B Llama-3.2-3B 4 3 5e-5"
    "infobatch rho 50 10 Meta-Llama-3-8B Llama-3.2-3B 4 3 5e-5"
    "infobatch rho 50 30 Meta-Llama-3-8B Llama-3.2-3B 4 3 5e-5"
    "infobatch rho 50 50 Meta-Llama-3-8B Llama-3.2-3B 4 3 5e-5"
    # "infobatch rho 50 100 Meta-Llama-3-8B Llama-3.2-3B 4 3 5e-5"
    # "infobatch rho 100 10 Meta-Llama-3-8B Llama-3.2-3B 4 3 5e-5"
    # "infobatch rho 100 30 Meta-Llama-3-8B Llama-3.2-3B 4 3 5e-5"
    # "infobatch rho 100 50 Meta-Llama-3-8B Llama-3.2-3B 4 3 5e-5"
    # "infobatch rho 100 100 Meta-Llama-3-8B Llama-3.2-3B 4 3 5e-5"
)

for EXPERIMENT in "${EXPERIMENTS[@]}"; do
    conda activate llama_factory
    read -r DATA_METHOD TOKEN_METHOD DATA_RATIO TOKEN_RATIO MODEL REFERENCE_MODEL BATCH_SIZE EPOCH LR <<< "$EXPERIMENT"

    OUTPUT_DIR="${SAVE_DIR}/${DATASET}/${MODEL}/data_ratio=${DATA_RATIO}/token_ratio=${TOKEN_RATIO}/${DATA_METHOD}_${TOKEN_METHOD}_${LR}"
    MERGE_DIR="${CHECKPOINTS_DIR}/${DATASET}/${MODEL}/data_ratio=${DATA_RATIO}/token_ratio=${TOKEN_RATIO}/${DATA_METHOD}_${TOKEN_METHOD}_${LR}"
    WORK_DIR="${MERGE_DIR}/results"
    YAML_TMP="temp_${DATASET}_${DATA_METHOD}_${TOKEN_METHOD}_${DATA_RATIO}_${BATCH_SIZE}_${EPOCH}_${LR}.yaml"
    
    sed -e "s|dataset: .*|dataset: ${DATASET}|" \
            -e "s|output_dir: .*|output_dir: ${OUTPUT_DIR}|" \
            -e "s|per_device_train_batch_size: .*|per_device_train_batch_size: ${BATCH_SIZE}|" \
            -e "s|num_train_epochs: .*|num_train_epochs: ${EPOCH}|" \
            -e "s|learning_rate: .*|learning_rate: ${LR}|" \
            -e "s|data_method: .*|data_method: ${DATA_METHOD}|" \
            -e "s|data_ratio: .*|data_ratio: $(awk "BEGIN {print $DATA_RATIO / 100}")|" \
            -e "s|token_method: .*|token_method: ${TOKEN_METHOD}|" \
            -e "s|token_ratio: .*|token_ratio: $(awk "BEGIN {print $TOKEN_RATIO / 100}")|" \
            $TEMPLATE_YAML > $YAML_TMP
    llamafactory-cli train $YAML_TMP

    cat <<EOF > merge_config.yaml
model_name_or_path: $(grep "^model_name_or_path:" "$TEMPLATE_YAML" | cut -d':' -f2- | xargs)
adapter_name_or_path: $OUTPUT_DIR
template: llama3
finetuning_type: lora

export_dir: $MERGE_DIR
export_size: 2
export_device: cpu
export_legacy_format: false
EOF
    llamafactory-cli export merge_config.yaml

    # MERGE_DIR="${BASE}/base/Meta-Llama-3-8B"
    # WORK_DIR="${MERGE_DIR}/results"

    conda activate opencompass
    # PPL_DATASETS=("ARC_c_ppl" "hellaswag_ppl" "mmlu_ppl" "race_ppl" "winogrande_ppl_55a66e")
    PPL_DATASETS=("ARC_c_ppl" "race_ppl")
    for EVAL_DATASET in "${PPL_DATASETS[@]}"; do
        python ${BASE}/opencompass/run.py \
        --datasets $EVAL_DATASET \
        --hf-type base \
        --hf-path ${MERGE_DIR} \
        --work-dir ${WORK_DIR}/$EVAL_DATASET \
        --debug 
    done
    # GEN_DATASETS=("bbh_gen" "drop_gen_a2697c" "gsm8k_gen" "humaneval_gen" "mbpp_gen" "squad20_gen" "SuperGLUE_BoolQ_gen" "triviaqa_gen")
    GEN_DATASETS=("gsm8k_gen" "humaneval_gen" "squad20_gen" "SuperGLUE_BoolQ_gen" "triviaqa_gen")
    for EVAL_DATASET in "${GEN_DATASETS[@]}"; do
        python ${BASE}/opencompass/run.py \
        --datasets $EVAL_DATASET \
        --hf-num-gpus $HF_NUM_GPUS \
        --hf-type base \
        --hf-path ${MERGE_DIR} \
        --work-dir ${WORK_DIR}/$EVAL_DATASET \
        --debug \
        --model-kwargs gpu_memory_utilization=0.8 \
        --accelerator vllm
    done
    
    # EVAL_DATASETS=("ARC_c_ppl" "bbh_gen" "drop_gen_a2697c" "gsm8k_gen" "hellaswag_ppl" "humaneval_gen" "mbpp_gen" "mmlu_ppl" "race_ppl" "squad20_gen" "SuperGLUE_BoolQ_ppl" "triviaqa_gen" "winogrande_ppl_55a66e")
    # EVAL_DATASETS=("SuperGLUE_BoolQ_gen")
    # for EVAL_DATASET in "${EVAL_DATASETS[@]}"; do
    #     python ${BASE}/opencompass/run.py \
    #     --datasets $EVAL_DATASET \
    #     --hf-type base \
    #     --hf-path ${MERGE_DIR} \
    #     --work-dir ${WORK_DIR}/$EVAL_DATASET \
    #     --debug \
    #     --model-kwargs gpu_memory_utilization=0.8 \
    #     --accelerator vllm
    # done
    rm -f $YAML_TMP
done

    