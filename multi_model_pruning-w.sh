#!/bin/bash

# ---------------------------
# Multi-Model Data Token Pruning Experiments
# Models: llama2-7b, qwen2-7b, mistral-7b
# Datasets: limr, wizard
# 
# 使用 wise 插件参数：
# 在 EXPERIMENTS 数组中为每个实验指定 wise_lambda 值
# 格式：DATA_METHOD TOKEN_METHOD DATA_RATIO TOKEN_RATIO MODEL BATCH_SIZE EPOCH LR PLUG WISE_LAMBDA
# 
# 参数说明：
# - WISE_LAMBDA: wise token scoring 中邻居 ppl 的权重 (推荐 0.5)
# - 当 WISE_LAMBDA != 0.5 时，会自动添加到模型命名后缀中，如：_wisely_0.3
# ---------------------------

cd /mnt/public/gpfs-jd/code/wangshaobo/Data_Token_Pruning/LLaMA-Factory
export COMPASS_DATA_CACHE="/mnt/public/gpfs-jd/code/wangshaobo/Data_Token_Pruning/opencompass"
BASE="/mnt/public/gpfs-jd/code/wangshaobo/Data_Token_Pruning"
TEMPLATE_YAML="${BASE}/multi_model_sft.yaml"  # 新的模板文件，不包含reference_model
DATA_BASE="/mnt/public/gpfs-jd/data/wangshaobo/Data_Token_Pruning"
export FORCE_TORCHRUN=1

# 禁用 PEFT Hub 下载，避免 HFValidationError
export PEFT_DISABLE_HUB_DOWNLOAD=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

#--------设置实验参数--------#
SET_BATCH_SIZE=8
EPOCHS=3
LEARNING_RATE=1e-4
# wise 插件参数
# 使用方法：bash multi_model_pruning-w.sh
# 注意：wise_lambda 参数现在在 EXPERIMENTS 数组中为每个实验单独指定
WISE_LAMBDA=${WISE_LAMBDA:-0.5}  # wise_lambda 参数，默认 0.5
#--------设置实验参数--------#

# 检查是否跳过训练（通过环境变量控制）
SKIP_TRAINING=${SKIP_TRAINING:-false}
SKIP_PRUNING=${SKIP_PRUNING:-false}
SKIP_EVALUATION=${SKIP_EVALUATION:-false}

if [ "$SKIP_TRAINING" = true ]; then
    echo "[INFO] 跳过训练模式：将直接使用现有模型进行评测"
fi

if [ "$SKIP_PRUNING" = true ]; then
    echo "[INFO] 跳过剪枝模式：将直接对剪枝后的模型进行评测"
fi

if [ "$SKIP_EVALUATION" = true ]; then
    echo "[INFO] 跳过评测模式：仅执行训练和剪枝"
fi

# 检查输出目录是否有模型文件
check_existing_models() {
    local output_dir="$1"
    local model_type="$2"

    if [ -d "$output_dir" ]; then
        echo "[INFO] 检测到 $model_type 输出目录已存在: $output_dir"
        echo "[INFO] 目录内容:"
        ls -la "$output_dir"

        # 检查是否有模型文件
        local model_files=$(find "$output_dir" -name "*.bin" -o -name "*.safetensors" -o -name "pytorch_model*" | wc -l)
        if [ "$model_files" -gt 0 ]; then
            echo "[WARNING] 发现 $model_files 个模型文件"
            echo ""
            echo "⚠️  检测到 $model_type 训练输出目录已存在并包含模型文件"
            echo "目录: $output_dir"
            echo ""
            echo "跳过$model_type 训练"
            REPLY="y"
            echo ""
            return 0
            
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                echo "[INFO] 跳过 $model_type 训练"
            return 0  # 跳过训练
            else
                echo "[INFO] 重新执行 $model_type 训练"
                return 1  # 继续训练
            fi
        else
            echo "[INFO] $model_type 目录存在但未找到模型文件，将继续训练"
            return 1
        fi
    else
        echo "[INFO] $model_type 输出目录不存在，将开始训练"
        echo output_dir: $output_dir
        echo model_type: $model_type
        return 1
    fi
}

TIME_STAMP=$(date +%Y%m%d%H%M%S)

# 初始化结果记录文件（动态表头，可由环境变量覆盖数据集列表）
RESULTS_SUMMARY_FILE="${BASE}/evaluation_results_summary/evaluation_results_summary_${TIME_STAMP}.csv"

# 允许通过环境变量覆盖数据集列表，格式：逗号分隔的字符串
# 默认数据集列表
PPL_DATASETS=("ARC_e_ppl" "ARC_c_ppl")
GEN_DATASETS=("gsm8k_gen" "squad20_gen" "triviaqa_gen")

if [ -n "$PPL_DATASETS_STR" ]; then
    IFS=',' read -r -a PPL_DATASETS <<< "$PPL_DATASETS_STR"
fi
if [ -n "$GEN_DATASETS_STR" ]; then
    IFS=',' read -r -a GEN_DATASETS <<< "$GEN_DATASETS_STR"
fi

HEADER="Experiment,DataMethod,TokenMethod,DataRatio,TokenRatio,Model,BatchSize,Epochs,LR"
for ds in "${PPL_DATASETS[@]}"; do HEADER="$HEADER,${ds}"; done
for ds in "${GEN_DATASETS[@]}"; do HEADER="$HEADER,${ds}"; done
HEADER="$HEADER,Average"
echo "$HEADER" > $RESULTS_SUMMARY_FILE

source /mnt/public/wangshaobo/miniconda3/etc/profile.d/conda.sh
ENV_BASE=/mnt/public/gpfs-jd/data/wangshaobo/conda_envs
ENV_NAME_TRAIN=llama_factory
ENV_NAME_EVAL=opencompass_env
ENV_PATH_TRAIN=${ENV_BASE}/${ENV_NAME_TRAIN}
ENV_PATH_EVAL=${ENV_BASE}/${ENV_NAME_EVAL}

MODELS_DIR="${DATA_BASE}/models"
CHECKPOINTS_DIR="${DATA_BASE}/checkpoints"
SAVE_DIR="${DATA_BASE}/saves"

# 支持的数据集列表
DATASETS=("limr" "wizard")
# 支持的模型列表
MODELS=("llama2-7b" "mistral-7b")

# 自动检测GPU数量并设置CUDA_VISIBLE_DEVICES
echo "=== GPU检测 ==="
nvidia-smi
echo ""

# 检测可用GPU数量
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo "检测到GPU数量: $GPU_COUNT"

# 设置CUDA_VISIBLE_DEVICES
if [ $GPU_COUNT -ge 8 ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    echo "使用8卡训练: CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"
elif [ $GPU_COUNT -ge 4 ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    echo "使用4卡训练: CUDA_VISIBLE_DEVICES=0,1,2,3"
elif [ $GPU_COUNT -ge 2 ]; then
    export CUDA_VISIBLE_DEVICES=0,1
    echo "使用2卡训练: CUDA_VISIBLE_DEVICES=0,1"
elif [ $GPU_COUNT -ge 1 ]; then
    export CUDA_VISIBLE_DEVICES=0
    echo "使用单卡训练: CUDA_VISIBLE_DEVICES=0"
else
    echo "警告: 未检测到可用GPU，将使用CPU训练"
    export CUDA_VISIBLE_DEVICES=""
fi

echo "最终设置: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo ""
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# 基于实际设置的CUDA_VISIBLE_DEVICES计算进程数
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    NPROC_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
else
    NPROC_PER_NODE=1  # CPU模式
fi
echo "NPROC_PER_NODE设置为: $NPROC_PER_NODE"

if [ "$NPROC_PER_NODE" -ge 8 ]; then
    HF_NUM_GPUS=8
elif [ "$NPROC_PER_NODE" -ge 4 ]; then
    HF_NUM_GPUS=4
elif [ "$NPROC_PER_NODE" -ge 2 ]; then
    HF_NUM_GPUS=2
else
    HF_NUM_GPUS=1
fi

# 实验配置数组
# 格式：DATA_METHOD TOKEN_METHOD DATA_RATIO TOKEN_RATIO MODEL BATCH_SIZE EPOCH LR [PLUG] [WISE_LAMBDA]
# 注意：WISE_LAMBDA 为可选参数，默认为 0.5
# 当 WISE_LAMBDA != 0.5 时，会自动添加到模型命名后缀中
EXPERIMENTS=(
    # # wise数据方法 + wise token方法 + 不同lambda值
    # "wise wise 12.5 50 mistral-7b 8 3 1e-4 wisely"
    # "wise wise 25 50 mistral-7b 8 3 1e-4 wisely"
    # "wise wise 50 50 mistral-7b 8 3 1e-4 wisely"
    # "wise wise 12.5 70 mistral-7b 8 3 1e-4 wisely"
    # "wise wise 25 70 mistral-7b 8 3 1e-4 wisely"
    # "wise wise 50 70 mistral-7b 8 3 1e-4 wisely"

    # "wise wise 12.5 50 llama2-7b 8 3 1e-4 wisely"
    # "wise wise 25 50 llama2-7b 8 3 1e-4 wisely"
    # "wise wise 50 50 llama2-7b 8 3 1e-4 wisely"
    # "wise wise 12.5 70 llama2-7b 8 3 1e-4 wisely"
    # "wise wise 25 70 llama2-7b 8 3 1e-4 wisely"
    # "wise wise 50 70 llama2-7b 8 3 1e-4 wisely"


    # # # #ablation
    # "random wise 100 50 mistral-7b 8 3 1e-4 wisely"
    # "random wise 100 70 mistral-7b 8 3 1e-4 wisely"
    # "wise wise 12.5 100.0 mistral-7b 8 3 1e-4 wisely"
    # "wise wise 25 100.0 mistral-7b 8 3 1e-4 wisely"
    # "wise wise 50 100.0 mistral-7b 8 3 1e-4 wisely"

    # "random wise 100 50 llama2-7b 8 3 1e-4 wisely"
    # "random wise 100 70 llama2-7b 8 3 1e-4 wisely"
    # "wise wise 12.5 100.0 llama2-7b 8 3 1e-4 wisely"
    # "wise wise 25 100.0 llama2-7b 8 3 1e-4 wisely"
    # "wise wise 50 100.0 llama2-7b 8 3 1e-4 wisely"

    # # #hyper--bs
    # "wise wise 25 50 mistral-7b 32 3 4e-4 wisely"
    # "wise wise 25 70 mistral-7b 32 3 4e-4 wisely"
    # "wise wise 50 70 mistral-7b 32 3 4e-4 wisely"
    # "wise wise 25 50 mistral-7b 16 3 2e-4 wisely"
    # "wise wise 25 70 mistral-7b 16 3 2e-4 wisely"
    # "wise wise 50 70 mistral-7b 16 3 2e-4 wisely"
    # "wise wise 25 50 mistral-7b 8 3 1e-4 wisely"
    # "wise wise 25 70 mistral-7b 8 3 1e-4 wisely"
    # "wise wise 50 70 mistral-7b 8 3 1e-4 wisely"

    # #hyper--lambda
    # "wise wise 25 50 mistral-7b 8 3 1e-4 wisely 0.0"
    # "wise wise 25 50 mistral-7b 8 3 1e-4 wisely 0.3" 
    # "wise wise 25 50 mistral-7b 8 3 1e-4 wisely 0.5"
    # "wise wise 25 50 mistral-7b 8 3 1e-4 wisely 0.7" 
    # "wise wise 25 50 mistral-7b 8 3 1e-4 wisely 1.0"
    # "wise wise 25 70 mistral-7b 8 3 1e-4 wisely 0.0"
    # "wise wise 25 70 mistral-7b 8 3 1e-4 wisely 0.3" 
    # "wise wise 25 70 mistral-7b 8 3 1e-4 wisely 0.5"
    # "wise wise 25 70 mistral-7b 8 3 1e-4 wisely 0.7" 
    # "wise wise 25 70 mistral-7b 8 3 1e-4 wisely 1.0"
    # "wise wise 50 70 mistral-7b 8 3 1e-4 wisely 0.0"
    # "wise wise 50 70 mistral-7b 8 3 1e-4 wisely 0.3" 
    # "wise wise 33 70 mistral-7b 8 3 1e-4 wisely 0.5"
    # "wise wise 50 70 mistral-7b 8 3 1e-4 wisely 0.7" 
    # "wise wise 50 70 mistral-7b 8 3 1e-4 wisely 1.0"

    # #ablation-wise
    # "wise wise 25 50 mistral-7b 8 3 1e-4 wisely"
    # "wise wise 25 70 mistral-7b 8 3 1e-4 wisely"
    # "wise wise 50 70 mistral-7b 8 3 1e-4 wisely"
    # "wise fwise 25 50 mistral-7b 8 3 1e-4 wisely"
    # "wise fwise 25 70 mistral-7b 8 3 1e-4 wisely"
    # "wise fwise 50 70 mistral-7b 8 3 1e-4 wisely"
    # "wise rho 25 50 mistral-7b 8 3 1e-4 wisely"
    # "wise rho 25 50.0 mistral-7b 8 3 1e-4 wisely"
    # "wise rho 25 70 mistral-7b 8 3 1e-4 wisely"
    # "wise rho 50 70 mistral-7b 8 3 1e-4 wisely"
    # "wise loss 25 50 mistral-7b 8 3 1e-4 wisely"
    # "wise loss 25 70 mistral-7b 8 3 1e-4 wisely"
    "wise random 25 50 mistral-7b 8 3 1e-4 wisely"
)

# 选择数据集 (默认使用limr，可通过环境变量DATASET覆盖)
DATASET=${DATASET:-"limr"}

echo "[INFO] 使用数据集: $DATASET"
echo "[INFO] 支持的模型: ${MODELS[*]}"
echo "[INFO] 总实验数量: ${#EXPERIMENTS[@]}"

for EXPERIMENT in "${EXPERIMENTS[@]}"; do
    conda activate ${ENV_PATH_TRAIN}
    # 解析实验参数：前8列固定，第9列为 plug（可选），第10列为 wise_lambda（可选）
    read -r -a FIELDS <<< "$EXPERIMENT"
    DATA_METHOD=${FIELDS[0]}
    TOKEN_METHOD=${FIELDS[1]}
    DATA_RATIO=${FIELDS[2]}
    TOKEN_RATIO=${FIELDS[3]}
    MODEL=${FIELDS[4]}
    BATCH_SIZE=${FIELDS[5]}
    EPOCH=${FIELDS[6]}
    LR=${FIELDS[7]}
    PLUG=${FIELDS[8]:-none}
    WISE_LAMBDA=${FIELDS[9]:-0.5}  # 从实验配置中读取 wise_lambda，默认为 0.5
    REFERENCE_MODEL=${REFERENCE_MODEL:-"${MODEL}_sft"}
    # distill 插件的可选超参（T、lambda1、lambda2）
    if [ "$PLUG" = "distill" ]; then
        DISTILL_T=${FIELDS[10]:-4.0}
        DISTILL_LAMBDA1=${FIELDS[11]:-1.0}
        DISTILL_LAMBDA2=${FIELDS[12]:-1.0}
    fi
    
    # 统一构造插件后缀（用于目录/文件命名）
    if [ "$PLUG" = "distill" ]; then
        DISTILL_SUFFIX="_distill_${DISTILL_T:-4.0}_${DISTILL_LAMBDA1:-1.0}_${DISTILL_LAMBDA2:-1.0}"
    else
        DISTILL_SUFFIX=""
    fi
    
    # 统一构造插件后缀（用于目录/文件命名）
    if [ "$PLUG" = "budget" ]; then
        BUDGET_SUFFIX="_budget"
    elif [ "$PLUG" = "wisely" ]; then
        # 如果 wise_lambda 不等于 0.5，则添加到后缀中
        if [ "$WISE_LAMBDA" != "0.5" ]; then
            BUDGET_SUFFIX="_wisely_${WISE_LAMBDA}"
        else
            BUDGET_SUFFIX="_wisely"
        fi
    else
        BUDGET_SUFFIX=""
    fi
    
    echo "[INFO] 开始处理实验: ${MODEL}_${DATA_METHOD}_${TOKEN_METHOD}_${DATA_RATIO}_${TOKEN_RATIO}"
    echo "[INFO] 实验参数: PLUG=${PLUG}, WISE_LAMBDA=${WISE_LAMBDA}"

    # Replace placeholder values with actual parameters
    if [ "$BATCH_SIZE" == "SET_BATCH_SIZE" ]; then
        BATCH_SIZE=$SET_BATCH_SIZE
    fi
    if [ "$EPOCH" == "EPOCHS" ]; then
        EPOCH=$EPOCHS
    fi
    if [ "$LR" == "LEARNING_RATE" ]; then
        LR=$LEARNING_RATE
    fi

    # 检查是否已有训练输出
    SKIP_CURRENT_EXPERIMENT=false
    if [ "$SKIP_TRAINING" = false ]; then
        NAME_SUFFIX_CHECK="${DATA_METHOD}_${TOKEN_METHOD}_${LR}${DISTILL_SUFFIX}${BUDGET_SUFFIX}"
        experiment_output_dir="${SAVE_DIR}/${DATASET}/${MODEL}/data_ratio_${DATA_RATIO}/token_ratio_${TOKEN_RATIO}/${NAME_SUFFIX_CHECK}"
        echo "[INFO] experiment_output_dir: ${experiment_output_dir}"
        # read -p "Press Enter to continue"
        if check_existing_models "$experiment_output_dir" "${MODEL}实验-${DATA_METHOD}_${TOKEN_METHOD}"; then
            echo "[INFO] 跳过当前实验的训练阶段，但仍执行评估"
            SKIP_CURRENT_EXPERIMENT=true
        fi
    fi

    NAME_SUFFIX="${DATA_METHOD}_${TOKEN_METHOD}_${LR}${DISTILL_SUFFIX}${BUDGET_SUFFIX}"
    OUTPUT_DIR="${SAVE_DIR}/${DATASET}/${MODEL}/data_ratio_${DATA_RATIO}/token_ratio_${TOKEN_RATIO}/${NAME_SUFFIX}"
    MERGE_DIR="${CHECKPOINTS_DIR}/${DATASET}/${MODEL}/data_ratio_${DATA_RATIO}/token_ratio_${TOKEN_RATIO}/${NAME_SUFFIX}"
    WORK_DIR="${MERGE_DIR}/results"
    
    echo "[INFO] OUTPUT_DIR: ${OUTPUT_DIR}"
    echo "[INFO] MERGE_DIR: ${MERGE_DIR}"
    echo "[INFO] WORK_DIR: ${WORK_DIR}"
    echo "[INFO] SKIP_TRAINING: ${SKIP_TRAINING}"
    echo "[INFO] SKIP_CURRENT_EXPERIMENT: ${SKIP_CURRENT_EXPERIMENT}"
    echo "[INFO] SKIP_PRUNING: ${SKIP_PRUNING}"
    echo "[INFO] PLUG: ${PLUG}"
    
    if [ "$PLUG" = "distill" ]; then
        echo "[INFO] DISTILL_T: ${DISTILL_T}, DISTILL_LAMBDA1: ${DISTILL_LAMBDA1}, DISTILL_LAMBDA2: ${DISTILL_LAMBDA2}"
    fi
    
    # 只有在需要训练时才创建临时YAML文件
    if [ "$SKIP_TRAINING" = false ]; then
        YAML_TMP="${BASE}/temp_${MODEL}_${DATASET}_${DATA_METHOD}_${TOKEN_METHOD}_${DATA_RATIO}_${TOKEN_RATIO}_${BATCH_SIZE}_${EPOCH}_${LR}${DISTILL_SUFFIX}${BUDGET_SUFFIX}.yaml"

        sed -e "s|dataset: .*|dataset: ${DATASET}|" \
                -e "s|model_name_or_path: .*|model_name_or_path: ${MODELS_DIR}/${MODEL}|" \
                -e "s|reference_model_name_or_path: .*|reference_model_name_or_path: ${MODELS_DIR}/${REFERENCE_MODEL}|" \
                -e "s|output_dir: .*|output_dir: ${OUTPUT_DIR}|" \
                -e "s|per_device_train_batch_size: .*|per_device_train_batch_size: ${BATCH_SIZE}|" \
                -e "s|num_train_epochs: .*|num_train_epochs: ${EPOCH}|" \
                -e "s|learning_rate: .*|learning_rate: ${LR}|" \
                -e "s|data_method: .*|data_method: ${DATA_METHOD}|" \
                -e "s|data_ratio: .*|data_ratio: $(awk "BEGIN {print $DATA_RATIO / 100}")|" \
                -e "s|token_method: .*|token_method: ${TOKEN_METHOD}|" \
                -e "s|token_ratio: .*|token_ratio: $(awk "BEGIN {print $TOKEN_RATIO / 100}")|" \
                -e "s|wise_lambda: .*|wise_lambda: ${WISE_LAMBDA}|" \
                $TEMPLATE_YAML > $YAML_TMP

        # 按模型选择模板
        case "$MODEL" in
            llama2-*) TEMPLATE=llama2 ;;
            qwen2*|qwen2.*|qwen2_*) TEMPLATE=qwen ;;
            qwen*-chat*|qwen1.5*-chat*|qwen1_5*-chat*) TEMPLATE=qwen ;;
            qwen-7b|qwen-*) TEMPLATE=default ;;
            mistral-*) TEMPLATE=mistral ;;
            *) TEMPLATE=default ;;
        esac
        if grep -q "^template:" "$YAML_TMP"; then
            sed -i "s|^template: .*|template: ${TEMPLATE}|" "$YAML_TMP"
        else
            echo "template: ${TEMPLATE}" >> "$YAML_TMP"
        fi

        # # 确保 dataset_dir 正确设置到数据根目录（若存在则替换，否则追加）
        # if grep -q "^dataset_dir:" "$YAML_TMP"; then
        #     sed -i "s|^dataset_dir: .*|dataset_dir: ${DATA_BASE}|" "$YAML_TMP"
        # else
        #     echo "dataset_dir: ${DATA_BASE}" >> "$YAML_TMP"
        # fi
                
        # 追加插件相关配置（YAML 末尾追加键，模板中无需预置）
        case "$PLUG" in
            # distill)
            #     {
            #         echo "plug: distill"
            #         echo "self_distill: true"
            #         echo "distill_temperature: ${DISTILL_T:-4.0}"
            #         echo "distill_lambda_1: ${DISTILL_LAMBDA1:-1.0}"
            #         echo "distill_lambda_2: ${DISTILL_LAMBDA2:-1.0}"
            #     } >> $YAML_TMP
            #     ;;
            # budget)
            #     {
            #         echo "plug: budget"
            #         echo "self_distill: false"
            #     } >> $YAML_TMP
            #     ;;
            wisely)
                {
                    echo "plug: wisely"
                    echo "self_distill: false"
                    echo "wise_lambda: ${WISE_LAMBDA}"
                } >> $YAML_TMP
                ;;
            *)
                {
                    echo "plug: none"
                    echo "self_distill: false"
                } >> $YAML_TMP
                ;;
        esac
    else
        echo "[INFO] 跳过YAML配置文件生成"
        YAML_TMP=""
    fi
    
    export PEFT_DISABLE_HUB_DOWNLOAD=1
    export HF_HUB_DISABLE_TELEMETRY=1
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export FORCE_TORCHRUN=1
    
    # 执行训练步骤（如果没有跳过）
    if [ "$SKIP_TRAINING" = false ] && [ "$SKIP_CURRENT_EXPERIMENT" = false ]; then
        echo "[INFO] 开始执行训练步骤"
        # 打印将要使用的关键训练超参，确保与模板一致
        echo "[INFO] Using params: BATCH_SIZE=${BATCH_SIZE}, EPOCH=${EPOCH}, LR=${LR}"
        # 使用双引号避免变量展开导致的问题
        echo "[INFO] YAML check: $(grep -E "^(per_device_train_batch_size|num_train_epochs|learning_rate):" "$YAML_TMP" 2>/dev/null || echo "Failed to read YAML file")"
        llamafactory-cli train $YAML_TMP

        if [ "$SKIP_PRUNING" = false ]; then
            # 执行正常的剪枝流程（不需要reference_model）
            cat <<EOF > ${BASE}/merge_config_${MODEL}.yaml
model_name_or_path: $(grep "^model_name_or_path:" "$YAML_TMP" | cut -d':' -f2- | xargs)
adapter_name_or_path: $OUTPUT_DIR
template: $(echo "${MODEL}" | grep -q "llama2" && echo "llama2" || (echo "${MODEL}" | grep -q "qwen" && echo "qwen" || echo "mistral"))
finetuning_type: lora

export_dir: $MERGE_DIR
export_size: 2
export_device: cpu
export_legacy_format: false
EOF
            llamafactory-cli export ${BASE}/merge_config_${MODEL}.yaml
        else
            echo "[INFO] 跳过模型剪枝步骤"
            # 对于跳过剪枝的情况，直接使用原始模型
            MERGE_DIR="${MODELS_DIR}/${MODEL}"
        fi
    else
        echo "[INFO] 跳过训练步骤"
        # 设置用于评测的模型路径
        if [ "$SKIP_PRUNING" = true ] || [ "$SKIP_CURRENT_EXPERIMENT" = true ]; then
            # 跳过训练和剪枝，或检测到已有模型，使用预剪枝模型进行评测
            MERGE_DIR="${CHECKPOINTS_DIR}/${DATASET}/${MODEL}/data_ratio_${DATA_RATIO}/token_ratio_${TOKEN_RATIO}/${NAME_SUFFIX}"
            echo "[INFO] 使用预剪枝模型进行评测: $MERGE_DIR"
        fi
    fi

    # 若训练完成但未进行合并导出（MERGE_DIR 不存在或无权重），尝试自动执行一次合并
    if [ "$SKIP_PRUNING" = false ]; then
        NEED_EXPORT=false
        if [ ! -d "$MERGE_DIR" ] || [ -z "$MERGE_DIR" ]; then
            NEED_EXPORT=true
        else
            model_files_count=$(find "$MERGE_DIR" -name "*.bin" -o -name "*.safetensors" -o -name "pytorch_model*" | wc -l)
            if [ "$model_files_count" -eq 0 ]; then
                NEED_EXPORT=true
            fi
        fi

        if [ "$NEED_EXPORT" = true ]; then
            echo "[INFO] 检测到尚未生成合并模型，尝试执行一次合并导出到: $MERGE_DIR"
            # 仅当训练产物目录存在时才尝试合并
            if [ -d "$OUTPUT_DIR" ]; then
                # 与上方保持一致的模板推断
                case "$MODEL" in
                    llama2-*) TEMPLATE=llama2 ;;
                    qwen2*|qwen2.*|qwen2_*) TEMPLATE=qwen ;;
                    qwen*-chat*|qwen1.5*-chat*|qwen1_5*-chat*) TEMPLATE=qwen ;;
                    qwen-7b|qwen-*) TEMPLATE=default ;;
                    mistral-*) TEMPLATE=mistral ;;
                    *) TEMPLATE=default ;;
                esac

                # 生成合并导出配置，并执行导出
                cat <<EOF > ${BASE}/merge_config_${MODEL}.yaml
model_name_or_path: ${MODELS_DIR}/${MODEL}
adapter_name_or_path: $OUTPUT_DIR
template: ${TEMPLATE}
finetuning_type: lora

export_dir: $MERGE_DIR
export_size: 2
export_device: cpu
export_legacy_format: false
EOF
                llamafactory-cli export ${BASE}/merge_config_${MODEL}.yaml
            else
                echo "[WARNING] 未找到训练输出目录: $OUTPUT_DIR，无法自动合并导出。"
            fi
        fi
    fi

    # 合并导出后再次检查模型输出是否存在
    if [ ! -d "$MERGE_DIR" ] || [ -z "$MERGE_DIR" ]; then
        echo "[ERROR] 模型输出目录不存在或为空: $MERGE_DIR，跳过评测步骤"
        continue
    fi

    # 检查是否需要执行评测
    if [ "$SKIP_EVALUATION" = false ]; then
        conda activate ${ENV_PATH_EVAL}
        
        # 定义PPL和GEN数据集
        PPL_DATASETS=("ARC_e_ppl" "ARC_c_ppl")
        GEN_DATASETS=("gsm8k_gen" "squad20_gen" "triviaqa_gen")

        # 动态评测并行度与vLLM参数（尽量加速且不OOM）
        TENSOR_PARALLEL_SIZE=${HF_NUM_GPUS}
        if [ -z "${TENSOR_PARALLEL_SIZE}" ] || [ "${TENSOR_PARALLEL_SIZE}" -lt 1 ]; then
            TENSOR_PARALLEL_SIZE=1
        fi

        # GEN 任务通常占显存更大：并行worker不超过GPU数且封顶4
        if [ "${HF_NUM_GPUS}" -ge 4 ]; then
            EVAL_MAX_WORKERS_GEN=4
        elif [ "${HF_NUM_GPUS}" -ge 2 ]; then
            EVAL_MAX_WORKERS_GEN=2
        else
            EVAL_MAX_WORKERS_GEN=1
        fi

        # PPL 任务显存相对友好，但保持保守
        if [ "${HF_NUM_GPUS}" -ge 4 ]; then
            EVAL_MAX_WORKERS_PPL=4
        elif [ "${HF_NUM_GPUS}" -ge 2 ]; then
            EVAL_MAX_WORKERS_PPL=2
        else
            EVAL_MAX_WORKERS_PPL=1
        fi

        # 检查是否有有效结果文件的函数
        check_if_results_exist() {
            local dataset_dir=$1
            
            # 检查数据集目录是否存在
            if [[ ! -d "$dataset_dir" ]]; then
                return 1  # 目录不存在
            fi
            
            # 查找所有时间戳目录并按时间戳排序（最新的在前）
            local timestamp_dirs
            timestamp_dirs=$(find "$dataset_dir" -maxdepth 1 -type d -name "20*" | sort -r)
            
            if [[ -z "$timestamp_dirs" ]]; then
                return 1  # 没找到时间戳目录
            fi
            
            # 从最新到最旧检查每个时间戳目录
            while IFS= read -r timestamp_dir; do
                if [[ -n "$timestamp_dir" ]]; then
                    # 搜索结果子目录下的任何JSON文件，无论model_abbr是什么
                    local result_root="$timestamp_dir/results"
                    if [[ -d "$result_root" ]]; then
                        # 递归查找.json结果文件
                        local json_files
                        json_files=$(find "$result_root" -type f -name "*.json")
                        for json_file in $json_files; do
                            if [[ -f "$json_file" ]] && [[ -s "$json_file" ]]; then
                                # 检查JSON文件是否包含有效数据和指标
                                if python3 -c "
import json
try:
    with open('$json_file', 'r') as f:
        data = json.load(f)
    metrics = ['accuracy', 'acc', 'exact_match', 'score', 'average_ppl', 'humaneval_pass@1', 'pass@1']
    if any(k in data and data[k] is not None for k in metrics):
        print('valid')
    else:
        print('invalid')
except Exception:
    print('invalid')
" 2>/dev/null | grep -q "valid"; then
                                    echo "Found valid results in: $timestamp_dir"
                                    return 0  # 跳过评估
                                fi
                            fi
                        done
                    fi
                fi
            done <<< "$timestamp_dirs"
            
            return 1  # 在任何时间戳目录中都没找到有效结果文件
        }

      # ====== 资源与通用优化 ======
        export OMP_NUM_THREADS=8
        export TOKENIZERS_PARALLELISM=false

        # ====== 你的机器：H200 141GB × 8 ======
        TOTAL_GPUS=8

        # —— 档位 A：吞吐优先 —— #
        # PPL：8 实例 × 1 卡
        EVAL_MAX_WORKERS_PPL=${EVAL_MAX_WORKERS_PPL:-8}
        HF_NUM_GPUS_PPL=1
        TP_SIZE_PPL=1

        # GEN：4 实例 × 2 卡
        EVAL_MAX_WORKERS_GEN=${EVAL_MAX_WORKERS_GEN:-4}
        HF_NUM_GPUS_GEN=2
        TP_SIZE_GEN=2

        # vLLM：批内并发与显存使用
        VLLM_GPU_UTIL_PPL=0.96
        VLLM_MAX_BTOK_PPL=131072
        VLLM_MAX_SEQS_PPL=1024

        VLLM_GPU_UTIL_GEN=0.94
        VLLM_MAX_BTOK_GEN=65536
        VLLM_MAX_SEQS_GEN=512

        # ====== PPL 评测（ARC_e_ppl,ARC_c_ppl） ======
        for EVAL_DATASET in "${PPL_DATASETS[@]}"; do
        dataset_dir="${WORK_DIR}/$EVAL_DATASET"
        if check_if_results_exist "$dataset_dir"; then
            echo "Results for $EVAL_DATASET already exist, skipping evaluation."
        else
            python ${BASE}/opencompass/run.py \
            --datasets $EVAL_DATASET \
            --hf-type base \
            --hf-path ${MERGE_DIR} \
            --work-dir ${WORK_DIR}/$EVAL_DATASET \
            --max-num-workers ${EVAL_MAX_WORKERS_PPL} \
            --hf-num-gpus ${HF_NUM_GPUS_PPL} \
            --accelerator vllm \
            --model-kwargs tensor_parallel_size=${TP_SIZE_PPL} gpu_memory_utilization=${VLLM_GPU_UTIL_PPL} max_num_batched_tokens=${VLLM_MAX_BTOK_PPL} max_num_seqs=${VLLM_MAX_SEQS_PPL}
        fi
        done

        # ====== GEN 评测（按数据集分别设 max-out-len） ======
        gen_len_of () {
        case "$1" in
            "gsm8k_gen") echo 768 ;;
            "triviaqa_gen") echo 384 ;;
            "squad20_gen") echo 256 ;;
            *) echo 256 ;;
        esac
        }

        for EVAL_DATASET in "${GEN_DATASETS[@]}"; do
        dataset_dir="${WORK_DIR}/$EVAL_DATASET"
        if check_if_results_exist "$dataset_dir"; then
            echo "Results for $EVAL_DATASET already exist, skipping evaluation."
        else
            MAXLEN=$(gen_len_of "$EVAL_DATASET")
            python ${BASE}/opencompass/run.py \
            --datasets $EVAL_DATASET \
            --hf-type base \
            --hf-path ${MERGE_DIR} \
            --work-dir ${WORK_DIR}/$EVAL_DATASET \
            --max-num-workers ${EVAL_MAX_WORKERS_GEN} \
            --hf-num-gpus ${HF_NUM_GPUS_GEN} \
            --accelerator vllm \
            --max-out-len ${MAXLEN} \
            --model-kwargs tensor_parallel_size=${TP_SIZE_GEN} gpu_memory_utilization=${VLLM_GPU_UTIL_GEN} max_num_batched_tokens=${VLLM_MAX_BTOK_GEN} max_num_seqs=${VLLM_MAX_SEQS_GEN}
        fi
        done


    fi  # 结束 SKIP_EVALUATION 条件检查

    # 清理临时文件
    if [ -n "$YAML_TMP" ] && [ -f "$YAML_TMP" ]; then
        rm -f "$YAML_TMP"
    fi
    if [ -f "${BASE}/merge_config_${MODEL}.yaml" ]; then
        rm -f "${BASE}/merge_config_${MODEL}.yaml"
    fi
    
    # 收集评测结果
    EXPERIMENT_NAME="${MODEL}_${DATA_METHOD}_${TOKEN_METHOD}_${DATA_RATIO}_${TOKEN_RATIO}${DISTILL_SUFFIX}${BUDGET_SUFFIX}"
    
    # 动态提取评测结果并写入CSV
    ROW_PREFIX="${EXPERIMENT_NAME},${DATA_METHOD},${TOKEN_METHOD},${DATA_RATIO},${TOKEN_RATIO},${MODEL},${BATCH_SIZE},${EPOCH},${LR}"
    ROW_VALUES=""

    # 通用解析函数：优先从最深层summary/metrics中抓取acc/exact_match/score等常用字段
    parse_metric() {
        local DIR_PATH="$1"
        local METRIC
        METRIC=$(python3 - <<PY 2>/dev/null
import json
import glob
import os

def pick_metric(data, *keys):
    """从数据中提取指标值，按优先级顺序"""
    for k in keys:
        if k in data and isinstance(data[k], (int, float, str)):
            return data[k]
    return None

def collect_metrics(work_dir, eval_dataset):
    """收集指定评估数据集的所有指标"""
    metrics = {}
    
    # 构建搜索模式：查找所有时间戳下的结果文件
    pattern = os.path.join(work_dir, eval_dataset, "*", "results", "*", "*.json")
    files = sorted(glob.glob(pattern, recursive=True))
    
    if not files:
        return None
    
    # 处理每个结果文件
    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # 提取文件名作为指标标识（去掉.json后缀）
            file_name = os.path.basename(file_path).replace('.json', '')
            
            # 按优先级提取指标值
            metric_value = pick_metric(data, 
                                    'accuracy', 'acc', 'exact_match', 'score', 
                                    'average_ppl', 'humaneval_pass@1', 'pass@1')
            
            if metric_value is not None:
                metrics[file_name] = metric_value
                
        except Exception as e:
            print(f"Error reading {file_path}: {e}", file=sys.stderr)
            continue
    
    return metrics

# 主逻辑
work_dir = r"${WORK_DIR}"
eval_dataset = r"${DIR_PATH}"

# 收集所有指标
all_metrics = collect_metrics(work_dir, eval_dataset)

if all_metrics:
    # 如果有多个指标文件，按文件名排序输出
    if len(all_metrics) == 1:
        # 单个文件，直接输出值
        print(list(all_metrics.values())[0])
    else:
        # 多个文件，输出所有指标（用逗号分隔）
        sorted_metrics = sorted(all_metrics.items())
        metric_strings = [f"{k}:{v}" for k, v in sorted_metrics]
        print(";".join(metric_strings))
else:
    print('N/A')
PY
)
        echo "$METRIC"
    }

    # 依配置顺序拼接各数据集结果
    ROW_VALUES=""
    METRIC_VALUES=()
    
    # 只收集数据集结果列（从第9列开始），不包括配置信息
    for ds in "${PPL_DATASETS[@]}"; do
        if [ -d "${WORK_DIR}/${ds}" ]; then
            metric=$(parse_metric "$ds")
            ROW_VALUES+="${metric},"
            METRIC_VALUES+=("$metric")
        else
            ROW_VALUES+="N/A,"
        fi
    done
    for ds in "${GEN_DATASETS[@]}"; do
        if [ -d "${WORK_DIR}/${ds}" ]; then
            metric=$(parse_metric "$ds")
            ROW_VALUES+="${metric},"
            METRIC_VALUES+=("$metric")
        else
            ROW_VALUES+="N/A,"
        fi
    done
    
    # 计算平均值
    AVERAGE_VALUE="N/A"
    VALID_COUNT=0
    TOTAL_SUM=0
    
    for metric in "${METRIC_VALUES[@]}"; do
        if [[ "$metric" != "N/A" ]]; then
            # 处理包含多个值的情况（如 race-high:86.53;race-middle:90.59）
            if [[ "$metric" == *";"* ]]; then
                # 提取所有数值并计算平均值
                sub_metrics=(${metric//;/ })
                sub_sum=0
                sub_count=0
                for sub_metric in "${sub_metrics[@]}"; do
                    # 提取冒号后的数值
                    value=${sub_metric#*:}
                    # 如果提取失败，尝试其他方法
                    if [ -z "$value" ]; then
                        value="$sub_metric"
                    fi
                    if [[ "$value" =~ ^[0-9]+\.?[0-9]*$ ]] || [[ "$value" =~ ^[0-9]+\.?[0-9]*[eE][+-]?[0-9]+$ ]]; then
                        # 使用更可靠的数值计算
                        if command -v bc >/dev/null 2>&1; then
                            sub_sum=$(echo "scale=10; $sub_sum + $value" | bc -l)
                        else
                            sub_sum=$(awk "BEGIN {printf \"%.10f\", $sub_sum + $value}")
                        fi
                        sub_count=$((sub_count + 1))
                    fi
                done
                if [ "$sub_count" -gt 0 ]; then
                    if command -v bc >/dev/null 2>&1; then
                        sub_avg=$(echo "scale=10; $sub_sum / $sub_count" | bc -l)
                    else
                        sub_avg=$(awk "BEGIN {printf \"%.10f\", $sub_sum / $sub_count}")
                    fi

                    if command -v bc >/dev/null 2>&1; then
                        TOTAL_SUM=$(echo "scale=10; $TOTAL_SUM + $sub_avg" | bc -l)
                    else
                        TOTAL_SUM=$(awk "BEGIN {printf \"%.10f\", $TOTAL_SUM + $sub_avg}")
                    fi
                    VALID_COUNT=$((VALID_COUNT + 1))
                fi
            else
                # 处理单个数值（支持科学计数法）
                if [[ "$metric" =~ ^[0-9]+\.?[0-9]*$ ]] || [[ "$metric" =~ ^[0-9]+\.?[0-9]*[eE][+-]?[0-9]+$ ]]; then
                    if command -v bc >/dev/null 2>&1; then
                        TOTAL_SUM=$(echo "scale=10; $TOTAL_SUM + $metric" | bc -l)
                    else
                        TOTAL_SUM=$(awk "BEGIN {printf \"%.10f\", $TOTAL_SUM + $metric}")
                    fi
                    VALID_COUNT=$((VALID_COUNT + 1))
                else
                    echo "[DEBUG] 数值无效: '$metric'"
                fi
            fi
        fi
    done
    
    if [ "$VALID_COUNT" -gt 0 ]; then
        if command -v bc >/dev/null 2>&1; then
            AVERAGE_VALUE=$(echo "scale=4; $TOTAL_SUM / $VALID_COUNT" | bc -l)
        else
            AVERAGE_VALUE=$(echo "$TOTAL_SUM $VALID_COUNT" | awk '{printf "%.4f", $1/$2}')
        fi
    fi
    
    # 去掉最后一个逗号并添加平均值
    ROW_VALUES=${ROW_VALUES%,}
    echo "${ROW_PREFIX},${ROW_VALUES},${AVERAGE_VALUE}" >> $RESULTS_SUMMARY_FILE
done

# 生成markdown格式的评测结果表格
MARKDOWN_TABLE="${BASE}/evaluation_results/evaluation_results_${TIME_STAMP}.md"
mkdir -p "$(dirname "$MARKDOWN_TABLE")"
echo "# Multi-Model Data Token Pruning 实验评测结果汇总表" > $MARKDOWN_TABLE
echo "" >> $MARKDOWN_TABLE
echo "| 实验配置 | 数据方法 | Token方法 | 数据比例 | Token比例 | 模型 | 批大小 | 轮数 | 学习率 |ARC-e | ARC-c | GSM8K | SQuAD2.0 | TriviaQA | 平均值 |" >> $MARKDOWN_TABLE
echo "|----------|----------|-----------|----------|-----------|------|--------|------|---------|--------|-------|-----------|-------|----------|--------|" >> $MARKDOWN_TABLE

# 从CSV文件读取结果并转换为markdown格式
tail -n +2 $RESULTS_SUMMARY_FILE | while IFS=',' read -r experiment data_method token_method data_ratio token_ratio model batch_size epochs lr arc_e_result arc_c_result gsm8k_result squad20_result triviaqa_result average_value; do
    echo "| ${experiment} | ${data_method} | ${token_method} | ${data_ratio}% | ${token_ratio}% | ${model} | ${batch_size} | ${epochs} | ${lr} | ${arc_e_result} | ${arc_c_result} | ${gsm8k_result} | ${squad20_result} | ${triviaqa_result} | ${average_value} |" >> $MARKDOWN_TABLE
done

echo "" >> $MARKDOWN_TABLE
echo "评测完成时间: $(date)" >> $MARKDOWN_TABLE
echo "数据集: $DATASET" >> $MARKDOWN_TABLE
echo "CSV结果文件: $RESULTS_SUMMARY_FILE" >> $MARKDOWN_TABLE
echo "Markdown表格文件: $MARKDOWN_TABLE" >> $MARKDOWN_TABLE

# 将本次评测的元数据（时间戳与结果路径）写出，供上层脚本读取
META_DIR="${BASE}/experiment_logs"
META_FILE="${META_DIR}/multi_model_last_run.env"
mkdir -p "$META_DIR"
{
    echo "TIME_STAMP=${TIME_STAMP}"
    echo "RESULTS_SUMMARY_FILE=${RESULTS_SUMMARY_FILE}"
    echo "MARKDOWN_TABLE=${MARKDOWN_TABLE}"
    echo "DATASET=${DATASET}"
} > "$META_FILE"
echo "[INFO] 元数据已写入: $META_FILE"

echo "=================================================="
echo "所有Multi-Model实验完成！"
echo "使用数据集: $DATASET"
echo "支持模型: ${MODELS[*]}"
echo "评测结果CSV文件: $RESULTS_SUMMARY_FILE"
echo "评测结果Markdown表格: $MARKDOWN_TABLE"
echo "=================================================="
