#!/bin/bash

# ========================================
# Multi-Model Data Token Pruning 完整实验执行脚本
# Models: llama2-7b, qwen2-7b, mistral-7b
# Datasets: limr, wizard
# ========================================

bash /mnt/public/wangshaobo/conda_init.sh

echo "🚀 开始Multi-Model Data Token Pruning实验"
echo "支持模型: llama2-7b, qwen2-7b, mistral-7b"
echo "支持数据集: limr, wizard"
echo "========================================"

export PEFT_DISABLE_HUB_DOWNLOAD=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export FORCE_TORCHRUN=1

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# 基础路径
BASE="/mnt/public/gpfs-jd/code/wangshaobo/Data_Token_Pruning"
DATA_BASE="/mnt/public/gpfs-jd/data/wangshaobo/Data_Token_Pruning"
MODELS_DIR="${DATA_BASE}/models"
SAVES_DIR="${DATA_BASE}/saves"
CHECKPOINTS_DIR="${DATA_BASE}/checkpoints"
LOG_DIR="${BASE}/experiment_logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 创建必要目录
mkdir -p "$LOG_DIR"
mkdir -p "${BASE}/evaluation_results_summary"
mkdir -p "${BASE}/evaluation_results"

# 日志文件
MASTER_LOG="${LOG_DIR}/multi_model_experiment_${TIMESTAMP}.log"
PRUNING_LOG="${LOG_DIR}/multi_model_pruning_${TIMESTAMP}.log"

# 执行状态函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$MASTER_LOG"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$MASTER_LOG"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$MASTER_LOG"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$MASTER_LOG"
}

log_stage() {
    echo -e "${BOLD}${BLUE}========================================${NC}" | tee -a "$MASTER_LOG"
    echo -e "${BOLD}${BLUE}$1${NC}" | tee -a "$MASTER_LOG"
    echo -e "${BOLD}${BLUE}========================================${NC}" | tee -a "$MASTER_LOG"
}

# 检查脚本和文件是否存在
check_dependencies() {
    log_info "检查必要文件和脚本是否存在..."
    
    local dependencies=(
        "${BASE}/multi_model_pruning-w.sh"
        "${BASE}/multi_model_sft.yaml"
        "${BASE}/gpu_environment_check.sh"
        "${DATA_BASE}/limr.json"
        "${DATA_BASE}/wizard.json"
    )
    
    for dep in "${dependencies[@]}"; do
        if [ -f "$dep" ]; then
            log_success "文件存在: $(basename $dep)"
        else
            log_error "文件不存在: $dep"
            return 1
        fi
    done
    
    # 检查模型目录
    local models=("llama2-7b" "qwen2-7b" "mistral-7b")
    for model in "${models[@]}"; do
        if [ -d "${MODELS_DIR}/${model}" ]; then
            log_success "模型存在: $model"
        else
            log_warning "模型不存在: ${MODELS_DIR}/${model}"
        fi
    done
    
    return 0
}

# 运行环境检查
run_environment_check() {
    log_stage "阶段0: GPU环境检查"
    
    log_info "运行GPU环境检查..."
    # if [ -f "${BASE}/gpu_environment_check.sh" ]; then
    #     bash "${BASE}/gpu_environment_check.sh" 2>&1 | tee -a "$MASTER_LOG"
    #     local exit_code=$?
        
    #     if [ $exit_code -eq 0 ]; then
    #         log_success "GPU环境检查通过，可以开始训练"
    #     elif [ $exit_code -eq 1 ]; then
    #         log_warning "GPU环境检查有警告，但可以继续执行"
    #     else
    #         log_error "GPU环境检查失败，请先修复环境问题"
    #         return 1
    #     fi
    # else
    #     log_warning "GPU环境检查脚本不存在，跳过环境检查"
    # fi
    
    return 0
}

# 选择数据集
select_dataset() {
    local dataset_choice="$1"
    
    case "$dataset_choice" in
        "limr"|"1")
            export DATASET="limr"
            log_info "选择数据集: limr"
            ;;
        "wizard"|"2")
            export DATASET="wizard"  
            log_info "选择数据集: wizard"
            ;;
        "alpaca"|"3")
            export DATASET="alpaca"  
            log_info "选择数据集: alpaca"
            ;;
        *)
            export DATASET="wizard"  # 默认使用wizard
            log_warning "无效的数据集选择，使用默认数据集: limr"
            ;;
    esac
}

# 执行Multi-Model剪枝实验
run_multi_model_pruning() {
    log_stage "Multi-Model Data Token Pruning 实验"

    log_info "开始Multi-Model剪枝实验..."
    log_info "数据集: $DATASET"
    log_info "支持模型: llama2-7b, qwen2-7b, mistral-7b"
    log_info "数据方法: random, infobatch, longest, entropy"
    log_info "Token方法: random, sample_fastv (移除rho)"
    log_info "日志文件: $PRUNING_LOG"

    # 根据全局变量设置环境变量
    if [ "$SKIP_TRAINING" = true ]; then
        export SKIP_TRAINING=true
        log_info "设置: 跳过训练"
    fi
    if [ "$SKIP_PRUNING" = true ]; then
        export SKIP_PRUNING=true
        log_info "设置: 跳过剪枝"
    fi
    if [ "$SKIP_EVALUATION" = true ]; then
        export SKIP_EVALUATION=true
        log_info "设置: 跳过评测"
    fi

    # 记录开始时间
    local start_time=$(date)
    log_info "剪枝实验开始时间: $start_time"

    # 执行剪枝实验脚本
    bash "${BASE}/multi_model_pruning-w.sh" 2>&1 | tee "$PRUNING_LOG"
    local pruning_exit_code=$?

    # 记录结束时间
    local end_time=$(date)
    log_info "剪枝实验结束时间: $end_time"

    if [ $pruning_exit_code -eq 0 ]; then
        log_success "Multi-Model剪枝实验完成"
        return 0
    else
        log_error "Multi-Model剪枝实验失败 (退出代码: $pruning_exit_code)"
        return 1
    fi
}

# 执行仅评测（跳过训练和剪枝）
run_evaluation_only() {
    log_stage "Multi-Model 仅评测模式"

    log_info "开始Multi-Model仅评测..."
    log_info "数据集: $DATASET"
    log_info "直接使用现有模型进行评测"
    log_info "日志文件: $PRUNING_LOG"

    # 设置环境变量：跳过训练和剪枝，但执行评测
    export SKIP_TRAINING=true
    export SKIP_PRUNING=true  
    export SKIP_EVALUATION=false

    # 记录开始时间
    local start_time=$(date)
    log_info "评测开始时间: $start_time"

    # 执行评测脚本
    bash "${BASE}/multi_model_pruning-w.sh" 2>&1 | tee "$PRUNING_LOG"
    local eval_exit_code=$?

    # 记录结束时间
    local end_time=$(date)
    log_info "评测结束时间: $end_time"

    if [ $eval_exit_code -eq 0 ]; then
        log_success "Multi-Model仅评测完成"
        return 0
    else
        log_error "Multi-Model仅评测失败 (退出代码: $eval_exit_code)"
        return 1
    fi
}

# 生成最终报告
generate_final_report() {
    log_stage "生成最终实验报告"
    
    # 从脚本写出的元数据中读取时间戳与结果路径
    local META_FILE="${LOG_DIR}/multi_model_last_run.env"
    local MULTI_TS="$TIMESTAMP"
    local report_file
    local csv_results
    local markdown_results

    if [ -f "$META_FILE" ]; then
        log_info "检测到评测元数据: $META_FILE"
        source "$META_FILE"
        if [ -n "$TIME_STAMP" ]; then
            MULTI_TS="$TIME_STAMP"
        fi
        if [ -n "$RESULTS_SUMMARY_FILE" ]; then
            csv_results="$RESULTS_SUMMARY_FILE"
        else
            csv_results="${BASE}/evaluation_results_summary/evaluation_results_summary_${MULTI_TS}.csv"
        fi
        if [ -n "$MARKDOWN_TABLE" ]; then
            markdown_results="$MARKDOWN_TABLE"
        else
            markdown_results="${BASE}/evaluation_results/evaluation_results_${MULTI_TS}.md"
        fi
    else
        log_warning "未发现评测元数据文件，回退使用当前时间戳"
        csv_results="${BASE}/evaluation_results_summary/evaluation_results_summary_${TIMESTAMP}.csv"
        markdown_results="${BASE}/evaluation_results/evaluation_results_${TIMESTAMP}.md"
    fi

    report_file="${LOG_DIR}/multi_model_experiment_report_${MULTI_TS}.md"
    
    {
        echo "# Multi-Model Data Token Pruning 实验报告"
        echo ""
        echo "**实验时间:** $(date)"
        echo "**实验时间戳:** $MULTI_TS"
        echo "**数据集:** ${DATASET:-limr}"
        echo ""
        echo "## 实验配置"
        echo "- **支持模型:** llama2-7b, qwen2-7b, mistral-7b"
        echo "- **数据集:** ${DATASET:-limr} (limr.json 或 wizard.json)"
        echo "- **数据剪枝方法:** random, infobatch, longest, entropy"
        echo "- **Token剪枝方法:** random, sample_fastv (移除rho)"
        echo "- **数据比例:** 30%, 50%, 70%"
        echo "- **Token比例:** 30%, 50%, 70%"
        echo ""
        echo "## 主要改进"
        echo "1. **简化架构:** 移除reference_model，简化路径结构"
        echo "2. **多模型支持:** 支持llama2, qwen2, mistral三种模型架构"
        echo "3. **新数据集:** 使用limr.json和wizard.json替代alpaca"
        echo "4. **精简token方法:** 移除rho方法，专注于random和sample_fastv"
        echo ""
        echo "## 实验流程"
        echo "1. **数据预处理:** 根据选择的数据集加载对应的JSON文件"
        echo "2. **模型训练:** 使用LoRA微调各个基础模型"
        echo "3. **模型剪枝:** 应用数据和token剪枝策略"
        echo "4. **评测:** OpenCompass多数据集评测"
        echo ""
        echo "## 日志文件"
        echo "- **主日志:** $MASTER_LOG"
        echo "- **剪枝实验日志:** $PRUNING_LOG"
        echo ""
        echo "## 评测结果"
        
        if [ -f "$markdown_results" ]; then
            echo "详细评测结果请查看: $markdown_results"
            echo ""
            echo "### 结果摘要"
            cat "$markdown_results" | tail -n +3
        else
            echo "评测结果文件未找到: $markdown_results"
        fi
        
        echo ""
        echo "## 结果文件位置"
        echo "- **CSV结果:** $csv_results"
        echo "- **Markdown结果:** $markdown_results"
        echo "- **模型输出:** ${CHECKPOINTS_DIR}/"
        echo "- **训练输出:** ${SAVES_DIR}/"
        echo ""
        echo "## 路径结构变化"
        echo "**旧结构:** \`\${SAVE_DIR}/\${DATASET}/\${MODEL}/\${REFERENCE_MODEL}/...\`"
        echo "**新结构:** \`\${SAVE_DIR}/\${DATASET}/\${MODEL}/...\`"
        echo ""
        echo "移除了reference_model层级，简化了目录结构。"
    } > "$report_file"
    
    log_success "最终实验报告已生成: $report_file"
}

# 显示使用说明
show_usage() {
    echo "用法: $0 [模式] [数据集]"
    echo ""
    echo "模式选项:"
    echo "  1 - 完整实验 (训练 + 剪枝 + 评测)"
    echo "  2 - 仅评测 (使用现有模型)"
    echo "  3 - 仅训练和剪枝 (跳过评测)"
    echo ""
    echo "数据集选项:"
    echo "  limr/1 - 使用limr.json数据集"
    echo "  wizard/2 - 使用wizard.json数据集"
    echo ""
    echo "示例:"
    echo "  $0 1 limr    # 完整实验，使用limr数据集"
    echo "  $0 2 wizard  # 仅评测，使用wizard数据集"
    echo ""
}

# 主执行函数
main() {
    # 记录开始时间
    local experiment_start_time=$(date)
    log_info "Multi-Model实验开始时间: $experiment_start_time"

    # 解析参数
    local mode_choice=${1:-1}
    local dataset_choice=${2:-limr}
    
    # 如果没有提供参数，显示交互式选择
    if [ $# -eq 0 ]; then
        echo "请选择实验模式:"
        echo "  1) 完整实验 (训练 + 剪枝 + 评测)"
        echo "  2) 仅评测 (使用现有模型)"
        echo "  3) 仅训练和剪枝 (跳过评测)"
        echo ""
        read -p "请输入选择 (1/2/3): " mode_choice
        
        echo ""
        echo "请选择数据集:"
        echo "  1) limr (limr.json)"
        echo "  2) wizard (wizard.json)"
        echo ""
        read -p "请输入选择 (1/2): " dataset_choice
    fi

    # 选择数据集
    select_dataset "$dataset_choice"

    # 用户选择模式
    log_stage "实验模式配置"
    local SKIP_TRAINING=false
    local SKIP_PRUNING=false
    local SKIP_EVALUATION=false

    case "$mode_choice" in
        1)
            log_info "选择：完整实验模式 (训练 + 剪枝 + 评测)"
            SKIP_TRAINING=false
            SKIP_PRUNING=false
            SKIP_EVALUATION=false
            ;;
        2)
            log_info "选择：仅评测模式"
            SKIP_TRAINING=true
            SKIP_PRUNING=true
            SKIP_EVALUATION=false
            ;;
        3)
            log_info "选择：仅训练和剪枝模式"
            SKIP_TRAINING=false
            SKIP_PRUNING=false
            SKIP_EVALUATION=true
            ;;
        *)
            log_warning "无效选择，使用默认完整实验模式"
            SKIP_TRAINING=false
            SKIP_PRUNING=false
            SKIP_EVALUATION=false
            ;;
    esac

    echo ""

    # 检查依赖
    if ! check_dependencies; then
        log_error "依赖检查失败，退出"
        exit 1
    fi

    # 环境检查
    if ! run_environment_check; then
        log_error "环境检查失败，退出"
        exit 1
    fi

    # 设置全局环境变量
    export SKIP_TRAINING
    export SKIP_PRUNING  
    export SKIP_EVALUATION

    # 执行实验
    if [ "$SKIP_TRAINING" = true ] && [ "$SKIP_PRUNING" = true ]; then
        # 仅评测模式
        if ! run_evaluation_only; then
            log_error "仅评测失败"
            exit 1
        fi
    else
        # 完整实验或仅训练剪枝模式
        if ! run_multi_model_pruning; then
            log_error "Multi-Model剪枝实验失败"
            exit 1
        fi
    fi

    # 生成报告
    generate_final_report
    
    # 记录完成时间
    local experiment_end_time=$(date)
    log_info "Multi-Model实验结束时间: $experiment_end_time"
    
    log_stage "🎉 Multi-Model实验执行完成！"

    echo ""
    echo "实验总结:"
    echo "  ✅ 数据集: ${DATASET}"
    echo "  ✅ 支持模型: llama2-7b, qwen2-7b, mistral-7b"
    
    if [ "$SKIP_TRAINING" = false ]; then
        echo "  ✅ 训练: LoRA微调各个基础模型"
    else
        echo "  ⏭️  跳过训练阶段"
    fi

    if [ "$SKIP_PRUNING" = false ]; then
        echo "  ✅ 剪枝: 数据和Token剪枝实验"
    else
        echo "  ⏭️  跳过剪枝阶段"
    fi
    
    if [ "$SKIP_EVALUATION" = false ]; then
        echo "  ✅ 评测: OpenCompass多数据集评测"
    else
        echo "  ⏭️  跳过评测阶段"
    fi

    echo "  ✅ 结果汇总: 自动生成评测结果表格"
    echo ""

    echo "结果文件:"
    local final_ts="${TIME_STAMP:-$TIMESTAMP}"
    echo "  📊 评测结果: ${BASE}/evaluation_results/evaluation_results_${final_ts}.md"
    echo "  📈 CSV数据: ${BASE}/evaluation_results_summary/evaluation_results_summary_${final_ts}.csv"
    echo "  📝 实验报告: ${LOG_DIR}/multi_model_experiment_report_${final_ts}.md"
    echo ""

    echo "如需重新运行任何阶段，请查看对应的日志文件获取详细信息。"
    echo "使用 '$0 --help' 查看详细使用说明。"
}

# 检查帮助参数
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    show_usage
    exit 0
fi

# 脚本开始执行
echo "日志将保存到: $MASTER_LOG"
echo ""

# 捕获中断信号
trap 'log_error "实验被用户中断"; exit 130' INT

# 执行主函数
export WANDB_MODE=offline
export WANDB_DIR=${BASE}/wandb
mkdir -p "$WANDB_DIR"

main "$@"
