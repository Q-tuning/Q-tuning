#!/bin/bash

# ========================================
# GPU环境验证脚本 - 完整验证GPU训练环境
# ========================================

echo "开始GPU环境验证..."
echo "========================================"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 计数器
PASSED=0
FAILED=0
WARNINGS=0

# 详细错误日志文件
ERROR_LOG="/mnt/public/gpfs-jd/code/wangshaobo/Data_Token_Pruning/gpu_env_check_errors.log"
> $ERROR_LOG

# 验证函数
check_passed() {
    echo -e "${GREEN}[PASSED]${NC} $1"
    ((PASSED++))
}

check_failed() {
    echo -e "${RED}[FAILED]${NC} $1"
    echo "[ERROR] $1" >> $ERROR_LOG
    ((FAILED++))
}

check_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
    echo "[WARNING] $1" >> $ERROR_LOG
    ((WARNINGS++))
}

check_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# 首先运行基础环境检查
echo "========================================"
echo "第一步: 运行基础环境检查"
echo "========================================"

BASE_CHECK_SCRIPT="/mnt/public/gpfs-jd/code/wangshaobo/Data_Token_Pruning/environment_check.sh"
if [ -f "$BASE_CHECK_SCRIPT" ]; then
    bash "$BASE_CHECK_SCRIPT"
    BASE_CHECK_EXIT_CODE=$?
    if [ $BASE_CHECK_EXIT_CODE -eq 0 ]; then
        check_passed "基础环境检查完全通过"
    elif [ $BASE_CHECK_EXIT_CODE -eq 1 ]; then
        check_warning "基础环境检查有警告"
    else
        check_failed "基础环境检查失败"
        echo "请先修复基础环境问题"
    fi
else
    check_failed "基础环境检查脚本不存在: $BASE_CHECK_SCRIPT"
fi

echo ""
echo "========================================"
echo "第二步: GPU特定环境检查"
echo "========================================"

# 1. 检查NVIDIA驱动和CUDA
echo "1. 检查NVIDIA驱动和CUDA..."

if command -v nvidia-smi >/dev/null 2>&1; then
    check_passed "nvidia-smi可用"
    
    # 获取GPU信息
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits)
    if [ ! -z "$GPU_INFO" ]; then
        check_info "GPU信息:"
        echo "$GPU_INFO" | while read line; do
            check_info "  - $line"
        done
        
        # 检查GPU数量
        GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
        check_info "检测到 $GPU_COUNT 个GPU"
        
        if [ $GPU_COUNT -ge 4 ]; then
            check_passed "GPU数量充足: $GPU_COUNT 个"
        else
            check_warning "GPU数量较少: $GPU_COUNT 个 (建议至少4个)"
        fi
        
        # 检查GPU内存
        GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
        if [ $GPU_MEMORY -ge 20000 ]; then
            check_passed "GPU内存充足: ${GPU_MEMORY}MB"
        else
            check_warning "GPU内存可能不足: ${GPU_MEMORY}MB (建议至少20GB)"
        fi
    else
        check_failed "无法获取GPU信息"
    fi
else
    check_failed "nvidia-smi不可用，请检查NVIDIA驱动安装"
fi

# 2. 检查CUDA版本
echo ""
echo "2. 检查CUDA版本..."

if command -v nvcc >/dev/null 2>&1; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    check_passed "CUDA编译器可用，版本: $CUDA_VERSION"
else
    check_warning "nvcc不可用，但这可能是正常的（如果使用conda安装的CUDA）"
fi

# 从nvidia-smi获取CUDA版本
if command -v nvidia-smi >/dev/null 2>&1; then
    CUDA_DRIVER_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    if [ ! -z "$CUDA_DRIVER_VERSION" ]; then
        check_info "CUDA驱动版本: $CUDA_DRIVER_VERSION"
    fi
fi

# 3. 检查PyTorch CUDA支持
echo ""
echo "3. 检查PyTorch CUDA支持..."

# 在llama_factory环境中检查
LLAMA_FACTORY_PYTHON="/mnt/public/gpfs-jd/data/wangshaobo/conda_envs/llama_factory/bin/python3"

TORCH_CUDA_CHECK=$("$LLAMA_FACTORY_PYTHON" -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA版本: {torch.version.cuda}')
    print(f'GPU数量: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'  内存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB')
" 2>/dev/null)

if [ $? -eq 0 ]; then
    check_passed "PyTorch CUDA检查成功:"
    echo "$TORCH_CUDA_CHECK" | while read line; do
        check_info "  $line"
    done
else
    check_failed "PyTorch CUDA检查失败，请检查llama_factory环境中的PyTorch安装"
    echo "错误详情:" >> $ERROR_LOG
    "$LLAMA_FACTORY_PYTHON" -c "import torch; print(torch.cuda.is_available())" >> $ERROR_LOG 2>&1
fi

# 4. 检查transformers和其他关键包
echo ""
echo "4. 检查关键Python包..."

PACKAGES_TO_CHECK=("transformers" "datasets" "accelerate" "deepspeed" "peft")

for package in "${PACKAGES_TO_CHECK[@]}"; do
    VERSION=$("$LLAMA_FACTORY_PYTHON" -c "import $package; print($package.__version__)" 2>/dev/null)
    if [ $? -eq 0 ]; then
        check_passed "$package: $VERSION"
    else
        check_failed "$package 包未安装或版本有问题"
        echo "$package导入错误:" >> $ERROR_LOG
        "$LLAMA_FACTORY_PYTHON" -c "import $package" >> $ERROR_LOG 2>&1
    fi
done

# 5. 检查DeepSpeed配置和GPU兼容性
echo ""
echo "5. 检查DeepSpeed GPU兼容性..."

# 检查ds_report命令
DEEPSPEED_DS_REPORT="/mnt/public/gpfs-jd/data/wangshaobo/conda_envs/llama_factory/bin/ds_report"
if [ -f "$DEEPSPEED_DS_REPORT" ]; then
    DS_ENV_REPORT=$("$DEEPSPEED_DS_REPORT" 2>/dev/null)
else
    DS_ENV_REPORT=""
fi
if [ $? -eq 0 ]; then
    check_passed "DeepSpeed环境报告生成成功"
    
    # 检查CUDA ops编译状态
    if echo "$DS_ENV_REPORT" | grep -q "torch_cuda_ops_compatible"; then
        check_passed "DeepSpeed CUDA ops兼容"
    else
        check_warning "DeepSpeed CUDA ops可能不兼容，可能影响性能"
    fi
else
    check_warning "无法生成DeepSpeed环境报告，ds_report命令失败"
fi

# 6. 测试简单的GPU计算
echo ""
echo "6. 测试GPU计算能力..."

GPU_TEST_RESULT=$("$LLAMA_FACTORY_PYTHON" -c "
import torch
if torch.cuda.is_available():
    try:
        # 创建简单的tensor并移到GPU
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print('GPU矩阵运算测试成功')
        print(f'结果shape: {z.shape}')
        
        # 测试多GPU
        if torch.cuda.device_count() > 1:
            x1 = torch.randn(100, 100).cuda(0)
            x2 = torch.randn(100, 100).cuda(1)
            print('多GPU访问测试成功')
    except Exception as e:
        print(f'GPU计算测试失败: {e}')
        exit(1)
else:
    print('CUDA不可用')
    exit(1)
" 2>/dev/null)

if [ $? -eq 0 ]; then
    check_passed "GPU计算测试成功"
    echo "$GPU_TEST_RESULT" | while read line; do
        check_info "  $line"
    done
else
    check_failed "GPU计算测试失败"
    echo "GPU计算测试错误:" >> $ERROR_LOG
    "$LLAMA_FACTORY_PYTHON" -c "
import torch
x = torch.randn(100, 100).cuda()
" >> $ERROR_LOG 2>&1
fi

# 7. 检查内存和存储
echo ""
echo "7. 检查系统内存和存储..."

# 检查系统内存
TOTAL_RAM=$(free -g | awk '/^Mem:/{print $2}')
AVAILABLE_RAM=$(free -g | awk '/^Mem:/{print $7}')

check_info "系统总内存: ${TOTAL_RAM}GB"
check_info "可用内存: ${AVAILABLE_RAM}GB"

if [ $AVAILABLE_RAM -ge 50 ]; then
    check_passed "系统内存充足"
else
    check_warning "系统内存可能不足，建议至少50GB可用内存"
fi

# 8. 检查网络连接（用于模型下载）
echo ""
echo "8. 检查网络连接..."

if ping -c 1 huggingface.co >/dev/null 2>&1; then
    check_passed "HuggingFace Hub连接正常"
else
    check_warning "无法连接到HuggingFace Hub，可能需要设置代理"
fi

# 9. 运行llamafactory-cli验证
echo ""
echo "9. 运行LLaMA-Factory命令验证..."

# 测试llamafactory-cli help
LLAMA_FACTORY_CLI="/mnt/public/gpfs-jd/data/wangshaobo/conda_envs/llama_factory/bin/llamafactory-cli"
if [ -f "$LLAMA_FACTORY_CLI" ] && [ -x "$LLAMA_FACTORY_CLI" ]; then
    LLAMAFACTORY_HELP=$("$LLAMA_FACTORY_CLI" --help 2>/dev/null)
    if [ $? -eq 0 ]; then
        check_passed "llamafactory-cli命令可用"
    else
        check_failed "llamafactory-cli命令不可用"
        echo "llamafactory-cli错误:" >> $ERROR_LOG
        "$LLAMA_FACTORY_CLI" --help >> $ERROR_LOG 2>&1
    fi
else
    check_failed "llamafactory-cli不存在: $LLAMA_FACTORY_CLI"
fi

# 10. 生成详细的系统信息报告
echo ""
echo "10. 生成系统信息报告..."

SYSTEM_INFO_FILE="/tmp/gpu_system_info.txt"
{
    echo "=== 系统信息报告 ==="
    echo "生成时间: $(date)"
    echo ""
    echo "=== 硬件信息 ==="
    echo "CPU信息:"
    lscpu | grep "Model name" || echo "无法获取CPU信息"
    echo ""
    echo "内存信息:"
    free -h
    echo ""
    echo "GPU信息:"
    nvidia-smi || echo "无法获取GPU信息"
    echo ""
    echo "=== 软件环境 ==="
    echo "操作系统:"
    uname -a
    echo ""
    echo "Python版本 (llama_factory环境):"
    "$LLAMA_FACTORY_PYTHON" --version 2>/dev/null || echo "无法获取Python版本"
    echo ""
    echo "关键包版本:"
    for pkg in torch transformers datasets accelerate deepspeed peft; do
        version=$("$LLAMA_FACTORY_PYTHON" -c "import $pkg; print('$pkg:', $pkg.__version__)" 2>/dev/null)
        if [ $? -eq 0 ]; then
            echo "$version"
        else
            echo "$pkg: 未安装或有问题"
        fi
    done
    echo ""
    echo "=== CUDA信息 ==="
    echo "nvcc版本:"
    nvcc --version 2>/dev/null || echo "nvcc不可用"
    echo ""
    echo "PyTorch CUDA信息:"
    "$LLAMA_FACTORY_PYTHON" -c "
import torch
print('PyTorch版本:', torch.__version__)
print('CUDA可用:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA版本:', torch.version.cuda)
    print('cuDNN版本:', torch.backends.cudnn.version())
    print('GPU数量:', torch.cuda.device_count())
" 2>/dev/null || echo "无法获取PyTorch CUDA信息"
} > $SYSTEM_INFO_FILE

check_passed "系统信息报告已生成: $SYSTEM_INFO_FILE"

# 总结和建议
echo ""
echo "========================================"
echo "GPU环境验证完成！"
echo "========================================"
echo -e "${GREEN}通过检查: $PASSED${NC}"
echo -e "${YELLOW}警告: $WARNINGS${NC}"
echo -e "${RED}失败检查: $FAILED${NC}"

echo ""
echo "详细报告文件:"
echo "  - 错误日志: $ERROR_LOG"
echo "  - 系统信息: $SYSTEM_INFO_FILE"

if [ $FAILED -eq 0 ]; then
    if [ $WARNINGS -eq 0 ]; then
        echo -e "${GREEN}🎉 GPU环境完全正常，可以开始GPU训练！${NC}"
        echo ""
        echo "建议的下一步:"
        echo "1. 运行 bash slm.sh 开始第一阶段训练"
        echo "2. 训练完成后运行 bash qwen2.5.sh 进行剪枝实验"
        exit 0
    else
        echo -e "${YELLOW}⚠️  GPU环境基本正常，但有一些警告需要注意${NC}"
        echo ""
        echo "可以尝试开始训练，但建议先解决警告项以获得最佳性能"
        exit 1
    fi
else
    echo -e "${RED}❌ GPU环境存在严重问题，请修复失败项后再运行训练${NC}"
    echo ""
    echo "常见问题解决方案:"
    echo "1. NVIDIA驱动问题: 重新安装或更新NVIDIA驱动"
    echo "2. CUDA问题: 检查CUDA安装和环境变量"
    echo "3. PyTorch CUDA问题: 重新安装支持CUDA的PyTorch版本"
    echo "4. 包缺失问题: 在对应conda环境中安装缺失的包"
    echo ""
    echo "详细错误信息请查看: $ERROR_LOG"
    exit 2
fi
