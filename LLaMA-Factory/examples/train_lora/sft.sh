#!/bin/bash
#SBATCH --job-name=data_token            # 作业名
#SBATCH --output=log/output.log         # 标准输出日志
#SBATCH --error=log/error.log           # 错误日志
#SBATCH --partition=SLLab-own         # 分区名，根据集群配置而定
#SBATCH --nodes=1                   # 节点数
#SBATCH --cpus-per-task=30           # 每个任务使用的CPU数
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=300GB                    # 内存

# 执行你的脚本或命令
bash /net/scratch/zhaorun/shaobo/Data_Token_Pruning/LLaMA-Factory/examples/train_lora/llama3_instruct_template.sh