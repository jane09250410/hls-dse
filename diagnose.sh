#!/bin/bash
# ============================================================
# 诊断脚本：检查你现有 PA-DSE 框架的状态
# ============================================================
# 使用方法：
#   cd ~/hls-dse
#   bash diagnose.sh
#
# 然后把输出全部复制发给我
# ============================================================

echo "=========================================="
echo "  PA-DSE 项目诊断"
echo "=========================================="
echo ""

# 1. 当前目录
echo "=== 1. 当前目录 ==="
pwd
echo ""

# 2. 项目整体结构
echo "=== 2. 项目目录结构 ==="
find . -maxdepth 3 -type f \( -name "*.py" -o -name "*.c" -o -name "*.sh" \) | sort
echo ""

# 3. benchmarks 目录
echo "=== 3. benchmarks/ 内容 ==="
ls -la benchmarks/*/
echo ""

# 4. scripts 目录
echo "=== 4. scripts/ 内容 ==="
ls -la scripts/
echo ""

# 5. results 目录
echo "=== 5. results/ 结构 ==="
if [ -d "results" ]; then
    find results -maxdepth 3 -type d | head -30
    echo "..."
    echo "result 文件数量:"
    find results -type f | wc -l
else
    echo "results/ 目录不存在"
fi
echo ""

# 6. 关键文件内容：run_pa_dse.py
echo "=== 6. run_pa_dse.py 内容 ==="
if [ -f "scripts/run_pa_dse.py" ]; then
    cat scripts/run_pa_dse.py
else
    echo "文件不存在: scripts/run_pa_dse.py"
    echo "查找可能的替代文件:"
    find . -name "*pa_dse*" -o -name "*run*dse*" -o -name "*exploration*" | head -10
fi
echo ""

# 7. pa_dse.py 主流程
echo "=== 7. pa_dse.py 内容 ==="
if [ -f "scripts/pa_dse.py" ]; then
    cat scripts/pa_dse.py
else
    echo "文件不存在: scripts/pa_dse.py"
fi
echo ""

# 8. config_generator.py
echo "=== 8. config_generator.py 内容 ==="
if [ -f "scripts/config_generator.py" ]; then
    cat scripts/config_generator.py
else
    echo "文件不存在: scripts/config_generator.py"
fi
echo ""

# 9. feasibility_filter.py
echo "=== 9. feasibility_filter.py 内容 ==="
if [ -f "scripts/feasibility_filter.py" ]; then
    cat scripts/feasibility_filter.py
else
    echo "文件不存在: scripts/feasibility_filter.py"
fi
echo ""

# 10. pattern_learner.py
echo "=== 10. pattern_learner.py 内容 ==="
if [ -f "scripts/pattern_learner.py" ]; then
    cat scripts/pattern_learner.py
else
    echo "文件不存在: scripts/pattern_learner.py"
fi
echo ""

# 11. run_exploration.py 或 compare_strategies.py
echo "=== 11. run_exploration.py 内容 ==="
if [ -f "scripts/run_exploration.py" ]; then
    cat scripts/run_exploration.py
else
    echo "文件不存在: scripts/run_exploration.py"
fi
echo ""

echo "=== 12. compare_strategies.py 内容 ==="
if [ -f "scripts/compare_strategies.py" ]; then
    cat scripts/compare_strategies.py
else
    echo "文件不存在: scripts/compare_strategies.py"
fi
echo ""

# 12. heuristic.py
echo "=== 13. heuristic.py 内容 ==="
if [ -f "scripts/heuristic.py" ]; then
    cat scripts/heuristic.py
else
    echo "文件不存在: scripts/heuristic.py"
fi
echo ""

# 13. analyze.py
echo "=== 14. analyze.py 内容 ==="
if [ -f "scripts/analyze.py" ]; then
    cat scripts/analyze.py
else
    echo "文件不存在: scripts/analyze.py"
fi
echo ""

# 14. Bambu 是否可用
echo "=== 15. Bambu 检查 ==="
if command -v bambu &> /dev/null; then
    echo "bambu 路径: $(which bambu)"
    bambu --version 2>&1 | head -3
else
    echo "bambu 不在 PATH 中"
    echo "查找 bambu:"
    find / -name "bambu" -type f 2>/dev/null | head -5
fi
echo ""

# 15. Python 版本
echo "=== 16. Python 检查 ==="
python3 --version 2>&1
echo ""

# 16. 已有的实验结果样例
echo "=== 17. 已有结果文件样例 ==="
SAMPLE=$(find results -name "*.csv" -type f 2>/dev/null | head -1)
if [ -n "$SAMPLE" ]; then
    echo "样例文件: $SAMPLE"
    echo "前 5 行:"
    head -5 "$SAMPLE"
else
    echo "没有找到 CSV 结果文件"
    SAMPLE2=$(find results -name "*.json" -type f 2>/dev/null | head -1)
    if [ -n "$SAMPLE2" ]; then
        echo "找到 JSON 文件: $SAMPLE2"
        echo "前 20 行:"
        head -20 "$SAMPLE2"
    fi
fi
echo ""

echo "=========================================="
echo "  诊断完成！请把以上全部输出复制发给我"
echo "=========================================="
