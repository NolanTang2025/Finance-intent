#!/bin/bash
# 启动可视化网页服务器（Shell版本）

cd "$(dirname "$0")"

if [ ! -f "intent_visualizer.html" ]; then
    echo "❌ 错误: 找不到 intent_visualizer.html 文件"
    exit 1
fi

PORT=8000
URL="http://localhost:${PORT}/intent_visualizer.html"

echo "============================================================"
echo "🚀 可视化服务器已启动！"
echo "============================================================"
echo "📊 访问地址: ${URL}"
echo "📁 工作目录: $(pwd)"
echo ""
echo "💡 使用说明:"
echo "   1. 在网页中点击 '选择意图分析结果文件' 按钮"
echo "   2. 选择你的 JSON 结果文件（如 intent_result_*.json）"
echo "   3. 系统会自动加载并可视化数据"
echo ""
echo "按 Ctrl+C 停止服务器"
echo "============================================================"

# 尝试自动打开浏览器
if command -v open > /dev/null; then
    # macOS
    open "$URL" &
elif command -v xdg-open > /dev/null; then
    # Linux
    xdg-open "$URL" &
elif command -v start > /dev/null; then
    # Windows
    start "$URL" &
fi

# 启动Python HTTP服务器
python3 -m http.server $PORT

