# 用户意图分析系统

基于 Gemini AI 的金融信用卡行业用户意图分析系统，通过分析用户连续的有效行为数据来判断用户某个阶段的意图。

## 功能特点

- 按用户维度分析连续的有效行为数据
- 使用 Google Gemini AI 进行意图识别
- 专门适配金融信用卡行业的行为数据
- 支持会话分组和意图演变分析
- 提供意图得分和下一步行为预测

## 安装依赖

```bash
pip install -r requirements.txt
```

## 配置

1. 获取 Google Gemini API 密钥：访问 [Google AI Studio](https://makersuite.google.com/app/apikey)

2. 设置环境变量：
```bash
export GEMINI_API_KEY='your-api-key-here'
```

或者在代码中直接设置：
```python
import os
os.environ['GEMINI_API_KEY'] = 'your-api-key-here'
```

## 使用方法

### 可视化结果（推荐）

最简单的方式是使用可视化网页查看分析结果：

#### 方法1: 使用Python脚本启动（推荐）

```bash
python3 start_visualizer.py
```

脚本会自动：
- 启动本地服务器（端口8000）
- 自动打开浏览器
- 显示可视化页面

#### 方法2: 使用Shell脚本启动

```bash
./start_visualizer.sh
```

#### 方法3: 手动启动

```bash
# 启动HTTP服务器
python3 -m http.server 8000

# 然后在浏览器中访问
# http://localhost:8000/intent_visualizer.html
```

#### 使用步骤

1. 启动服务器后，浏览器会自动打开可视化页面
2. 点击页面上的 **"选择意图分析结果文件"** 按钮
3. 选择你的JSON结果文件（如 `intent_result_*.json`）
4. 系统会自动加载并显示：
   - 📊 统计概览（总用户数、意图节点数、平均把握度等）
   - 📈 各类图表（意图类别分布、把握度分布、信任度分布等）
   - 👤 每个用户的详细意图分析
   - 🔄 意图演变路径可视化（时间线图、网络图）

#### 功能特点

- ✅ 中英文双语切换
- ✅ 交互式图表展示
- ✅ 导出功能（完整页面、图表、统计面板）
- ✅ 用户意图路径可视化
- ✅ 响应式设计，支持各种屏幕尺寸

### 基本使用

```python
from intent_analyzer import IntentAnalyzer
import os

# 设置API密钥
api_key = os.getenv('GEMINI_API_KEY')

# 创建分析器
analyzer = IntentAnalyzer(api_key)

# 分析单个用户
results = analyzer.analyze_user_intent(
    csv_path='data.csv',
    user_uuid='58815058a8944719b07726cb6fa492d1',
    session_timeout_minutes=30
)

# 分析所有用户
results = analyzer.analyze_user_intent(
    csv_path='data.csv',
    session_timeout_minutes=30
)
```

### 运行示例

```bash
python example_usage.py
```

## 数据格式

CSV文件应包含以下列：
- `user_uuid`: 用户唯一标识
- `approved_time`: 审批时间
- `first_payment_time`: 首次支付时间
- `event_time`: 事件发生时间
- `event_name`: 事件名称（如 show_home_page, click_xxx 等）
- `extra_info`: 额外信息（如营销活动信息）

## 意图类别

系统可以识别以下金融信用卡行业特定的意图类型：

1. **支付意图**: 用户想要进行支付交易
2. **额度管理意图**: 用户关注可用额度
3. **分期意图**: 用户想要使用分期付款
4. **优惠券/会员意图**: 用户寻求优惠或会员权益
5. **营销活动参与意图**: 用户对营销活动感兴趣
6. **探索/浏览意图**: 用户正在探索功能

## 输出格式

分析结果以JSON格式返回，包含：

```json
{
  "user_uuid": "用户ID",
  "total_sessions": 会话数量,
  "sessions": [
    {
      "intent": "用户的主要意图描述",
      "intent_category": "意图类别",
      "confidence_score": 0.0-1.0,
      "key_behaviors": ["关键行为1", "关键行为2"],
      "reasoning": "分析推理过程",
      "next_action_prediction": "预测用户下一步可能的行为",
      "session_index": 会话索引,
      "session_size": 会话中的行为数量,
      "timestamp": "分析时间戳"
    }
  ]
}
```

## 参数说明

- `session_timeout_minutes`: 会话超时时间（分钟），默认30分钟。如果两次行为之间的时间间隔超过此值，将视为新的会话。

## 注意事项

1. Gemini API 有调用频率限制，大量用户分析时建议使用批量处理并添加延迟
2. 分析结果的质量取决于行为数据的完整性和准确性
3. 建议先用小样本测试，确认效果后再进行大规模分析

