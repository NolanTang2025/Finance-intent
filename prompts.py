from pydantic import BaseModel, Field

INTENT_SEGMENTATION = """
You are a user behavior analysis expert in the financial credit card industry. Please analyze the following user behaviors and segment them into different intent phases based on intent consistency.

## Task Description

Analyze the behavior sequence and identify when the user's intent changes. Segment behaviors into different phases where each phase represents a consistent intent.

## Intent Consistency Criteria

Behaviors should be grouped into the same phase if they:
1. **Share the same primary intent**: All behaviors in a phase should serve the same goal (e.g., all related to payment, all related to credit limit checking, all related to voucher usage)
2. **Form a coherent sequence**: Behaviors flow logically toward the same objective
3. **Show continuous focus**: No significant shift in user's attention or goal

## Intent Change Indicators

A new intent phase should start when:
1. **Clear intent shift**: User switches from one goal to another (e.g., from browsing vouchers to checking credit limit)
2. **Different product/service focus**: Behaviors shift to a different financial product or service
3. **Significant context change**: User moves from one functional area to another (e.g., from payment flow to membership page)
4. **Time gap with intent change**: Long time gap combined with different behavior pattern

## Segmentation Rules

- Each segment should contain at least 2-3 behaviors (unless it's a clear isolated intent)
- Overlapping behaviors are allowed if they serve as transition points
- Consider behavior context and sequence, not just event names
- A segment should represent a complete intent phase, not just a single action

## Input Data

User behavior list (total {{actions_count}} behaviors):
{{actions_text}}

## Your Task

Analyze the behavior sequence and segment it into different intent phases. Each phase should represent behaviors with consistent intent.
"""


class IntentSegment(BaseModel):
    segment_index: int = Field(description="sequential number starting from 0")
    start_index: int = Field(
        description="first behavior index in this segment (inclusive). Examples: 0, 6"
        ) 
    end_index: int = Field(
        description="last behavior index in this segment (inclusive). Examples: 5, 12"
        )
    intent_description: str = Field(
        description="brief description of the intent for this segment (in Chinese). Examples: 'User is exploring voucher options and selecting payment method', 'User is checking credit limit and available balance'"
             )
    behavior_indices: list[int] = Field(
        description="list of all behavior indices in this segment (should be consecutive). Examples: [0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11, 12]"
        )

class IntentSegmentationOutput(BaseModel):
    intent_segments: list[IntentSegment]


INTENT_ANALYSIS = """你是一位金融信用卡行业的用户行为分析专家。请综合分析所有输入信息来提取用户意图。

## 分析数据源（请使用所有数据源）

1. **用户信息**: 用户ID、审批时间、首次支付时间 → 了解用户状态和生命周期阶段

2. **用户行为序列**: 按时间顺序的行为事件 → 了解用户实际做了什么

3. **行为上下文**: 事件类型、时间间隔、额外信息 → 了解行为深度和模式

4. **历史意图**: 之前的意图分析结果（如果存在）→ 了解意图演变

## 埋点类型说明（重要：理解用户主动行为vs被动展示）

用户行为埋点分为三类，**关键理解**：只有用户的主动回应才反映真实意图：

1. **show曝光类**: 页面或内容展示给用户的事件
   - 特征: 事件名通常以 `show_` 开头
   - **重要理解**: show操作**没有任何含义**，只是商家/系统展现给用户一些信息，这是**被动展示**，不代表用户有任何意图或兴趣
   - **分析原则**: 
     - **单独的show操作不能作为意图判断的依据**
     - show只是信息展示，用户可能根本没注意到或不在意
     - 只有后续出现用户的主动回应（click或on_app_stop），才能说明用户对show的内容有反应
   - 权重: **极低权重**，仅作为上下文参考，不能单独用于判断意图

2. **click点击类**: 用户主动点击操作的事件
   - 特征: 事件名通常以 `click_` 开头
   - **重要理解**: 这是用户的**主动回应**，表示用户对show的内容产生了兴趣并采取了行动
   - 含义: 用户主动选择或操作，表示明确的兴趣和意图
   - **分析原则**:
     - click是判断用户意图的**核心证据**
     - 需要结合前面的show来理解用户点击了什么内容
     - show + click 的组合才能完整反映用户意图（看到了什么 + 点击了什么）
   - 权重: **高权重**，是判断用户意图的重要信号

3. **on_app_stop杀死app进程类**: 用户关闭或退出应用的事件
   - 特征: 事件名包含 `on_app_stop` 或类似系统级事件
   - **重要理解**: 这是用户的**主动停止进程**，是用户明确的回应行为
   - **关键分析点**: **推断用户在这步之前看到了什么非常重要**
     - on_app_stop是用户对之前看到的内容做出的回应
     - 必须分析on_app_stop之前最近的show操作序列
     - 了解用户在看到什么内容后选择了停止
   - 含义: 用户主动结束使用，可能表示：
     - **对当前展示内容不感兴趣**（负面回应）：看到show后立即停止
     - **已完成操作，退出应用**：完成支付/操作后正常退出
     - **遇到问题或困惑**：看到复杂内容后选择退出
     - **其他原因退出**
   - **分析原则**:
     - **必须追溯on_app_stop之前的show序列**：用户看到了什么 → 选择了停止
     - 如果show后立即出现on_app_stop，强烈暗示用户对show的内容不感兴趣或感到困惑
     - 如果show → click → on_app_stop，可能是完成操作后正常退出
     - 如果多个show后on_app_stop，需要分析用户对哪些内容做出了"停止"的回应
     - **重点**: 分析用户停止前最后看到的内容，这能揭示用户的真实反应
   - 权重: 中等权重，用于判断用户对展示内容的反应，但必须结合之前的show序列分析

**核心分析逻辑**:
- **show操作本身没有含义**，只是商家展示信息
- **只有用户使用click点击或on_app_stop杀死app进程来回应show，才算是有意义的用户行为**
- 分析意图时，必须关注：show了什么 → 用户如何回应（click还是on_app_stop）
- 如果只有show没有后续回应，不能判断用户意图
- show + click 的组合才是完整的用户意图表达

## 数据清洗规则说明

在分析时，请注意以下数据清洗规则，这些信息已写入 `extra_info` 字段：

1. **券（Voucher）埋点清洗**:
   - 券相关的埋点（如 `show_voucher_xxx`, `click_myvoucher_xxx` 等）会清洗ID
   - `extra_info` 字段包含券的名称信息，包括：
     - 支付方式（如：虚拟账户支付、二维码支付等）
     - 优惠类型（如：满减券、折扣券、新人专享券等）
     - 其他券的详细信息
   - **分析要点**: 通过 `extra_info` 了解用户关注的券类型和支付方式，判断用户对优惠和支付方式的偏好

2. **弹窗（Popup）埋点清洗**:
   - 弹窗相关的埋点（如 `click_fullpopup_pribtn_xxx`, `show_new_homebanner_xxx` 等）会清洗ID
   - `extra_info` 字段包含弹窗内容的类型分类，包括：
     - 开卡礼类型（如：新人开卡礼、首刷礼等）
     - 促留存类型（如：任务奖励、限时活动等）
     - 营销活动类型（如：推广活动、会员权益等）
     - 其他弹窗分类信息
   - **分析要点**: 通过 `extra_info` 了解用户对哪些类型的营销活动感兴趣，判断用户的参与动机和意图

**重要提示**: 在分析用户意图时，务必关注 `extra_info` 字段中的详细信息，这些信息能帮助你更准确地理解用户关注的具体内容、优惠类型和活动类型，从而做出更精准的意图判断。

## 用户行为信号权重（基于主动回应原则）

**高权重**（明确兴趣 - 用户主动回应）:
- **click_xxx（点击操作）**: 用户主动选择，是判断意图的**核心证据**
  - 必须结合前面的show来理解：用户点击了什么内容
  - show + click 组合 = 完整的用户意图表达
- **click_fullpopup_pribtn_xxx（弹窗主按钮点击）**: 显示用户对营销活动有明确兴趣并主动参与
- **click_pay_checkout_submit_btn_xxx（支付提交按钮点击）**: 明确的支付意图和转化行为

**中权重**（用户主动回应但含义需结合上下文）:
- **on_app_stop（杀死app进程）**: 用户的主动回应，需要结合前面的show判断：
  - 如果show后立即on_app_stop，可能表示不感兴趣（负面回应）
  - 如果完成操作后on_app_stop，可能是正常退出

**极低权重/无效**（被动展示，不能单独判断意图）:
- **show_xxx（页面展示）**: **单独使用无效**，只是商家展示信息
  - 不能单独作为意图判断依据
  - 仅作为上下文参考，需要等待用户的主动回应（click或on_app_stop）
  - 只有与后续的click结合，才能形成有意义的意图信号

**关键分析原则**:
1. **show操作本身没有含义**，只是被动展示
2. **只有用户主动回应（click或on_app_stop）才算有意义**
3. **show + click 的组合**才能完整反映用户意图
4. **单独的show不能判断意图**，必须等待用户回应
5. 分析时关注：商家展示了什么 → 用户如何回应 → 这才是真实意图

## 行为序列分析（对多个行为至关重要）

**顺序很重要！** 序列揭示用户的思考过程：

- 首次 → 最后: 显示兴趣演变（从浏览到支付，从探索到决策）

- 行为类型序列（注意：show只是展示，需要看用户回应）: 
  - **show → click**: 商家展示 → 用户点击回应 = 明确的兴趣和意图
  - **show → on_app_stop**: 商家展示 → 用户退出 = **重要：分析用户看到了什么后选择停止**
    - 必须追溯on_app_stop之前最近的show操作
    - 如果show后立即on_app_stop，强烈暗示用户对show的内容不感兴趣（负面回应）
    - 这能揭示用户对哪些内容做出了"停止"的回应
  - **show → show → on_app_stop**: 多次展示后用户停止 = 分析用户对哪些内容做出了停止决定
  - **show → click → on_app_stop**: 展示 → 点击 → 停止 = 可能完成操作后正常退出
  - **show → show → click**: 多次展示后用户点击 = 用户经过考虑后的选择
  - **只有show没有回应**: 不能判断意图，用户可能根本没注意或不在意
  - **click → click**: 连续的点击操作 = 强烈的参与意图
  - **show支付页面 → click支付按钮**: 显示明确的支付意图和转化行为
  - **show优惠券 → click使用优惠券**: 显示优惠寻求意图
  - **关键序列分析**: 对于on_app_stop，必须分析：
    1. on_app_stop之前最近的show操作是什么
    2. 用户在看到什么内容后选择了停止
    3. 这个停止是对哪些内容的回应
    4. 停止的原因可能是什么（不感兴趣/完成操作/遇到问题等）

- 返回之前的行为: 强烈兴趣或比较锚点

- 每个行为的时间间隔: 哪些行为获得了更多关注

## 金融信用卡行业特定意图类型（需要具体化分析）

对于每个意图，必须明确：
- **具体探索的产品功能**: 用户正在探索哪个具体的产品功能（如：现金借贷、虚拟账户支付、分期付款、额度查询、优惠券使用等）
- **探索目的**: 用户探索这个功能的目的是什么（如：了解如何使用、比较选项、准备首次交易、解决特定需求等）
- **与首次交易的关系**: 这个意图如何帮助用户完成第一笔交易

1. **支付意图**: 用户想要进行支付交易
   - 具体功能: 虚拟账户支付(VA)、二维码支付(QRIS)、手机充值、电商支付等
   - 探索目的: 了解支付流程、选择支付方式、准备完成交易
   - 信号: show_pay_checkout_xxx, click_pay_checkout_submit_btn_xxx, show_paymentpage_xxx
   - 与首次交易: 直接关联，用户正在完成或准备完成第一笔支付
   
2. **额度管理意图**: 用户关注可用额度
   - 具体功能: 查看可用额度、临时额度提升、现金借贷额度、分期额度等
   - 探索目的: 了解可用资金、评估购买能力、准备大额交易
   - 信号: show_limit_xxx, show_limit_page_module_xxx, show_homepage_tempolimit_increase_tooltips
   - 与首次交易: 用户可能在确认是否有足够额度进行首次交易
   
3. **分期意图**: 用户想要使用分期付款
   - 具体功能: 分期付款选项、分期计划详情、分期计算器等
   - 探索目的: 了解分期规则、选择分期期数、降低首次交易压力
   - 信号: show_homepage_installment_section, show_installplandetail_xxx, click_pay_checkout_installment_btn_xxx
   - 与首次交易: 用户可能希望通过分期降低首次交易的门槛
   
4. **优惠券/会员意图**: 用户寻求优惠或会员权益
   - 具体功能: 优惠券查看、优惠券使用、会员权益、新人福利等
   - 探索目的: 寻找优惠、最大化首次交易价值、了解会员特权
   - 信号: show_membership_xxx, show_paymentpage_voucher, click_myvoucher_xxx, show_voucherusage_pg
   - 与首次交易: 用户希望在首次交易时使用优惠，降低交易成本
   
5. **营销活动参与意图**: 用户对营销活动感兴趣
   - 具体功能: 新人开卡礼、限时活动、任务奖励、推广活动等
   - 探索目的: 获取奖励、了解活动规则、参与任务获得福利
   - 信号: click_fullpopup_pribtn_xxx, click_new_homebanner_xxx, show_new_user_zone_page
   - 与首次交易: 用户可能希望通过参与活动获得首次交易的优惠或奖励
   
6. **探索/浏览意图**: 用户正在探索功能
   - 具体功能: 需要明确指出探索的具体功能（如：现金借贷功能、虚拟账户功能、支付方式、额度管理、会员体系等）
   - 探索目的: 了解产品功能、熟悉App操作、寻找适合的支付方式、评估产品价值等
   - 信号: 多个show_xxx但无点击，重复访问，浏览多个功能页面
   - 与首次交易: 用户可能在为首次交易做准备，探索最适合的交易方式

## 历史使用

- 如果存在历史: 基于之前的分析，注意变化或确认一致性
- 如果没有变化: 确认持续的兴趣模式 + 添加任何新见解
- 如果有变化: 突出不同之处及其重要性
- 无论历史如何，始终提供完整的意图分析

## 你的任务

1. **分析用户意图**: 综合USER + ACTIONS + HISTORY的见解
   - 交叉引用用户生命周期阶段（审批时间、首次支付时间）与行为模式
   - 如果多个行为，分析浏览顺序及其揭示的意图
   - 连接行为类型与金融产品/服务的相关性
   - **必须明确指出**: 用户正在探索的具体产品功能是什么
   - **必须分析**: 用户探索这个功能的目的（了解如何使用、准备交易、比较选项等）
   - **必须关联**: 这个意图如何帮助用户完成第一笔交易
   - **特别关注on_app_stop分析（重要）**: 
     - 如果行为序列中包含on_app_stop，这是用户主动停止了进程
     - **必须追溯分析**: 用户在这步之前看到了什么（最近的show操作）
     - **推断用户反应**: 用户在看到什么内容后选择了停止
     - **分析停止原因**: 
       - 如果show后立即on_app_stop：用户对show的内容不感兴趣（负面回应）
       - 如果show → click → on_app_stop：可能是完成操作后正常退出
       - 如果多个show后on_app_stop：分析用户对哪些内容做出了"停止"的回应
     - **揭示用户态度**: 通过分析停止前看到的内容，揭示用户对哪些功能/内容不感兴趣或感到困惑
     - 这能帮助理解用户的真实反应和潜在流失原因
   - 所有分析必须引用输入中的具体证据 - 不要猜测

2. **分析用户心理状态**:
   - **Baseline Trust (基础信任度)**: 用户对产品和服务的信任程度 (0.0-1.0)
     - 高信任(0.7-1.0): 快速激活、积极使用、无反复验证行为
     - 中信任(0.4-0.7): 有探索但谨慎，需要更多信息
     - 低信任(0.0-0.4): 反复查看、犹豫不决、大量验证行为
   - **Concern (担忧点)**: 用户可能担心的问题
     - 安全性担忧: 反复查看安全相关页面
     - 额度担忧: 频繁查看额度、担心不够用
     - 费用担忧: 查看费率、分期成本等
     - 使用难度担忧: 反复查看使用教程、帮助页面
     - 其他具体担忧点
   - **心理参考值 vs 实际感知**: 
     - 用户的心理预期是什么（期望的额度、优惠、功能等）
     - 实际感知到的与预期的差距
     - 这种差距如何影响首次交易决策

3. **计算意图得分和置信度**:
   - **Intent Confidence Score (意图置信度)**: 评估此意图分析的置信度 (0.0-1.0)
     - 0.9-1.0: 非常明确的意图，有明确的转化行为
     - 0.7-0.9: 较强的意图，有明显的兴趣信号
     - 0.5-0.7: 中等意图，有一些兴趣但不够明确
     - 0.3-0.5: 弱意图，主要是浏览行为
     - 0.0-0.3: 意图不明确，行为很少或无效
   - **Certainty Level (确信程度)**: 你对这个意图判断有多确信
     - "Very High": 有明确的转化行为，意图非常清晰
     - "High": 有强烈的兴趣信号，意图比较明确
     - "Medium": 有一些信号，但不够明确
     - "Low": 信号较弱，主要是推测
   - **Evidence Quality (证据质量)**: 支持这个意图的证据质量
     - 强证据: 明确的点击、支付页面访问等
     - 中等证据: 页面展示、浏览模式等
     - 弱证据: 间接信号、推测性证据

4. **提供运营建议**: 基于用户意图和心理状态，为运营人员提供帮助用户完成第一笔交易的建议
   - **线上解决方案**: App内推送、消息提醒、优惠券发放、功能引导等
   - **线下解决方案**: 电话回访、短信提醒、邮件营销、客户经理联系等
   - 建议要具体、可执行，针对用户当前意图、信任度和担忧点

## 输入数据

用户信息:
- 用户ID: {{user_context.get('user_uuid', 'N/A')}}
- 审批时间: {{user_context.get('approved_time', 'N/A')}}
- 首次支付时间: {{user_context.get('first_payment_time', 'N/A')}}
- 首次行为时间: {{user_context.get('first_action_time', 'N/A')}}
- 最后行为时间: {{user_context.get('last_action_time', 'N/A')}}
- 总行为数: {{user_context.get('total_actions', 0)}}
- 唯一事件类型数: {{user_context.get('unique_events', 0)}}

用户行为序列:
{{actions_text}}

{{history_text}}
请开始分析用户意图。"""


INTENT_ONLY_ANALYSIS = """你是一位金融信用卡行业的用户行为分析专家。请综合分析所有输入信息来提取用户意图。

## 分析数据源（请使用所有数据源）

1. **用户信息**: 用户ID、审批时间、首次支付时间 → 了解用户状态和生命周期阶段

2. **用户行为序列**: 按时间顺序的行为事件 → 了解用户实际做了什么

3. **行为上下文**: 事件类型、时间间隔、额外信息 → 了解行为深度和模式

4. **历史意图**: 之前的意图分析结果（如果存在）→ 了解意图演变

## 埋点类型说明（重要：理解用户主动行为vs被动展示）

用户行为埋点分为三类，**关键理解**：只有用户的主动回应才反映真实意图：

1. **show曝光类**: 页面或内容展示给用户的事件
   - 特征: 事件名通常以 `show_` 开头
   - **重要理解**: show操作**没有任何含义**，只是商家/系统展现给用户一些信息，这是**被动展示**，不代表用户有任何意图或兴趣
   - **分析原则**: 
     - **单独的show操作不能作为意图判断的依据**
     - show只是信息展示，用户可能根本没注意到或不在意
     - 只有后续出现用户的主动回应（click或on_app_stop），才能说明用户对show的内容有反应
   - 权重: **极低权重**，仅作为上下文参考，不能单独用于判断意图

2. **click点击类**: 用户主动点击操作的事件
   - 特征: 事件名通常以 `click_` 开头
   - **重要理解**: 这是用户的**主动回应**，表示用户对show的内容产生了兴趣并采取了行动
   - 含义: 用户主动选择或操作，表示明确的兴趣和意图
   - **分析原则**:
     - click是判断用户意图的**核心证据**
     - 需要结合前面的show来理解用户点击了什么内容
     - show + click 的组合才能完整反映用户意图（看到了什么 + 点击了什么）
   - 权重: **高权重**，是判断用户意图的重要信号

3. **on_app_stop杀死app进程类**: 用户关闭或退出应用的事件
   - 特征: 事件名包含 `on_app_stop` 或类似系统级事件
   - **重要理解**: 这是用户的**主动停止进程**，是用户明确的回应行为
   - **关键分析点**: **推断用户在这步之前看到了什么非常重要**
     - on_app_stop是用户对之前看到的内容做出的回应
     - 必须分析on_app_stop之前最近的show操作序列
     - 了解用户在看到什么内容后选择了停止
   - 含义: 用户主动结束使用，可能表示：
     - **对当前展示内容不感兴趣**（负面回应）：看到show后立即停止
     - **已完成操作，退出应用**：完成支付/操作后正常退出
     - **遇到问题或困惑**：看到复杂内容后选择退出
     - **其他原因退出**
   - **分析原则**:
     - **必须追溯on_app_stop之前的show序列**：用户看到了什么 → 选择了停止
     - 如果show后立即出现on_app_stop，强烈暗示用户对show的内容不感兴趣或感到困惑
     - 如果show → click → on_app_stop，可能是完成操作后正常退出
     - 如果多个show后on_app_stop，需要分析用户对哪些内容做出了"停止"的回应
     - **重点**: 分析用户停止前最后看到的内容，这能揭示用户的真实反应
   - 权重: 中等权重，用于判断用户对展示内容的反应，但必须结合之前的show序列分析

**核心分析逻辑**:
- **show操作本身没有含义**，只是商家展示信息
- **只有用户使用click点击或on_app_stop杀死app进程来回应show，才算是有意义的用户行为**
- 分析意图时，必须关注：show了什么 → 用户如何回应（click还是on_app_stop）
- 如果只有show没有后续回应，不能判断用户意图
- show + click 的组合才是完整的用户意图表达

## 数据清洗规则说明

在分析时，请注意以下数据清洗规则，这些信息已写入 `extra_info` 字段：

1. **券（Voucher）埋点清洗**:
   - 券相关的埋点（如 `show_voucher_xxx`, `click_myvoucher_xxx` 等）会清洗ID
   - `extra_info` 字段包含券的名称信息，包括：
     - 支付方式（如：虚拟账户支付、二维码支付等）
     - 优惠类型（如：满减券、折扣券、新人专享券等）
     - 其他券的详细信息
   - **分析要点**: 通过 `extra_info` 了解用户关注的券类型和支付方式，判断用户对优惠和支付方式的偏好

2. **弹窗（Popup）埋点清洗**:
   - 弹窗相关的埋点（如 `click_fullpopup_pribtn_xxx`, `show_new_homebanner_xxx` 等）会清洗ID
   - `extra_info` 字段包含弹窗内容的类型分类，包括：
     - 开卡礼类型（如：新人开卡礼、首刷礼等）
     - 促留存类型（如：任务奖励、限时活动等）
     - 营销活动类型（如：推广活动、会员权益等）
     - 其他弹窗分类信息
   - **分析要点**: 通过 `extra_info` 了解用户对哪些类型的营销活动感兴趣，判断用户的参与动机和意图

**重要提示**: 在分析用户意图时，务必关注 `extra_info` 字段中的详细信息，这些信息能帮助你更准确地理解用户关注的具体内容、优惠类型和活动类型，从而做出更精准的意图判断。

## 用户行为信号权重（基于主动回应原则）

**高权重**（明确兴趣 - 用户主动回应）:
- **click_xxx（点击操作）**: 用户主动选择，是判断意图的**核心证据**
  - 必须结合前面的show来理解：用户点击了什么内容
  - show + click 组合 = 完整的用户意图表达
- **click_fullpopup_pribtn_xxx（弹窗主按钮点击）**: 显示用户对营销活动有明确兴趣并主动参与
- **click_pay_checkout_submit_btn_xxx（支付提交按钮点击）**: 明确的支付意图和转化行为

**中权重**（用户主动回应但含义需结合上下文）:
- **on_app_stop（杀死app进程）**: 用户的主动回应，需要结合前面的show判断：
  - 如果show后立即on_app_stop，可能表示不感兴趣（负面回应）
  - 如果完成操作后on_app_stop，可能是正常退出

**极低权重/无效**（被动展示，不能单独判断意图）:
- **show_xxx（页面展示）**: **单独使用无效**，只是商家展示信息
  - 不能单独作为意图判断依据
  - 仅作为上下文参考，需要等待用户的主动回应（click或on_app_stop）
  - 只有与后续的click结合，才能形成有意义的意图信号

**关键分析原则**:
1. **show操作本身没有含义**，只是被动展示
2. **只有用户主动回应（click或on_app_stop）才算有意义**
3. **show + click 的组合**才能完整反映用户意图
4. **单独的show不能判断意图**，必须等待用户回应
5. 分析时关注：商家展示了什么 → 用户如何回应 → 这才是真实意图

## 行为序列分析（对多个行为至关重要）

**顺序很重要！** 序列揭示用户的思考过程：

- 首次 → 最后: 显示兴趣演变（从浏览到支付，从探索到决策）

- 行为类型序列（注意：show只是展示，需要看用户回应）: 
  - **show → click**: 商家展示 → 用户点击回应 = 明确的兴趣和意图
  - **show → on_app_stop**: 商家展示 → 用户退出 = **重要：分析用户看到了什么后选择停止**
    - 必须追溯on_app_stop之前最近的show操作
    - 如果show后立即on_app_stop，强烈暗示用户对show的内容不感兴趣（负面回应）
    - 这能揭示用户对哪些内容做出了"停止"的回应
  - **show → show → on_app_stop**: 多次展示后用户停止 = 分析用户对哪些内容做出了停止决定
  - **show → click → on_app_stop**: 展示 → 点击 → 停止 = 可能完成操作后正常退出
  - **show → show → click**: 多次展示后用户点击 = 用户经过考虑后的选择
  - **只有show没有回应**: 不能判断意图，用户可能根本没注意或不在意
  - **click → click**: 连续的点击操作 = 强烈的参与意图
  - **show支付页面 → click支付按钮**: 显示明确的支付意图和转化行为
  - **show优惠券 → click使用优惠券**: 显示优惠寻求意图
  - **关键序列分析**: 对于on_app_stop，必须分析：
    1. on_app_stop之前最近的show操作是什么
    2. 用户在看到什么内容后选择了停止
    3. 这个停止是对哪些内容的回应
    4. 停止的原因可能是什么（不感兴趣/完成操作/遇到问题等）

- 返回之前的行为: 强烈兴趣或比较锚点

- 每个行为的时间间隔: 哪些行为获得了更多关注

## 金融信用卡行业特定意图类型（需要具体化分析）

对于每个意图，必须明确：
- **具体探索的产品功能**: 用户正在探索哪个具体的产品功能（如：现金借贷、虚拟账户支付、分期付款、额度查询、优惠券使用等）
- **探索目的**: 用户探索这个功能的目的是什么（如：了解如何使用、比较选项、准备首次交易、解决特定需求等）
- **与首次交易的关系**: 这个意图如何帮助用户完成第一笔交易

1. **支付意图**: 用户想要进行支付交易
   - 具体功能: 虚拟账户支付(VA)、二维码支付(QRIS)、手机充值、电商支付等
   - 探索目的: 了解支付流程、选择支付方式、准备完成交易
   - 信号: show_pay_checkout_xxx, click_pay_checkout_submit_btn_xxx, show_paymentpage_xxx
   - 与首次交易: 直接关联，用户正在完成或准备完成第一笔支付
   
2. **额度管理意图**: 用户关注可用额度
   - 具体功能: 查看可用额度、临时额度提升、现金借贷额度、分期额度等
   - 探索目的: 了解可用资金、评估购买能力、准备大额交易
   - 信号: show_limit_xxx, show_limit_page_module_xxx, show_homepage_tempolimit_increase_tooltips
   - 与首次交易: 用户可能在确认是否有足够额度进行首次交易
   
3. **分期意图**: 用户想要使用分期付款
   - 具体功能: 分期付款选项、分期计划详情、分期计算器等
   - 探索目的: 了解分期规则、选择分期期数、降低首次交易压力
   - 信号: show_homepage_installment_section, show_installplandetail_xxx, click_pay_checkout_installment_btn_xxx
   - 与首次交易: 用户可能希望通过分期降低首次交易的门槛
   
4. **优惠券/会员意图**: 用户寻求优惠或会员权益
   - 具体功能: 优惠券查看、优惠券使用、会员权益、新人福利等
   - 探索目的: 寻找优惠、最大化首次交易价值、了解会员特权
   - 信号: show_membership_xxx, show_paymentpage_voucher, click_myvoucher_xxx, show_voucherusage_pg
   - 与首次交易: 用户希望在首次交易时使用优惠，降低交易成本
   
5. **营销活动参与意图**: 用户对营销活动感兴趣
   - 具体功能: 新人开卡礼、限时活动、任务奖励、推广活动等
   - 探索目的: 获取奖励、了解活动规则、参与任务获得福利
   - 信号: click_fullpopup_pribtn_xxx, click_new_homebanner_xxx, show_new_user_zone_page
   - 与首次交易: 用户可能希望通过参与活动获得首次交易的优惠或奖励
   
6. **探索/浏览意图**: 用户正在探索功能
   - 具体功能: 需要明确指出探索的具体功能（如：现金借贷功能、虚拟账户功能、支付方式、额度管理、会员体系等）
   - 探索目的: 了解产品功能、熟悉App操作、寻找适合的支付方式、评估产品价值等
   - 信号: 多个show_xxx但无点击，重复访问，浏览多个功能页面
   - 与首次交易: 用户可能在为首次交易做准备，探索最适合的交易方式

## 历史使用

- 如果存在历史: 基于之前的分析，注意变化或确认一致性
- 如果没有变化: 确认持续的兴趣模式 + 添加任何新见解
- 如果有变化: 突出不同之处及其重要性
- 无论历史如何，始终提供完整的意图分析

## 你的任务

1. **分析用户意图**: 综合USER + ACTIONS + HISTORY的见解
   - 交叉引用用户生命周期阶段（审批时间、首次支付时间）与行为模式
   - 如果多个行为，分析浏览顺序及其揭示的意图
   - 连接行为类型与金融产品/服务的相关性
   - **必须明确指出**: 用户正在探索的具体产品功能是什么
   - **必须分析**: 用户探索这个功能的目的（了解如何使用、准备交易、比较选项等）
   - **必须关联**: 这个意图如何帮助用户完成第一笔交易
   - **特别关注on_app_stop分析（重要）**: 
     - 如果行为序列中包含on_app_stop，这是用户主动停止了进程
     - **必须追溯分析**: 用户在这步之前看到了什么（最近的show操作）
     - **推断用户反应**: 用户在看到什么内容后选择了停止
     - **分析停止原因**: 
       - 如果show后立即on_app_stop：用户对show的内容不感兴趣（负面回应）
       - 如果show → click → on_app_stop：可能是完成操作后正常退出
       - 如果多个show后on_app_stop：分析用户对哪些内容做出了"停止"的回应
     - **揭示用户态度**: 通过分析停止前看到的内容，揭示用户对哪些功能/内容不感兴趣或感到困惑
     - 这能帮助理解用户的真实反应和潜在流失原因
   - 所有分析必须引用输入中的具体证据 - 不要猜测

2. **分析用户心理状态**:
   - **Baseline Trust (基础信任度)**: 用户对产品和服务的信任程度 (0.0-1.0)
     - 高信任(0.7-1.0): 快速激活、积极使用、无反复验证行为
     - 中信任(0.4-0.7): 有探索但谨慎，需要更多信息
     - 低信任(0.0-0.4): 反复查看、犹豫不决、大量验证行为
   - **Concern (担忧点)**: 用户可能担心的问题
     - 安全性担忧: 反复查看安全相关页面
     - 额度担忧: 频繁查看额度、担心不够用
     - 费用担忧: 查看费率、分期成本等
     - 使用难度担忧: 反复查看使用教程、帮助页面
     - 其他具体担忧点
   - **心理参考值 vs 实际感知**: 
     - 用户的心理预期是什么（期望的额度、优惠、功能等）
     - 实际感知到的与预期的差距
     - 这种差距如何影响首次交易决策

3. **计算意图得分和置信度**:
   - **Intent Confidence Score (意图置信度)**: 评估此意图分析的置信度 (0.0-1.0)
     - 0.9-1.0: 非常明确的意图，有明确的转化行为
     - 0.7-0.9: 较强的意图，有明显的兴趣信号
     - 0.5-0.7: 中等意图，有一些兴趣但不够明确
     - 0.3-0.5: 弱意图，主要是浏览行为
     - 0.0-0.3: 意图不明确，行为很少或无效
   - **Certainty Level (确信程度)**: 你对这个意图判断有多确信
     - "Very High": 有明确的转化行为，意图非常清晰
     - "High": 有强烈的兴趣信号，意图比较明确
     - "Medium": 有一些信号，但不够明确
     - "Low": 信号较弱，主要是推测
   - **Evidence Quality (证据质量)**: 支持这个意图的证据质量
     - 强证据: 明确的点击、支付页面访问等
     - 中等证据: 页面展示、浏览模式等
     - 弱证据: 间接信号、推测性证据

## 输入数据

用户信息:
- 用户ID: {{user_context.get('user_uuid', 'N/A')}}
- 审批时间: {{user_context.get('approved_time', 'N/A')}}
- 首次支付时间: {{user_context.get('first_payment_time', 'N/A')}}
- 首次行为时间: {{user_context.get('first_action_time', 'N/A')}}
- 最后行为时间: {{user_context.get('last_action_time', 'N/A')}}
- 总行为数: {{user_context.get('total_actions', 0)}}
- 唯一事件类型数: {{user_context.get('unique_events', 0)}}

用户行为序列:
{{actions_text}}

{{history_text}}

请开始分析用户意图（注意：本次分析不包含运营建议）。"""

class Concern(BaseModel):
    concern_type: str = Field(description="Type of concern. Examples: Security, Credit, Limit, Fees, Usage, Difficulty, Other")
    concern_description: str = Field(description="Specific concern description (in Chinese)")
    concern_severity: str = Field(description="Severity of the concern. Examples: High, Medium, Low")
    evidence: list[str] = Field(description="Behavior evidence list")

class PsychologicalReference(BaseModel):
    expected_value: str = Field(description="What user expects (in Chinese)")
    perceived_value: str = Field(description="What user actually perceives (in Chinese)")
    gap_analysis: str = Field(description="Gap between expected and perceived, and its impact on first transaction (in Chinese)")

class OperationRecommendation(BaseModel):
    online_solutions: list[str] = Field(description="List of online solutions (in Chinese)")
    offline_solutions: list[str] = Field(description="List of offline solutions (in Chinese)")
    priority: str = Field(description="Priority of operation recommendation. Examples: High, Medium, Low")
    targeted_message: str = Field(description="Specific message or intervention tailored to user's trust level and concerns (in Chinese)")


class IntentOnlyAnalysisOutput(BaseModel):
    intent: str = Field(description="User's main intent description (in Chinese, must be specific about what product feature they are exploring)")
    intent_category: str = Field(description="Intent category (payment_intent/credit_limit_intent/installment_intent/voucher_intent/marketing_intent/exploration_intent)")
    confidence_score: float = Field(description="A float number between 0.0-1.0")
    certainty_level: str = Field(description="How certain you are about this intent. Examples: Very High, High, Medium, Low")
    evidence_quality: str = Field(description="Quality of evidence supporting this intent. Examples: Strong, Medium, Weak")
    explored_feature: str = Field(description="Specific product feature the user is exploring (in Chinese). Examples: 虚拟账户支付, 现金借贷, 分期付款")
    exploration_purpose: str = Field(description="Purpose of exploration (in Chinese). Examples: 了解如何使用, 准备首次交易, 比较支付选项")
    first_transaction_connection: str = Field(description="How this intent helps user complete first transaction (in Chinese)")
    baseline_trust: float = Field(description="A float number between 0.0-1.0 representing user's baseline trust in the product/service")
    trust_indicators: list[str] = Field(description="List of trust indicators (in Chinese)")
    concerns: list[Concern] = Field(description="List of concerns")
    psychological_reference: PsychologicalReference = Field(description="Psychological reference")
    key_behaviors: list[str] = Field()
    reasoning: str = Field(description="Analysis reasoning process (in Chinese, detailed explanation including: what feature, why exploring, trust level, concerns, psychological factors, how it connects to first transaction)")
    next_action_prediction: str = Field(description="Predicted next possible user action (in Chinese)")

class IntentAnalysisOutput(IntentOnlyAnalysisOutput):
    operation_recommendation: OperationRecommendation = Field(description="Operation recommendation")


OPERATION_RECOMMENDATION = """你是一位金融信用卡行业的运营专家。请基于以下用户意图分析结果，为运营人员提供帮助用户完成第一笔交易的建议。

## 用户意图分析结果

**意图**: {{intent_result.get('intent', 'N/A')}}
**意图类别**: {{intent_result.get('intent_category', 'N/A')}}
**置信度**: {{intent_result.get('confidence_score', 0.0)}}
**探索的功能**: {{intent_result.get('explored_feature', 'N/A')}}
**探索目的**: {{intent_result.get('exploration_purpose', 'N/A')}}
**基础信任度**: {{intent_result.get('baseline_trust', 0.0)}}
**担忧点**: {{intent_result.get('concerns', [])}}
**心理参考值**: {{intent_result.get('psychological_reference', {})}
**关键行为**: {{intent_result.get('key_behaviors', [])}
**推理过程**: {{intent_result.get('reasoning', 'N/A')}}

## 你的任务

基于用户意图和心理状态，为运营人员提供帮助用户完成第一笔交易的建议：

1. **线上解决方案**: App内推送、消息提醒、优惠券发放、功能引导等
2. **线下解决方案**: 电话回访、短信提醒、邮件营销、客户经理联系等
3. **建议要具体、可执行**，针对用户当前意图、信任度和担忧点
4. **优先级判断**: 基于用户完成首次交易的准备程度（High/Medium/Low）
5. **针对性消息**: 针对用户信任度和担忧点的具体干预消息

请开始生成运营建议。"""


class OperationRecommendationOutput(BaseModel):
    operation_recommendation: OperationRecommendation = Field(description="Operation recommendation")


VALID_ACTIONS_FILTER = """You are a user behavior analysis expert in the financial credit card industry. Please analyze the following user behavior data and determine which behaviors are "valid", i.e., meaningful for analyzing user intent.

## Definition of Valid Behaviors

In the financial credit card industry, valid behaviors should:
1. **Reflect user's true intent**: User's active operations or content they focus on
2. **Have value for intent analysis**: Help understand what the user wants to do
3. **Have business significance**: Related to financial products, services, or features

## Characteristics of Invalid Behaviors

The following types of behaviors are usually considered invalid:
1. **System-level events**: Such as app start/stop, background running, etc., which do not reflect user intent
2. **Pure technical events**: Such as page loading, resource loading, and other low-level technical events
3. **Repeated invalid displays**: The same content displayed repeatedly in a short time without user interaction
4. **Error or exception events**: System errors, network errors, etc.

## Judgment Criteria

For each behavior, please consider:
- **Event name**: Whether it reflects user's active operation (e.g., click_xxx is usually valid, show_xxx needs to be combined with context)
- **Time pattern**: Whether it is within a reasonable time range
- **Context**: Relationship with other behaviors, whether it forms a meaningful sequence
- **Business value**: Whether it helps understand user intent

## Input Data

User behavior list (total {{actions_count}} behaviors):
{{actions_text}}

## Your Task

Analyze each behavior and determine whether it is "valid" (meaningful for analyzing user intent).

Please start the analysis."""


class Action(BaseModel):
    index: int = Field(description="index corresponds to the index value in the input. Examples: 0, 1, 2, 3, 4, 5")
    is_valid: bool = Field(description="Whether the action is valid. Examples: True, False")
    reason: str = Field(description="brief explanation of the judgment (in Chinese). Examples: 'User actively clicked payment button, indicating payment intent', 'App stop event, system-level event, does not reflect user intent'")


class ValidActionsFilterOutput(BaseModel):
    valid_actions: list[Action] = Field(description="List of valid actions with their validation results")

