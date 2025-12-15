"""
用户意图分析系统 - 基于Gemini AI
分析金融信用卡行业用户行为数据，判断用户意图
"""

import pandas as pd
import json
import re
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from collections import defaultdict


class IntentAnalyzer:
    """用户意图分析器"""
    
    def __init__(self, gemini_api_key: str):
        """
        初始化分析器
        
        Args:
            gemini_api_key: Google Gemini API密钥
        """
        genai.configure(api_key=gemini_api_key)
        # 自动选择可用的模型
        try:
            # 获取可用模型列表
            available_models = [m.name for m in genai.list_models() 
                              if 'generateContent' in m.supported_generation_methods]
            
            # 优先使用 gemini-2.5-flash（更快更便宜），如果没有则使用其他可用模型
            model_name = None
            for preferred in ['gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-1.5-flash', 'gemini-1.5-pro']:
                full_name = f'models/{preferred}'
                if full_name in available_models:
                    model_name = preferred
                    break
            
            if model_name is None and available_models:
                # 使用第一个可用模型
                model_name = available_models[0].replace('models/', '')
            
            if model_name:
                self.model = genai.GenerativeModel(model_name)
                print(f"使用模型: {model_name}")
            else:
                raise Exception("未找到可用的模型")
        except Exception as e:
            # 如果自动选择失败，使用默认模型
            print(f"自动选择模型失败: {e}，使用默认模型")
            self.model = genai.GenerativeModel('gemini-2.5-flash')
        
    def load_data(self, csv_path: str) -> pd.DataFrame:
        """
        加载CSV数据
        
        Args:
            csv_path: CSV文件路径
            
        Returns:
            处理后的DataFrame
        """
        # 尝试多种编码方式
        encodings = ['utf-8', 'gbk', 'gb2312', 'latin1', 'iso-8859-1']
        df = None
        for encoding in encodings:
            try:
                df = pd.read_csv(csv_path, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            # 如果所有编码都失败，使用errors='ignore'
            df = pd.read_csv(csv_path, encoding='utf-8', errors='ignore')
        # 转换时间字段
        df['event_time'] = pd.to_datetime(df['event_time'], format='%Y/%m/%d %H:%M', errors='coerce')
        df['approved_time'] = pd.to_datetime(df['approved_time'], format='%Y/%m/%d %H:%M', errors='coerce')
        df['first_payment_time'] = pd.to_datetime(df['first_payment_time'], format='%Y/%m/%d %H:%M', errors='coerce')
        
        # 按用户和时间排序
        df = df.sort_values(['user_uuid', 'event_time'])
        
        return df
    
    def filter_valid_actions(self, user_actions: pd.DataFrame) -> pd.DataFrame:
        """
        使用AI过滤有效行为数据
        
        Args:
            user_actions: 用户行为数据
            
        Returns:
            过滤后的有效行为数据
        """
        if len(user_actions) == 0:
            return user_actions
        
        # 使用AI判断哪些行为是有效的
        valid_indices = self._ai_filter_valid_actions(user_actions)
        
        # 根据AI的判断结果过滤
        valid_actions = user_actions.iloc[valid_indices].copy()
        
        return valid_actions
    
    def segment_actions_by_intent(self, valid_actions: pd.DataFrame) -> List[List[Dict]]:
        """
        根据意图一致性将行为分段
        
        Args:
            valid_actions: 有效行为数据
            
        Returns:
            按意图分段的行为列表，每个段代表一个意图阶段
        """
        if len(valid_actions) == 0:
            return []
        
        # 如果行为太少，不需要分段
        if len(valid_actions) <= 5:
            return [[row.to_dict() for _, row in valid_actions.iterrows()]]
        
        # 使用AI判断意图分段
        segments = self._ai_segment_by_intent(valid_actions)
        
        return segments
    
    def _ai_filter_valid_actions(self, user_actions: pd.DataFrame) -> List[int]:
        """
        使用AI判断哪些行为是有效的（支持分批处理和重试）
        
        Args:
            user_actions: 用户行为数据
            
        Returns:
            有效行为的索引列表
        """
        # 如果行为数量太多，分批处理
        MAX_ACTIONS_PER_BATCH = 50
        total_actions = len(user_actions)
        
        if total_actions <= MAX_ACTIONS_PER_BATCH:
            # 数量不多，直接处理
            return self._ai_filter_batch(user_actions, 0)
        else:
            # 分批处理
            all_valid_indices = []
            for batch_start in range(0, total_actions, MAX_ACTIONS_PER_BATCH):
                batch_end = min(batch_start + MAX_ACTIONS_PER_BATCH, total_actions)
                batch_actions = user_actions.iloc[batch_start:batch_end]
                batch_indices = self._ai_filter_batch(batch_actions, batch_start)
                all_valid_indices.extend(batch_indices)
                # 添加延迟避免API限流
                if batch_end < total_actions:
                    time.sleep(1)
            return all_valid_indices
    
    def _ai_filter_batch(self, batch_actions: pd.DataFrame, start_index: int, max_retries: int = 3) -> List[int]:
        """
        处理一批行为的有效性判断（带重试机制）
        
        Args:
            batch_actions: 一批行为数据
            start_index: 这批行为的起始索引
            max_retries: 最大重试次数
            
        Returns:
            有效行为的索引列表
        """
        # 格式化行为数据
        actions_list = []
        for pos_idx, (_, row) in enumerate(batch_actions.iterrows()):
            action_info = {
                'index': start_index + pos_idx,  # 使用全局索引
                'event_name': row.get('event_name', ''),
                'event_time': str(row.get('event_time', '')),
                'extra_info': str(row.get('extra_info', '')) if pd.notna(row.get('extra_info')) else ''
            }
            actions_list.append(action_info)
        
        # 构建prompt
        prompt = self._build_valid_action_filter_prompt(actions_list)
        
        # 重试机制
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config={
                        'temperature': 0.1,
                        'max_output_tokens': 8192,
                    }
                )
                result_text = response.text
                
                # 解析AI返回的结果
                valid_indices = self._parse_valid_action_indices(result_text, batch_actions, start_index)
                
                return valid_indices
                
            except Exception as e:
                error_msg = str(e)
                if '504' in error_msg or 'Deadline' in error_msg or 'timeout' in error_msg.lower():
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2  # 递增等待时间：2s, 4s, 6s
                        print(f"AI过滤行为超时，{wait_time}秒后重试 (尝试 {attempt + 1}/{max_retries})...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"AI过滤行为时出错（已重试{max_retries}次）: {e}，返回该批次所有行为")
                        return list(range(start_index, start_index + len(batch_actions)))
                else:
                    print(f"AI过滤行为时出错: {e}，返回该批次所有行为")
                    return list(range(start_index, start_index + len(batch_actions)))
        
        # 如果所有重试都失败，返回所有索引
        return list(range(start_index, start_index + len(batch_actions)))
    
    def _build_valid_action_filter_prompt(self, actions_list: List[Dict]) -> str:
        """
        构建用于过滤有效行为的prompt
        
        Args:
            actions_list: 行为数据列表
            
        Returns:
            prompt字符串
        """
        # Format action data
        actions_text = ""
        for i, action in enumerate(actions_list, 1):
            actions_text += f"{i}. Index: {action['index']}, Event: {action['event_name']}, "
            actions_text += f"Time: {action['event_time']}"
            if action['extra_info']:
                actions_text += f", Extra Info: {action['extra_info']}"
            actions_text += "\n"
        
        prompt = f"""You are a user behavior analysis expert in the financial credit card industry. Please analyze the following user behavior data and determine which behaviors are "valid", i.e., meaningful for analyzing user intent.

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

User behavior list (total {len(actions_list)} behaviors):
{actions_text}

## Your Task

Analyze each behavior and determine whether it is "valid" (meaningful for analyzing user intent).

## Output Format

Please output in JSON format, containing validity judgment for each action:

{{
  "valid_actions": [
    {{"index": 0, "is_valid": true, "reason": "User actively clicked payment button, indicating payment intent"}},
    {{"index": 1, "is_valid": false, "reason": "App stop event, system-level event, does not reflect user intent"}},
    ...
  ]
}}

Note:
- index corresponds to the index value in the input
- is_valid: true means valid, false means invalid
- reason: brief explanation of the judgment (in Chinese)

Please start the analysis."""
        
        return prompt
    
    def _fix_json_comma_errors(self, json_str: str) -> str:
        """
        修复JSON中缺少逗号的问题
        
        Args:
            json_str: JSON字符串
            
        Returns:
            修复后的JSON字符串
        """
        result = []
        i = 0
        in_string = False
        escape_next = False
        last_non_whitespace = None
        last_token_end = -1  # 上一个token结束的位置
        
        while i < len(json_str):
            char = json_str[i]
            
            if escape_next:
                result.append(char)
                escape_next = False
                i += 1
                continue
            
            if char == '\\':
                result.append(char)
                escape_next = True
                i += 1
                continue
            
            if char == '"':
                if not in_string:
                    # 字符串开始
                    in_string = True
                    result.append(char)
                else:
                    # 字符串结束
                    in_string = False
                    result.append(char)
                    last_token_end = len(result)
                    last_non_whitespace = '"'
                i += 1
                continue
            
            if in_string:
                result.append(char)
                i += 1
                continue
            
            # 不在字符串内
            if char in [' ', '\n', '\r', '\t']:
                result.append(char)
                i += 1
                continue
            
            # 检查是否需要添加逗号
            if last_token_end > 0 and last_non_whitespace:
                # 情况1: }" 或 ]" (对象/数组后直接跟键)
                if last_non_whitespace in ['}', ']'] and char == '"':
                    # 检查后面是否是键（有冒号）
                    lookahead = json_str[i:min(i+20, len(json_str))]
                    if '":' in lookahead or '": ' in lookahead:
                        result.append(',')
                
                # 情况2: "key": value"key" (值后直接跟键)
                elif last_non_whitespace == '"' and char == '"':
                    # 检查前面是否是值的结束，后面是否是键的开始
                    lookback = ''.join(result[max(0, last_token_end-20):])
                    lookahead = json_str[i:min(i+20, len(json_str))]
                    if ('":' in lookback or lookback.strip().endswith('"')) and ('":' in lookahead or '": ' in lookahead):
                        result.append(',')
                
                # 情况3: number" 或 true" 或 false" 或 null" (值后直接跟键)
                elif last_non_whitespace and char == '"':
                    # 检查前面是否是值的结束
                    lookback = ''.join(result[max(0, last_token_end-10):])
                    lookahead = json_str[i:min(i+20, len(json_str))]
                    if (lookback.strip().endswith(('}', ']', 'true', 'false', 'null')) or 
                        any(lookback.strip().endswith(str(n)) for n in range(10))) and '":' in lookahead:
                        result.append(',')
            
            result.append(char)
            last_non_whitespace = char
            last_token_end = len(result)
            i += 1
        
        return ''.join(result)
    
    def _fix_json_format(self, json_str: str) -> str:
        """
        修复常见的JSON格式问题
        
        Args:
            json_str: 可能有格式问题的JSON字符串
            
        Returns:
            修复后的JSON字符串
        """
        # 首先修复缺少逗号的问题
        json_str = self._fix_json_comma_errors(json_str)
        
        # 移除尾随逗号（在}或]之前，但要小心字符串中的逗号）
        # 使用更精确的正则，避免匹配字符串内的内容
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        # 修复字符串中的未转义字符
        # 使用状态机正确处理字符串内的换行符、制表符等
        result = []
        i = 0
        in_string = False
        escape_next = False
        
        while i < len(json_str):
            char = json_str[i]
            
            if escape_next:
                result.append(char)
                escape_next = False
                i += 1
                continue
            
            if char == '\\':
                result.append(char)
                escape_next = True
                i += 1
                continue
            
            if char == '"':
                in_string = not in_string
                result.append(char)
                i += 1
                continue
            
            if in_string:
                # 在字符串内，需要转义特殊字符
                if char == '\n':
                    result.append('\\n')
                elif char == '\r':
                    result.append('\\r')
                elif char == '\t':
                    result.append('\\t')
                elif char == '\b':
                    result.append('\\b')
                elif char == '\f':
                    result.append('\\f')
                elif ord(char) < 32:  # 其他控制字符
                    result.append(f'\\u{ord(char):04x}')
                else:
                    result.append(char)
            else:
                # 不在字符串内
                result.append(char)
            
            i += 1
        
        json_str = ''.join(result)
        
        # 将单引号替换为双引号（但要小心字符串内容）
        # 先处理键
        json_str = re.sub(r"'(\w+)':", r'"\1":', json_str)
        
        # 处理值（更保守的方法，只处理简单的字符串值）
        # 避免处理包含特殊字符的字符串
        def replace_simple_string_quotes(match):
            content = match.group(1)
            # 如果包含特殊字符，保持原样
            if any(c in content for c in ['\n', '\r', '\t', '\\', '"']):
                return match.group(0)
            return f': "{content}"'
        
        json_str = re.sub(r":\s*'([^']*)'", replace_simple_string_quotes, json_str)
        
        # 修复布尔值（True/False -> true/false）
        # 只在非字符串上下文中替换
        json_str = re.sub(r':\s*\bTrue\b', ': true', json_str)
        json_str = re.sub(r':\s*\bFalse\b', ': false', json_str)
        json_str = re.sub(r':\s*\bNone\b', ': null', json_str)
        
        # 修复数组和对象中的布尔值
        json_str = re.sub(r',\s*\bTrue\b', ', true', json_str)
        json_str = re.sub(r',\s*\bFalse\b', ', false', json_str)
        json_str = re.sub(r'\[\s*\bTrue\b', '[ true', json_str)
        json_str = re.sub(r'\[\s*\bFalse\b', '[ false', json_str)
        
        return json_str
    
    def _aggressive_json_fix(self, json_str: str) -> str:
        """
        更激进的JSON修复策略（当常规修复失败时使用）
        
        Args:
            json_str: JSON字符串
            
        Returns:
            修复后的JSON字符串
        """
        # 移除所有控制字符（除了已转义的）
        import string
        result = []
        i = 0
        in_string = False
        escape_next = False
        
        while i < len(json_str):
            char = json_str[i]
            
            if escape_next:
                result.append(char)
                escape_next = False
                i += 1
                continue
            
            if char == '\\':
                result.append(char)
                escape_next = True
                i += 1
                continue
            
            if char == '"':
                in_string = not in_string
                result.append(char)
                i += 1
                continue
            
            if in_string:
                # 在字符串内，移除或转义控制字符
                if ord(char) < 32 and char not in ['\n', '\r', '\t']:
                    # 跳过不可打印的控制字符
                    i += 1
                    continue
                elif char == '\n':
                    result.append('\\n')
                elif char == '\r':
                    result.append('\\r')
                elif char == '\t':
                    result.append('\\t')
                else:
                    result.append(char)
            else:
                # 不在字符串内，移除控制字符
                if char in string.printable or char in [' ', '\n', '\r', '\t']:
                    result.append(char)
            
            i += 1
        
        return ''.join(result)
    
    def _extract_json_safely(self, text: str) -> Optional[str]:
        """
        安全地提取JSON字符串，尝试多种方法
        
        Args:
            text: 包含JSON的文本
            
        Returns:
            提取的JSON字符串，如果失败返回None
        """
        # 方法1: 查找第一个{，然后通过括号匹配找到完整的JSON
        json_start = text.find('{')
        if json_start >= 0:
            # 从第一个{开始，找到匹配的}
            brace_count = 0
            json_end = -1
            in_string = False
            escape_next = False
            
            for i in range(json_start, len(text)):
                char = text[i]
                
                if escape_next:
                    escape_next = False
                    continue
                
                if char == '\\':
                    escape_next = True
                    continue
                
                if char == '"':
                    in_string = not in_string
                    continue
                
                if not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break
            
            if json_end > json_start:
                return text[json_start:json_end]
        
        # 方法2: 尝试找到所有可能的JSON块，选择最可能的一个
        # 查找所有 { 和 } 的位置
        brace_positions = []
        in_string = False
        escape_next = False
        
        for i, char in enumerate(text):
            if escape_next:
                escape_next = False
                continue
            
            if char == '\\':
                escape_next = True
                continue
            
            if char == '"':
                in_string = not in_string
                continue
            
            if not in_string:
                if char == '{':
                    brace_positions.append(('{', i))
                elif char == '}':
                    brace_positions.append(('}', i))
        
        # 尝试从每个 { 开始构建JSON
        for i, (char, pos) in enumerate(brace_positions):
            if char == '{':
                # 找到匹配的 }
                brace_count = 0
                for j in range(i, len(brace_positions)):
                    if brace_positions[j][0] == '{':
                        brace_count += 1
                    else:
                        brace_count -= 1
                        if brace_count == 0:
                            end_pos = brace_positions[j][1] + 1
                            candidate = text[pos:end_pos]
                            # 尝试解析这个候选JSON
                            try:
                                fixed = self._fix_json_format(candidate)
                                json.loads(fixed)  # 验证是否有效
                                return candidate
                            except:
                                continue
        
        # 方法3: 使用正则表达式查找JSON对象（作为最后手段）
        # 这个可能不够准确，但可以尝试
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)
        if matches:
            # 返回最长的匹配
            candidate = max(matches, key=len)
            try:
                fixed = self._fix_json_format(candidate)
                json.loads(fixed)  # 验证
                return candidate
            except:
                pass
        
        return None
    
    def _parse_valid_action_indices(self, ai_response: str, user_actions: pd.DataFrame, start_index: int = 0) -> List[int]:
        """
        解析AI返回的有效行为索引
        
        Args:
            ai_response: AI返回的文本
            user_actions: 原始行为数据（用于验证索引有效性）
            start_index: 起始索引（用于分批处理）
            
        Returns:
            有效行为的索引列表
        """
        try:
            # 尝试提取JSON
            json_start = ai_response.find('{')
            json_end = ai_response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = ai_response[json_start:json_end]
                
                # 尝试修复常见的JSON格式问题
                json_str = self._fix_json_format(json_str)
                
                result = json.loads(json_str)
                
                valid_indices = []
                if 'valid_actions' in result:
                    for item in result['valid_actions']:
                        if item.get('is_valid', False):
                            idx = item.get('index')
                            # 验证索引在有效范围内
                            if idx is not None and start_index <= idx < start_index + len(user_actions):
                                valid_indices.append(idx)
                
                if len(valid_indices) > 0:
                    return valid_indices
        except json.JSONDecodeError as e:
            # 不打印错误，直接尝试修复
            # 尝试多种修复策略
            try:
                # 策略1: 尝试从错误位置附近提取JSON
                json_str = self._extract_json_safely(ai_response)
                if json_str:
                    # 尝试多次修复，每次使用更强的策略
                    for attempt in range(5):
                        try:
                            if attempt == 0:
                                fixed_json = self._fix_json_format(json_str)
                            elif attempt == 1:
                                fixed_json = self._aggressive_json_fix(json_str)
                                fixed_json = self._fix_json_format(fixed_json)
                            else:
                                # 更激进的修复：移除所有可能导致问题的字符
                                fixed_json = self._aggressive_json_fix(json_str)
                                fixed_json = self._fix_json_format(fixed_json)
                                # 尝试修复常见的结构问题
                                fixed_json = re.sub(r'(\w+)"\s*(\w+)', r'\1", "\2', fixed_json)  # 添加缺失的逗号和引号
                            
                            result = json.loads(fixed_json)
                            valid_indices = []
                            if 'valid_actions' in result:
                                for item in result['valid_actions']:
                                    if item.get('is_valid', False):
                                        idx = item.get('index')
                                        if idx is not None and start_index <= idx < start_index + len(user_actions):
                                            valid_indices.append(idx)
                            if len(valid_indices) > 0:
                                return valid_indices
                            break
                        except json.JSONDecodeError:
                            if attempt < 4:
                                continue
                            # 最后一次尝试：使用正则表达式直接提取数据
                            return self._extract_indices_with_regex(ai_response, user_actions, start_index)
            except Exception:
                # 最后的后备方案：使用正则表达式
                return self._extract_indices_with_regex(ai_response, user_actions, start_index)
        except Exception as e:
            print(f"解析AI响应时出错: {e}")
        
        # 如果都失败了，使用正则表达式提取
        return self._extract_indices_with_regex(ai_response, user_actions, start_index)
    
    def _extract_indices_with_regex(self, ai_response: str, user_actions: pd.DataFrame, start_index: int = 0) -> List[int]:
        """
        使用正则表达式从AI响应中提取有效行为索引（最后的备用方案）
        
        Args:
            ai_response: AI返回的文本
            user_actions: 用户行为数据
            start_index: 起始索引
            
        Returns:
            有效行为的索引列表
        """
        valid_indices = []
        
        # 方法1: 查找 "index": X, "is_valid": true 模式
        pattern1 = r'"index":\s*(\d+)[^}]*"is_valid":\s*(true|false)'
        matches1 = re.findall(pattern1, ai_response, re.IGNORECASE)
        for idx_str, is_valid in matches1:
            if is_valid.lower() == 'true':
                idx = int(idx_str)
                if start_index <= idx < start_index + len(user_actions):
                    valid_indices.append(idx)
        
        if len(valid_indices) > 0:
            return valid_indices
        
        # 方法2: 查找所有index和is_valid的组合
        indices = re.findall(r'"index"\s*:\s*(\d+)', ai_response)
        valid_flags = re.findall(r'"is_valid"\s*:\s*(true|false)', ai_response, re.IGNORECASE)
        
        if len(indices) == len(valid_flags):
            for idx_str, is_valid in zip(indices, valid_flags):
                if is_valid.lower() == 'true':
                    idx = int(idx_str)
                    if start_index <= idx < start_index + len(user_actions):
                        valid_indices.append(idx)
        
        if len(valid_indices) > 0:
            return valid_indices
        
        # 如果都失败了，返回所有索引（保守策略）
        return list(range(start_index, start_index + len(user_actions)))
    
    def _ai_segment_by_intent(self, valid_actions: pd.DataFrame) -> List[List[Dict]]:
        """
        使用AI根据意图一致性将行为分段（支持分批处理和重试）
        
        Args:
            valid_actions: 有效行为数据
            
        Returns:
            按意图分段的行为列表
        """
        # 如果行为数量太多，分批处理
        MAX_ACTIONS_PER_BATCH = 50
        total_actions = len(valid_actions)
        
        if total_actions <= MAX_ACTIONS_PER_BATCH:
            # 数量不多，直接处理
            return self._ai_segment_batch(valid_actions, 0)
        else:
            # 分批处理，然后合并分段
            all_segments = []
            for batch_start in range(0, total_actions, MAX_ACTIONS_PER_BATCH):
                batch_end = min(batch_start + MAX_ACTIONS_PER_BATCH, total_actions)
                batch_actions = valid_actions.iloc[batch_start:batch_end]
                batch_segments = self._ai_segment_batch(batch_actions, batch_start)
                all_segments.extend(batch_segments)
                # 添加延迟避免API限流
                if batch_end < total_actions:
                    time.sleep(1)
            return all_segments
    
    def _ai_segment_batch(self, batch_actions: pd.DataFrame, start_index: int, max_retries: int = 3) -> List[List[Dict]]:
        """
        处理一批行为的意图分段（带重试机制）
        
        Args:
            batch_actions: 一批行为数据
            start_index: 这批行为的起始索引
            max_retries: 最大重试次数
            
        Returns:
            按意图分段的行为列表
        """
        # 格式化行为数据
        actions_list = []
        for pos_idx, (_, row) in enumerate(batch_actions.iterrows()):
            action_info = {
                'index': start_index + pos_idx,
                'event_name': row.get('event_name', ''),
                'event_time': str(row.get('event_time', '')),
                'extra_info': str(row.get('extra_info', '')) if pd.notna(row.get('extra_info')) else ''
            }
            actions_list.append(action_info)
        
        # 构建prompt
        prompt = self._build_intent_segmentation_prompt(actions_list)
        
        # 重试机制
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config={
                        'temperature': 0.1,
                        'max_output_tokens': 8192,
                    }
                )
                result_text = response.text
                
                # 解析AI返回的分段结果
                segments = self._parse_intent_segments(result_text, batch_actions, start_index)
                
                return segments
                
            except Exception as e:
                error_msg = str(e)
                if '504' in error_msg or 'Deadline' in error_msg or 'timeout' in error_msg.lower():
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2  # 递增等待时间：2s, 4s, 6s
                        print(f"AI意图分段超时，{wait_time}秒后重试 (尝试 {attempt + 1}/{max_retries})...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"AI意图分段时出错（已重试{max_retries}次）: {e}，返回单个段")
                        return [[row.to_dict() for _, row in batch_actions.iterrows()]]
                else:
                    print(f"AI意图分段时出错: {e}，返回单个段")
                    return [[row.to_dict() for _, row in batch_actions.iterrows()]]
        
        # 如果所有重试都失败，返回单个段
        return [[row.to_dict() for _, row in batch_actions.iterrows()]]
    
    def _build_intent_segmentation_prompt(self, actions_list: List[Dict]) -> str:
        """
        构建用于意图分段的prompt
        
        Args:
            actions_list: 行为数据列表
            
        Returns:
            prompt字符串
        """
        # Format action data
        actions_text = ""
        for i, action in enumerate(actions_list, 1):
            actions_text += f"{i}. Index: {action['index']}, Event: {action['event_name']}, "
            actions_text += f"Time: {action['event_time']}"
            if action['extra_info']:
                actions_text += f", Extra Info: {action['extra_info']}"
            actions_text += "\n"
        
        prompt = f"""You are a user behavior analysis expert in the financial credit card industry. Please analyze the following user behaviors and segment them into different intent phases based on intent consistency.

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

User behavior list (total {len(actions_list)} behaviors):
{actions_text}

## Your Task

Analyze the behavior sequence and segment it into different intent phases. Each phase should represent behaviors with consistent intent.

## Output Format

Please output in JSON format, containing intent segments:

{{
  "intent_segments": [
    {{
      "segment_index": 0,
      "start_index": 0,
      "end_index": 5,
      "intent_description": "User is exploring voucher options and selecting payment method",
      "behavior_indices": [0, 1, 2, 3, 4, 5]
    }},
    {{
      "segment_index": 1,
      "start_index": 6,
      "end_index": 12,
      "intent_description": "User is checking credit limit and available balance",
      "behavior_indices": [6, 7, 8, 9, 10, 11, 12]
    }},
    ...
  ]
}}

Note:
- segment_index: sequential number starting from 0
- start_index: first behavior index in this segment (inclusive)
- end_index: last behavior index in this segment (inclusive)
- intent_description: brief description of the intent for this segment (in Chinese)
- behavior_indices: list of all behavior indices in this segment (should be consecutive)

Please start the analysis."""
        
        return prompt
    
    def _parse_intent_segments(self, ai_response: str, valid_actions: pd.DataFrame, start_index: int = 0) -> List[List[Dict]]:
        """
        解析AI返回的意图分段结果
        
        Args:
            ai_response: AI返回的文本
            valid_actions: 有效行为数据（用于验证索引）
            start_index: 起始索引（用于分批处理）
            
        Returns:
            按意图分段的行为列表
        """
        try:
            # 尝试提取JSON
            json_start = ai_response.find('{')
            json_end = ai_response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = ai_response[json_start:json_end]
                
                # 尝试修复常见的JSON格式问题
                json_str = self._fix_json_format(json_str)
                
                result = json.loads(json_str)
                
                segments = []
                if 'intent_segments' in result:
                    for segment_info in result['intent_segments']:
                        behavior_indices = segment_info.get('behavior_indices', [])
                        
                        # 验证索引有效性并获取行为（考虑start_index偏移）
                        valid_indices = [idx for idx in behavior_indices 
                                        if start_index <= idx < start_index + len(valid_actions)]
                        
                        if len(valid_indices) > 0:
                            # 获取对应的行为数据（转换为相对索引）
                            segment_actions = [valid_actions.iloc[idx - start_index].to_dict() 
                                             for idx in valid_indices]
                            segments.append(segment_actions)
                
                if len(segments) > 0:
                    return segments
        except json.JSONDecodeError as e:
            # 不打印错误，直接尝试修复
            # 尝试多种修复策略
            try:
                json_str = self._extract_json_safely(ai_response)
                if json_str:
                    # 尝试多次修复，每次使用更强的策略
                    for attempt in range(5):
                        try:
                            if attempt == 0:
                                fixed_json = self._fix_json_format(json_str)
                            elif attempt == 1:
                                fixed_json = self._aggressive_json_fix(json_str)
                                fixed_json = self._fix_json_format(fixed_json)
                            else:
                                # 更激进的修复
                                fixed_json = self._aggressive_json_fix(json_str)
                                fixed_json = self._fix_json_format(fixed_json)
                                fixed_json = re.sub(r'(\w+)"\s*(\w+)', r'\1", "\2', fixed_json)
                            
                            result = json.loads(fixed_json)
                            segments = []
                            if 'intent_segments' in result:
                                for segment_info in result['intent_segments']:
                                    behavior_indices = segment_info.get('behavior_indices', [])
                                    valid_indices = [idx for idx in behavior_indices 
                                                    if start_index <= idx < start_index + len(valid_actions)]
                                    if len(valid_indices) > 0:
                                        segment_actions = [valid_actions.iloc[idx - start_index].to_dict() 
                                                         for idx in valid_indices]
                                        segments.append(segment_actions)
                            if len(segments) > 0:
                                return segments
                            break
                        except json.JSONDecodeError:
                            if attempt < 4:
                                continue
                            # 最后一次尝试：使用正则表达式
                            return self._extract_segments_with_regex(ai_response, valid_actions, start_index)
            except Exception:
                # 最后的后备方案
                return self._extract_segments_with_regex(ai_response, valid_actions, start_index)
        except Exception as e:
            print(f"解析意图分段响应时出错: {e}")
        
        # 如果解析失败，尝试从文本中提取
        # 查找behavior_indices模式
        indices_pattern = r'"behavior_indices":\s*\[([^\]]+)\]'
        matches = re.findall(indices_pattern, ai_response)
        
        if matches:
            segments = []
            for match in matches:
                # 提取数字
                indices = [int(x.strip()) for x in match.split(',') if x.strip().isdigit()]
                valid_indices = [idx for idx in indices if start_index <= idx < start_index + len(valid_actions)]
                
                if len(valid_indices) > 0:
                    segment_actions = [valid_actions.iloc[idx - start_index].to_dict() for idx in valid_indices]
                    segments.append(segment_actions)
            
            if len(segments) > 0:
                return segments
        
        # 如果都失败了，使用正则表达式提取
        return self._extract_segments_with_regex(ai_response, valid_actions, start_index)
    
    def _extract_segments_with_regex(self, ai_response: str, valid_actions: pd.DataFrame, start_index: int = 0) -> List[List[Dict]]:
        """
        使用正则表达式从AI响应中提取意图分段（最后的备用方案）
        
        Args:
            ai_response: AI返回的文本
            valid_actions: 有效行为数据
            start_index: 起始索引
            
        Returns:
            按意图分段的行为列表
        """
        segments = []
        
        # 查找 behavior_indices 数组
        pattern = r'"behavior_indices"\s*:\s*\[([^\]]+)\]'
        matches = re.findall(pattern, ai_response)
        
        for match in matches:
            # 提取所有数字
            indices = [int(x.strip()) for x in re.findall(r'\d+', match) if x.strip().isdigit()]
            valid_indices = [idx for idx in indices if start_index <= idx < start_index + len(valid_actions)]
            
            if len(valid_indices) > 0:
                segment_actions = [valid_actions.iloc[idx - start_index].to_dict() for idx in valid_indices]
                segments.append(segment_actions)
        
        if len(segments) > 0:
            return segments
        
        # 如果都失败了，返回单个段（所有行为）
        return [[row.to_dict() for _, row in valid_actions.iterrows()]]
    
    def group_user_actions_by_session(self, user_actions: pd.DataFrame, 
                                     session_timeout_minutes: int = 30) -> List[List[Dict]]:
        """
        按会话分组用户行为
        
        Args:
            user_actions: 用户行为数据
            session_timeout_minutes: 会话超时时间（分钟）
            
        Returns:
            分组后的会话列表
        """
        if len(user_actions) == 0:
            return []
        
        sessions = []
        current_session = []
        
        for idx, row in user_actions.iterrows():
            if len(current_session) == 0:
                current_session.append(row.to_dict())
            else:
                last_time = pd.to_datetime(current_session[-1]['event_time'])
                current_time = pd.to_datetime(row['event_time'])
                time_diff = (current_time - last_time).total_seconds() / 60
                
                if time_diff <= session_timeout_minutes:
                    current_session.append(row.to_dict())
                else:
                    if len(current_session) > 0:
                        sessions.append(current_session)
                    current_session = [row.to_dict()]
        
        if len(current_session) > 0:
            sessions.append(current_session)
        
        return sessions
    
    def format_actions_for_prompt(self, actions: List[Dict]) -> str:
        """
        格式化行为数据用于prompt
        
        Args:
            actions: 行为数据列表
            
        Returns:
            格式化后的字符串
        """
        formatted = []
        for i, action in enumerate(actions, 1):
            event_name = action.get('event_name', '')
            event_time = action.get('event_time', '')
            extra_info = action.get('extra_info', '')
            
            line = f"{i}. 时间: {event_time}, 事件: {event_name}"
            if pd.notna(extra_info) and extra_info.strip():
                line += f", 额外信息: {extra_info}"
            formatted.append(line)
        
        return "\n".join(formatted)
    
    def get_user_context(self, user_actions: pd.DataFrame) -> Dict[str, Any]:
        """
        提取用户上下文信息
        
        Args:
            user_actions: 用户行为数据
            
        Returns:
            用户上下文信息
        """
        first_action = user_actions.iloc[0]
        last_action = user_actions.iloc[-1]
        
        context = {
            'user_uuid': first_action.get('user_uuid', ''),
            'approved_time': str(first_action.get('approved_time', '')),
            'first_payment_time': str(first_action.get('first_payment_time', '')),
            'first_action_time': str(first_action.get('event_time', '')),
            'last_action_time': str(last_action.get('event_time', '')),
            'total_actions': len(user_actions),
            'unique_events': user_actions['event_name'].nunique(),
        }
        
        return context
    
    def analyze_intent(self, user_context: Dict, actions: List[Dict], 
                      history: Optional[Dict] = None) -> Dict[str, Any]:
        """
        使用Gemini AI分析用户意图
        
        Args:
            user_context: 用户上下文信息
            actions: 行为数据列表
            history: 历史意图分析结果（可选）
            
        Returns:
            意图分析结果
        """
        prompt = self._build_prompt(user_context, actions, history)
        
        try:
            response = self.model.generate_content(prompt)
            result_text = response.text
            
            # 尝试解析JSON
            try:
                # 提取JSON部分
                json_start = result_text.find('{')
                json_end = result_text.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = result_text[json_start:json_end]
                    # 尝试修复常见的JSON格式问题
                    json_str = self._fix_json_format(json_str)
                    result = json.loads(json_str)
                else:
                    result = {'intent': result_text, 'confidence_score': 0.5, 'raw_response': result_text}
            except json.JSONDecodeError as e:
                print(f"解析意图分析JSON时出错: {e}")
                # 尝试安全提取JSON
                try:
                    json_str = self._extract_json_safely(result_text)
                    if json_str:
                        json_str = self._fix_json_format(json_str)
                        result = json.loads(json_str)
                    else:
                        result = {'intent': result_text, 'confidence_score': 0.5, 'raw_response': result_text}
                except Exception:
                    result = {'intent': result_text, 'confidence_score': 0.5, 'raw_response': result_text}
            
            return result
            
        except Exception as e:
            return {
                'error': str(e),
                'intent': '分析失败',
                'score': 0.0
            }
    
    def _build_prompt(self, user_context: Dict, actions: List[Dict], 
                     history: Optional[Dict] = None) -> str:
        """
        构建Gemini AI的prompt
        
        Args:
            user_context: 用户上下文信息
            actions: 行为数据列表
            history: 历史意图分析结果（可选）
            
        Returns:
            完整的prompt字符串
        """
        actions_text = self.format_actions_for_prompt(actions)
        
        history_text = ""
        if history:
            history_text = f"""
历史意图分析:
- 之前意图: {history.get('intent', '无')}
- 之前得分: {history.get('confidence_score', history.get('score', 0.0))}
- 之前分析时间: {history.get('timestamp', '无')}
"""
        
        prompt = f"""你是一位金融信用卡行业的用户行为分析专家。请综合分析所有输入信息来提取用户意图。

## 分析数据源（请使用所有数据源）

1. **用户信息**: 用户ID、审批时间、首次支付时间 → 了解用户状态和生命周期阶段

2. **用户行为序列**: 按时间顺序的行为事件 → 了解用户实际做了什么

3. **行为上下文**: 事件类型、时间间隔、额外信息 → 了解行为深度和模式

4. **历史意图**: 之前的意图分析结果（如果存在）→ 了解意图演变

## 用户行为信号权重

**高权重**（明确兴趣）:
- click_xxx（点击操作）: 显示用户主动选择
- show_pay_checkout_xxx（支付页面）: 显示支付意图
- show_limit_xxx（额度相关）: 显示额度关注
- click_fullpopup_pribtn_xxx（弹窗主按钮点击）: 显示对营销活动的兴趣

**中权重**（参与度）:
- show_xxx（页面展示）: 显示浏览模式
- show_paymentpage_xxx（支付相关页面）: 显示支付流程参与

**低权重**（导航）:
- 重复访问相同页面: 显示深度探索或犹豫
- 时间间隔: 显示参与深度（短间隔=活跃，长间隔=思考）

关键: 高权重信号揭示用户关心什么。低权重模式揭示参与深度和上下文。

## 行为序列分析（对多个行为至关重要）

**顺序很重要！** 序列揭示用户的思考过程：

- 首次 → 最后: 显示兴趣演变（从浏览到支付，从探索到决策）

- 行为类型序列: 
  - 浏览 → 点击 → 支付页面: 显示购买意图
  - 额度页面 → 支付页面: 显示额度使用意图
  - 优惠券页面 → 支付页面: 显示优惠寻求意图
  - 重复访问相同页面: 显示犹豫或深度比较

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

## Output JSON Format

{{
  "intent": "User's main intent description (in Chinese, must be specific about what product feature they are exploring)",
  "intent_category": "Intent category (payment_intent/credit_limit_intent/installment_intent/voucher_intent/marketing_intent/exploration_intent)",
  "confidence_score": A float number between 0.0-1.0,
  "certainty_level": "Very High/High/Medium/Low - how certain you are about this intent",
  "evidence_quality": "Strong/Medium/Weak - quality of evidence supporting this intent",
  "explored_feature": "Specific product feature the user is exploring (in Chinese, e.g., '虚拟账户支付', '现金借贷', '分期付款'等)",
  "exploration_purpose": "Purpose of exploration (in Chinese, e.g., '了解如何使用', '准备首次交易', '比较支付选项'等)",
  "first_transaction_connection": "How this intent helps user complete first transaction (in Chinese)",
  "baseline_trust": A float number between 0.0-1.0 representing user's baseline trust in the product/service,
  "trust_indicators": ["Indicator 1 (in Chinese)", "Indicator 2 (in Chinese)", ...],
  "concerns": [
    {{
      "concern_type": "Security/Credit Limit/Fees/Usage Difficulty/Other",
      "concern_description": "Specific concern description (in Chinese)",
      "concern_severity": "High/Medium/Low",
      "evidence": ["Behavior evidence 1", "Behavior evidence 2", ...]
    }}
  ],
  "psychological_reference": {{
    "expected_value": "What user expects (in Chinese, e.g., expected credit limit, discount amount, etc.)",
    "perceived_value": "What user actually perceives (in Chinese)",
    "gap_analysis": "Gap between expected and perceived, and its impact on first transaction (in Chinese)"
  }},
  "key_behaviors": ["key behavior 1", "key behavior 2", ...],
  "reasoning": "Analysis reasoning process (in Chinese, detailed explanation including: what feature, why exploring, trust level, concerns, psychological factors, how it connects to first transaction)",
  "next_action_prediction": "Predicted next possible user action (in Chinese)",
  "operation_recommendation": {{
    "online_solutions": ["Online solution 1 (in Chinese)", "Online solution 2 (in Chinese)", ...],
    "offline_solutions": ["Offline solution 1 (in Chinese)", "Offline solution 2 (in Chinese)", ...],
    "priority": "High/Medium/Low (based on user's readiness for first transaction)",
    "targeted_message": "Specific message or intervention tailored to user's trust level and concerns (in Chinese)"
  }}
}}

## 输入数据

用户信息:
- 用户ID: {user_context.get('user_uuid', 'N/A')}
- 审批时间: {user_context.get('approved_time', 'N/A')}
- 首次支付时间: {user_context.get('first_payment_time', 'N/A')}
- 首次行为时间: {user_context.get('first_action_time', 'N/A')}
- 最后行为时间: {user_context.get('last_action_time', 'N/A')}
- 总行为数: {user_context.get('total_actions', 0)}
- 唯一事件类型数: {user_context.get('unique_events', 0)}

用户行为序列:
{actions_text}

{history_text}

请开始分析用户意图。"""
        
        return prompt
    
    def analyze_user_intent(self, csv_path: Optional[str] = None, user_uuid: Optional[str] = None,
                           session_timeout_minutes: int = 30,
                           preloaded_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        分析用户意图（主入口）
        
        Args:
            csv_path: CSV文件路径（若已提供preloaded_df则可为空）
            user_uuid: 指定用户ID（如果为None，分析所有用户）
            session_timeout_minutes: 会话超时时间（分钟）
            preloaded_df: 预加载的数据DataFrame，避免重复读取
            
        Returns:
            分析结果字典
        """
        # 加载数据（优先使用预加载数据以避免重复读盘）
        if preloaded_df is not None:
            df = preloaded_df.copy()
        else:
            if not csv_path:
                return {'error': '缺少数据源(csv_path或preloaded_df)'}
            df = self.load_data(csv_path)
        
        # 如果指定了用户，只分析该用户
        if user_uuid:
            df = df[df['user_uuid'] == user_uuid]
        
        if len(df) == 0:
            return {'error': '没有找到用户数据'}
        
        # 按用户分组
        results = {}
        
        for uuid, user_df in df.groupby('user_uuid'):
            # 过滤有效行为
            valid_actions = self.filter_valid_actions(user_df)
            
            if len(valid_actions) == 0:
                continue
            
            # 按会话分组（基于时间）
            time_sessions = self.group_user_actions_by_session(valid_actions, session_timeout_minutes)
            
            # 对每个时间会话，进一步按意图分段
            all_intent_segments = []
            for time_session in time_sessions:
                session_df = pd.DataFrame(time_session)
                # 按意图一致性分段
                intent_segments = self.segment_actions_by_intent(session_df)
                all_intent_segments.extend(intent_segments)
            
            # 分析每个意图段
            session_results = []
            history = None
            
            for segment_idx, segment_actions in enumerate(all_intent_segments):
                user_context = self.get_user_context(pd.DataFrame(segment_actions))
                
                # 分析意图
                intent_result = self.analyze_intent(user_context, segment_actions, history)
                intent_result['session_index'] = segment_idx
                intent_result['session_size'] = len(segment_actions)
                intent_result['timestamp'] = datetime.now().isoformat()
                
                session_results.append(intent_result)
                
                # 更新历史（使用最后一次分析结果）
                history = intent_result
            
            results[uuid] = {
                'user_uuid': uuid,
                'total_sessions': len(all_intent_segments),
                'sessions': session_results
            }
        
        return results


def main():
    """主函数 - 可以直接运行进行分析"""
    import os
    import json
    
    # 从环境变量获取API密钥
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("错误: 请设置环境变量 GEMINI_API_KEY")
        print("使用方法: export GEMINI_API_KEY='your-api-key'")
        return
    
    # 创建分析器
    print("正在初始化分析器...")
    analyzer = IntentAnalyzer(api_key)
    
    # 检查数据文件
    csv_path = 'data.csv'
    if not os.path.exists(csv_path):
        print(f"错误: 找不到文件 {csv_path}")
        return
    
    # 默认分析第一个用户（作为示例）
    print(f"\n正在加载数据文件: {csv_path}...")
    df = analyzer.load_data(csv_path)
    first_user = df['user_uuid'].iloc[0]
    
    print(f"\n开始分析用户: {first_user}")
    print("这可能需要一些时间，请耐心等待...\n")
    
    # 分析单个用户（复用已加载数据以避免重复读盘）
    results = analyzer.analyze_user_intent(
        user_uuid=first_user,
        session_timeout_minutes=30,
        preloaded_df=df[df['user_uuid'] == first_user]
    )
    
    # 保存结果
    output_file = f'intent_result_{first_user[:8]}.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 打印结果摘要
    print("\n" + "="*60)
    print("分析结果摘要")
    print("="*60)
    
    for user_uuid, user_result in results.items():
        if 'sessions' in user_result:
            print(f"\n用户: {user_uuid[:16]}...")
            print(f"  会话数: {user_result.get('total_sessions', 0)}")
            
            for session in user_result.get('sessions', []):
                intent = session.get('intent', 'N/A')
                score = session.get('confidence_score', 0)
                category = session.get('intent_category', 'N/A')
                
                print(f"\n  会话 {session.get('session_index', 0) + 1}:")
                print(f"    意图: {intent}")
                print(f"    类别: {category}")
                print(f"    置信度: {score:.2f}")
                
                if 'key_behaviors' in session:
                    print(f"    关键行为: {', '.join(session['key_behaviors'][:3])}")
    
    print(f"\n完整结果已保存到: {output_file}")
    print("\n提示: 要分析其他用户或批量分析，请使用 run_analysis.py")


if __name__ == '__main__':
    main()

