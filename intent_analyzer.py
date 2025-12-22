"""
用户意图分析系统 - 基于Gemini AI
分析金融信用卡行业用户行为数据，判断用户意图
"""

import asyncio
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional, TypeVar

import jinja2
import pandas as pd
from filelock import FileLock
from google import genai
from google.genai.types import GenerateContentConfig
from pydantic import BaseModel

from prompts import (
    INTENT_ANALYSIS,
    INTENT_ONLY_ANALYSIS,
    INTENT_SEGMENTATION,
    OPERATION_RECOMMENDATION,
    VALID_ACTIONS_FILTER,
    IntentAnalysisOutput,
    IntentOnlyAnalysisOutput,
    IntentSegmentationOutput,
    OperationRecommendationOutput,
    ValidActionsFilterOutput,
)

T = TypeVar("T", bound=BaseModel)


class IntentAnalyzer:
    """用户意图分析器"""

    def __init__(self, gemini_api_key: str):
        """
        初始化分析器

        Args:
            gemini_api_key: Google Gemini API密钥
        """
        self.client = genai.Client(api_key=gemini_api_key).aio
        self.model_name = "gemini-2.5-flash"

    def load_completed_users(self, result_file: Optional[str] = None) -> set[str]:
        """
        从结果文件中加载已完成的用户列表（带文件锁）

        Args:
            result_file: 结果文件路径（JSON格式，key为user_id）

        Returns:
            已完成的用户UUID集合
        """
        if not result_file or not os.path.exists(result_file):
            return set()
        
        lock_file = f"{result_file}.lock"
        lock = FileLock(lock_file, timeout=10)
        
        try:
            with lock:
                with open(result_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 结果文件的key就是user_id，直接返回所有key
                    return set(data.keys())
        except Exception as e:
            print(f"警告: 加载结果文件失败: {e}")
            return set()

    def update_result_file(self, result_file: Optional[str], user_uuid: str, user_result: Dict[str, Any]):
        """
        更新结果文件（添加一个完成的用户结果，带文件锁保证并发安全）

        Args:
            result_file: 结果文件路径
            user_uuid: 完成的用户UUID
            user_result: 用户分析结果
        """
        if not result_file:
            return
        
        lock_file = f"{result_file}.lock"
        lock = FileLock(lock_file, timeout=10)
        
        try:
            with lock:
                # 读取已有结果
                existing_results = {}
                if os.path.exists(result_file):
                    with open(result_file, 'r', encoding='utf-8') as f:
                        existing_results = json.load(f)
                
                # 更新结果
                existing_results[user_uuid] = user_result
                
                # 保存结果
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump(existing_results, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"警告: 更新结果文件失败: {e}")

    async def llm_request(self, prompt: str, response_model: type[T]) -> T:
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                response = await self.client.models.generate_content(
                    model=f"models/{self.model_name}",
                    contents=[prompt],
                    config=GenerateContentConfig(
                        temperature=0.1,
                        max_output_tokens=81920,
                        response_mime_type="application/json",
                        response_schema=response_model
                    ),
                )
                return response.parsed
            except Exception as e:
                if attempt == max_retries:
                    raise
                print(f"请求LLM失败，第 {attempt} 次尝试: {e}，即将重试...")


    def load_data(self, csv_path: str) -> pd.DataFrame:
        """
        加载CSV数据

        Args:
            csv_path: CSV文件路径

        Returns:
            处理后的DataFrame
        """
        # 尝试多种编码方式
        encodings = ["utf-8", "gbk", "gb2312", "latin1", "iso-8859-1"]
        df = None
        for encoding in encodings:
            try:
                df = pd.read_csv(csv_path, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue

        if df is None:
            # 如果所有编码都失败，使用errors='ignore'
            df = pd.read_csv(csv_path, encoding="utf-8", errors="ignore")
        # 转换时间字段
        df["event_time"] = pd.to_datetime(
            df["event_time"], format="%Y/%m/%d %H:%M", errors="coerce"
        )
        df["approved_time"] = pd.to_datetime(
            df["approved_time"], format="%Y/%m/%d %H:%M", errors="coerce"
        )
        df["first_payment_time"] = pd.to_datetime(
            df["first_payment_time"], format="%Y/%m/%d %H:%M", errors="coerce"
        )

        # 按用户和时间排序
        df = df.sort_values(["user_uuid", "event_time"])

        return df

    async def filter_valid_actions(self, user_actions: pd.DataFrame) -> pd.DataFrame:
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
        valid_indices = await self._ai_filter_valid_actions(user_actions)

        # 根据AI的判断结果过滤
        valid_actions = user_actions.iloc[valid_indices].copy()

        return valid_actions

    async def segment_actions_by_intent(
            self, valid_actions: pd.DataFrame
    ) -> List[List[Dict]]:
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
        segments = await self._ai_segment_by_intent(valid_actions)

        return segments

    async def _ai_filter_valid_actions(self, user_actions: pd.DataFrame) -> List[int]:
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
            return await self._ai_filter_batch(user_actions, 0)
        else:
            # 分批处理
            all_valid_indices = []
            for batch_start in range(0, total_actions, MAX_ACTIONS_PER_BATCH):
                batch_end = min(batch_start + MAX_ACTIONS_PER_BATCH, total_actions)
                batch_actions = user_actions.iloc[batch_start:batch_end]
                batch_indices = await self._ai_filter_batch(batch_actions, batch_start)
                all_valid_indices.extend(batch_indices)
                # 添加延迟避免API限流
                if batch_end < total_actions:
                    await asyncio.sleep(1)
            return all_valid_indices

    async def _ai_filter_batch(
            self, batch_actions: pd.DataFrame, start_index: int, max_retries: int = 3
    ) -> List[int]:
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
                "index": start_index + pos_idx,  # 使用全局索引
                "event_name": row.get("event_name", ""),
                "event_time": str(row.get("event_time", "")),
                "extra_info": (
                    str(row.get("extra_info", ""))
                    if pd.notna(row.get("extra_info"))
                    else ""
                ),
            }
            actions_list.append(action_info)

        # 构建prompt
        prompt = self._build_valid_action_filter_prompt(actions_list)
        result = await self.llm_request(prompt, ValidActionsFilterOutput)
        valid_indices = self._parse_valid_action_indices(result, batch_actions, start_index)
        return valid_indices

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
            actions_text += (
                f"{i}. Index: {action['index']}, Event: {action['event_name']}, "
            )
            actions_text += f"Time: {action['event_time']}"
            if action["extra_info"]:
                actions_text += f", Extra Info: {action['extra_info']}"
            actions_text += "\n"

        prompt = jinja2.Template(VALID_ACTIONS_FILTER).render(
            actions_text=actions_text, actions_count=len(actions_list)
        )

        return prompt

    def _parse_valid_action_indices(
            self,
            result: ValidActionsFilterOutput,
            user_actions: pd.DataFrame,
            start_index: int = 0,
    ) -> list[int]:
        """
        解析AI返回的有效行为索引

        Args:
            ai_response: AI返回的文本
            user_actions: 原始行为数据（用于验证索引有效性）
            start_index: 起始索引（用于分批处理）

        Returns:
            有效行为的索引列表
        """

        valid_indices = []
        for item in result.valid_actions:
            if item.is_valid:
                idx = item.index
                # 验证索引在有效范围内
                if idx is not None and start_index <= idx < start_index + len(
                        user_actions
                ):
                    valid_indices.append(idx)

        return valid_indices

    async def _ai_segment_by_intent(
            self, valid_actions: pd.DataFrame
    ) -> List[List[Dict]]:
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
            return await self._ai_segment_batch(valid_actions, 0)
        else:
            # 分批处理，然后合并分段
            all_segments = []
            for batch_start in range(0, total_actions, MAX_ACTIONS_PER_BATCH):
                batch_end = min(batch_start + MAX_ACTIONS_PER_BATCH, total_actions)
                batch_actions = valid_actions.iloc[batch_start:batch_end]
                batch_segments = await self._ai_segment_batch(batch_actions, batch_start)
                all_segments.extend(batch_segments)
                # 添加延迟避免API限流
                if batch_end < total_actions:
                    await asyncio.sleep(1)
            return all_segments

    async def _ai_segment_batch(
            self, batch_actions: pd.DataFrame, start_index: int, max_retries: int = 3
    ) -> List[List[Dict]]:
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
                "index": start_index + pos_idx,
                "event_name": row.get("event_name", ""),
                "event_time": str(row.get("event_time", "")),
                "extra_info": (
                    str(row.get("extra_info", ""))
                    if pd.notna(row.get("extra_info"))
                    else ""
                ),
            }
            actions_list.append(action_info)

        # 构建prompt
        prompt = self._build_intent_segmentation_prompt(actions_list)

        result = await self.llm_request(prompt, IntentSegmentationOutput)
        segments = self._parse_intent_segments(result, batch_actions, start_index)
        return segments

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
            actions_text += (
                f"{i}. Index: {action['index']}, Event: {action['event_name']}, "
            )
            actions_text += f"Time: {action['event_time']}"
            if action["extra_info"]:
                actions_text += f", Extra Info: {action['extra_info']}"
            actions_text += "\n"

        return jinja2.Template(INTENT_SEGMENTATION).render(
            actions_text=actions_text, actions_count=len(actions_list)
        )

    def _parse_intent_segments(
            self,
            result: IntentSegmentationOutput,
            valid_actions: pd.DataFrame,
            start_index: int = 0,
    ) -> List[List[Dict]]:
        """
        解析AI返回的意图分段结果

        Args:
            ai_response: AI返回的文本
            valid_actions: 有效行为数据（用于验证索引）
            start_index: 起始索引（用于分批处理）

        Returns:
            按意图分段的行为列表
        """

        segments = []
        used_indices = set()  # 跟踪已使用的索引

        for segment_info in result.intent_segments:
            behavior_indices = segment_info.behavior_indices

            # 验证索引有效性并获取行为（考虑start_index偏移）
            valid_indices = [
                idx
                for idx in behavior_indices
                if start_index <= idx < start_index + len(valid_actions)
                   and idx not in used_indices
            ]

            if len(valid_indices) > 0:
                # 获取对应的行为数据（转换为相对索引）
                segment_actions = [
                    valid_actions.iloc[idx - start_index].to_dict()
                    for idx in valid_indices
                ]
                segments.append(segment_actions)
                used_indices.update(valid_indices)

        # 检查是否有未包含的行为
        all_expected_indices = set(range(start_index, start_index + len(valid_actions)))
        missing_indices = all_expected_indices - used_indices

        # 如果有未包含的行为，将它们添加到最后一个段或创建新段
        if len(missing_indices) > 0:
            missing_actions = [
                valid_actions.iloc[idx - start_index].to_dict()
                for idx in sorted(missing_indices)
            ]
            if len(segments) > 0:
                # 添加到最后一个段
                segments[-1].extend(missing_actions)
            else:
                # 创建新段
                segments.append(missing_actions)
        return segments

    def group_user_actions_by_session(
            self, user_actions: pd.DataFrame, session_timeout_minutes: int = 30
    ) -> List[List[Dict]]:
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
                last_time = pd.to_datetime(current_session[-1]["event_time"])
                current_time = pd.to_datetime(row["event_time"])
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
            event_name = action.get("event_name", "")
            event_time = action.get("event_time", "")
            extra_info = action.get("extra_info", "")

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
            "user_uuid": first_action.get("user_uuid", ""),
            "approved_time": str(first_action.get("approved_time", "")),
            "first_payment_time": str(first_action.get("first_payment_time", "")),
            "first_action_time": str(first_action.get("event_time", "")),
            "last_action_time": str(last_action.get("event_time", "")),
            "total_actions": len(user_actions),
            "unique_events": user_actions["event_name"].nunique(),
        }

        return context

    async def analyze_intent(
            self,
            user_context: Dict,
            actions: List[Dict],
            history: Optional[Dict] = None,
            include_operation_recommendation: bool = False,
    ) -> Dict[str, Any]:
        """
        使用Gemini AI分析用户意图

        Args:
            user_context: 用户上下文信息
            actions: 行为数据列表
            history: 历史意图分析结果（可选）
            include_operation_recommendation: 是否包含运营建议（默认False，只生成意图分析）

        Returns:
            意图分析结果
        """
        if include_operation_recommendation:
            prompt = self._build_prompt(user_context, actions, history)
            response_model = IntentAnalysisOutput
        else:
            prompt = self._build_intent_only_prompt(user_context, actions, history)
            response_model = IntentOnlyAnalysisOutput

        response = await self.llm_request(prompt, response_model)
        return response.model_dump()

    def _build_prompt(
            self, user_context: Dict, actions: List[Dict], history: Optional[Dict] = None
    ) -> str:
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

        prompt = jinja2.Template(INTENT_ANALYSIS).render(
            user_context=user_context,
            actions_text=actions_text,
            history_text=history_text,
        )
        return prompt

    def _build_intent_only_prompt(
            self, user_context: Dict, actions: List[Dict], history: Optional[Dict] = None
    ) -> str:
        """
        构建只生成意图分析的prompt（不包含运营建议）

        Args:
            user_context: 用户上下文信息
            actions: 行为数据列表
            history: 历史意图分析结果（可选）

        Returns:
            完整的prompt字符串（不包含运营建议部分）
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

        prompt = jinja2.Template(INTENT_ONLY_ANALYSIS).render(
            user_context=user_context,
            actions_text=actions_text,
            history_text=history_text,
        )

        return prompt

    async def generate_operation_recommendation(
            self, intent_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        基于意图分析结果生成运营建议

        Args:
            intent_result: 已有的意图分析结果

        Returns:
            包含运营建议的完整结果
        """
        prompt = self._build_operation_recommendation_prompt(intent_result)
        response = await self.llm_request(prompt, OperationRecommendationOutput)
        intent_result["operation_recommendation"] = response.operation_recommendation.model_dump()
        return intent_result

    def _build_operation_recommendation_prompt(
            self, intent_result: Dict[str, Any]
    ) -> str:
        """
        构建生成运营建议的prompt

        Args:
            intent_result: 已有的意图分析结果

        Returns:
            prompt字符串
        """
        prompt = jinja2.Template(OPERATION_RECOMMENDATION).render(
            intent_result=intent_result
        )

        return prompt

    async def generate_operation_recommendations_batch(
            self, intent_results_file: str, output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        批量为已有的意图分析结果生成运营建议

        Args:
            intent_results_file: 意图分析结果JSON文件路径
            output_file: 输出文件路径（如果为None，覆盖原文件）

        Returns:
            包含运营建议的完整结果
        """
        import json

        # 加载意图分析结果
        with open(intent_results_file, "r", encoding="utf-8") as f:
            results = json.load(f)

        print(f"加载了 {len(results)} 个用户的意图分析结果")
        print("开始批量生成运营建议...\n")

        total_sessions = 0
        processed_sessions = 0

        # 为每个用户的每个会话生成运营建议
        for user_uuid, user_data in results.items():
            if "sessions" not in user_data:
                continue

            sessions = user_data.get("sessions", [])
            total_sessions += len(sessions)

            print(f"用户 {user_uuid[:8]}... ({len(sessions)} 个会话)")

            for session_idx, session in enumerate(sessions, 1):
                # 如果已经有运营建议，跳过
                if (
                        "operation_recommendation" in session
                        and session["operation_recommendation"]
                ):
                    print(f"  会话 {session_idx}: 已有运营建议，跳过")
                    continue

                print(f"  会话 {session_idx}: 生成运营建议...", end="", flush=True)
                try:
                    updated_session = await self.generate_operation_recommendation(session)
                    sessions[session_idx - 1] = updated_session
                    processed_sessions += 1
                    print(" 完成")
                except Exception as e:
                    print(f" 失败: {e}")
                    continue

                # 添加延迟避免API限流
                await asyncio.sleep(0.5)

            results[user_uuid]["sessions"] = sessions

        # 保存结果
        if output_file is None:
            output_file = intent_results_file

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n批量生成完成！")
        print(f"  总会话数: {total_sessions}")
        print(f"  已处理会话数: {processed_sessions}")
        print(f"  结果已保存到: {output_file}")

        return results

    async def _process_single_user(
            self,
            uuid: str,
            user_df: pd.DataFrame,
            session_timeout_minutes: int,
            include_operation_recommendation: bool,
    ) -> tuple[str, Dict[str, Any]]:
        """
        处理单个用户的意图分析（内部方法，用于并行处理）

        Args:
            uuid: 用户UUID
            user_df: 用户行为数据DataFrame
            session_timeout_minutes: 会话超时时间（分钟）
            include_operation_recommendation: 是否包含运营建议

        Returns:
            (uuid, 分析结果字典) 的元组
        """
        original_count = len(user_df)
        print(f"用户 {uuid[:8]}... 原始行为数: {original_count}")

        # 过滤有效行为
        valid_actions = await self.filter_valid_actions(user_df)
        valid_count = len(valid_actions)
        print(
            f"  过滤后有效行为数: {valid_count} (过滤掉 {original_count - valid_count} 个)"
        )

        if len(valid_actions) == 0:
            print(f"  跳过: 无有效行为")
            return uuid, {
                "user_uuid": uuid,
                "total_sessions": 0,
                "total_actions_original": original_count,
                "total_actions_valid": 0,
                "total_actions_analyzed": 0,
                "sessions": [],
            }

        # 按会话分组（基于时间）
        time_sessions = self.group_user_actions_by_session(
            valid_actions, session_timeout_minutes
        )
        print(f"  时间会话数: {len(time_sessions)}")

        # 对每个时间会话，进一步按意图分段
        all_intent_segments = []
        for session_idx, time_session in enumerate(time_sessions, 1):
            session_df = pd.DataFrame(time_session)
            print(f"    时间会话 {session_idx}: {len(time_session)} 个行为")
            # 按意图一致性分段
            intent_segments = await self.segment_actions_by_intent(session_df)
            print(f"      分段为 {len(intent_segments)} 个意图段")
            for seg_idx, seg in enumerate(intent_segments, 1):
                print(f"        意图段 {seg_idx}: {len(seg)} 个行为")
            all_intent_segments.extend(intent_segments)

        # 验证所有行为都被包含在段中
        total_in_segments = sum(len(seg) for seg in all_intent_segments)
        if total_in_segments != valid_count:
            print(
                f"  ⚠️  警告: 意图段中的行为总数 ({total_in_segments}) 与有效行为数 ({valid_count}) 不一致！"
            )
            print(
                f"  可能的原因: 分段逻辑丢失了 {valid_count - total_in_segments} 个行为"
            )

        # 分析每个意图段
        session_results = []
        history = None

        for segment_idx, segment_actions in enumerate(all_intent_segments):
            user_context = self.get_user_context(pd.DataFrame(segment_actions))

            # 分析意图（可选择是否包含运营建议）
            intent_result = await self.analyze_intent(
                user_context,
                segment_actions,
                history,
                include_operation_recommendation=include_operation_recommendation,
            )
            intent_result["session_index"] = segment_idx
            intent_result["session_size"] = len(segment_actions)
            intent_result["timestamp"] = datetime.now().isoformat()

            session_results.append(intent_result)

            # 更新历史（使用最后一次分析结果）
            history = intent_result

        total_analyzed = sum(len(seg) for seg in all_intent_segments)
        print(
            f"  最终分析的行为总数: {total_analyzed}/{original_count} (原始: {original_count})"
        )
        print()

        return uuid, {
            "user_uuid": uuid,
            "total_sessions": len(all_intent_segments),
            "total_actions_original": original_count,
            "total_actions_valid": valid_count,
            "total_actions_analyzed": total_analyzed,
            "sessions": session_results,
        }

    async def analyze_user_intent(
            self,
            csv_path: Optional[str] = None,
            user_uuids: list[str] | None = None,
            session_timeout_minutes: int = 30,
            preloaded_df: Optional[pd.DataFrame] = None,
            include_operation_recommendation: bool = False,
            max_concurrent: int = 50,
            result_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        分析用户意图（主入口）

        Args:
            csv_path: CSV文件路径（若已提供preloaded_df则可为空）
            user_uuids: 指定用户ID列表（如果为None，分析所有用户）
            session_timeout_minutes: 会话超时时间（分钟）
            preloaded_df: 预加载的数据DataFrame，避免重复读取
            include_operation_recommendation: 是否包含运营建议（默认False，只生成意图分析，可后续批量生成运营建议）
            max_concurrent: 最大并发处理用户数（默认3，避免API限流）
            result_file: 结果文件路径（用于断点续跑，如果提供则跳过已完成的用户）

        Returns:
            分析结果字典
        """
        # 从结果文件中加载已完成的用户列表
        completed_users = self.load_completed_users(result_file)
        if completed_users:
            print(f"从结果文件加载: 已完成 {len(completed_users)} 个用户")

        # 加载数据（优先使用预加载数据以避免重复读盘）
        if preloaded_df is not None:
            df = preloaded_df.copy()
        else:
            if not csv_path:
                return {"error": "缺少数据源(csv_path或preloaded_df)"}
            df = self.load_data(csv_path)

        # 如果指定了用户，只分析该用户
        if user_uuids:
            df = df[df["user_uuid"].isin(user_uuids)]

        if len(df) == 0:
            return {"error": "没有找到用户数据"}

        # 按用户分组
        user_groups = list(df.groupby("user_uuid"))
        total_users = len(user_groups)
        
        # 过滤掉已完成的用户
        if completed_users:
            user_groups = [(uuid, user_df) for uuid, user_df in user_groups if uuid not in completed_users]
            skipped_count = total_users - len(user_groups)
            if skipped_count > 0:
                print(f"跳过已完成的用户: {skipped_count} 个")
        
        remaining_users = len(user_groups)
        print(f"共 {remaining_users} 个用户需要分析，最大并发数: {max_concurrent}")

        if remaining_users == 0:
            print("所有用户已完成分析")
            return {}

        # 使用信号量限制并发数
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_semaphore(uuid: str, user_df: pd.DataFrame):
            """带信号量限制的处理函数"""
            async with semaphore:
                result = await self._process_single_user(
                    uuid, user_df, session_timeout_minutes, include_operation_recommendation
                )
                # 每完成一个用户，立即更新结果文件
                if result_file:
                    self.update_result_file(result_file, uuid, result)
                return result

        # 并行处理所有用户
        tasks = [
            process_with_semaphore(uuid, user_df)
            for uuid, user_df in user_groups
        ]

        # 等待所有任务完成
        results_list = await asyncio.gather(*tasks)

        # 转换为字典格式
        results = {uuid: result for uuid, result in results_list}

        return results


async def main():
    """主函数 - 可以直接运行进行分析"""
    import os
    import json

    # 从环境变量获取API密钥
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("错误: 请设置环境变量 GEMINI_API_KEY")
        print("使用方法: export GEMINI_API_KEY='your-api-key'")
        return

    # 创建分析器
    print("正在初始化分析器...")
    analyzer = IntentAnalyzer(api_key)

    # 检查数据文件
    csv_path = "data.csv"
    if not os.path.exists(csv_path):
        print(f"错误: 找不到文件 {csv_path}")
        return

    # 默认分析第一个用户（作为示例）
    print(f"\n正在加载数据文件: {csv_path}...")
    df = analyzer.load_data(csv_path)
    first_user = df["user_uuid"].iloc[0]

    print(f"\n开始分析用户: {first_user}")
    print("这可能需要一些时间，请耐心等待...\n")

    # 分析单个用户（复用已加载数据以避免重复读盘）
    results = await analyzer.analyze_user_intent(
        user_uuids=[first_user],
        session_timeout_minutes=30,
        preloaded_df=df[df["user_uuid"] == first_user],
    )

    # 保存结果
    output_file = f"intent_result_{first_user[:8]}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 打印结果摘要
    print("\n" + "=" * 60)
    print("分析结果摘要")
    print("=" * 60)

    for user_uuid, user_result in results.items():
        if "sessions" in user_result:
            print(f"\n用户: {user_uuid[:16]}...")
            print(f"  会话数: {user_result.get('total_sessions', 0)}")

            for session in user_result.get("sessions", []):
                intent = session.get("intent", "N/A")
                score = session.get("confidence_score", 0)
                category = session.get("intent_category", "N/A")

                print(f"\n  会话 {session.get('session_index', 0) + 1}:")
                print(f"    意图: {intent}")
                print(f"    类别: {category}")
                print(f"    置信度: {score:.2f}")

                if "key_behaviors" in session:
                    print(f"    关键行为: {', '.join(session['key_behaviors'][:3])}")

    print(f"\n完整结果已保存到: {output_file}")
    print("\n提示: 要分析其他用户或批量分析，请使用 run_analysis.py")


if __name__ == "__main__":
    asyncio.run(main())
