"""
用户意图分析系统 - 基于Gemini AI（单次调用版本）
分析金融信用卡行业用户行为数据，判断用户意图
将所有AI调用合并为一次调用以提高效率
"""

import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional, TypeVar

import jinja2
import pandas as pd
from google import genai
from google.genai.types import GenerateContentConfig
from pydantic import BaseModel

# 导入网络连接错误类型
try:
    from aiohttp.client_exceptions import ClientConnectorError
except ImportError:
    ClientConnectorError = Exception

from prompts import (
    COMPREHENSIVE_INTENT_ANALYSIS,
    ComprehensiveIntentAnalysisOutput,
    OPERATION_RECOMMENDATION,
    OperationRecommendationOutput,
)

T = TypeVar("T", bound=BaseModel)


class IntentAnalyzer:
    """用户意图分析器（单次调用版本）"""

    def __init__(self, gemini_api_key: str):
        """
        初始化分析器

        Args:
            gemini_api_key: Google Gemini API密钥
        """
        self.client = genai.Client(api_key=gemini_api_key).aio
        self.model_name = "gemini-2.5-flash"

    async def llm_request(self, prompt: str, response_model: type[T], max_retries: int = 3) -> T:
        """
        发送LLM请求，带重试机制处理网络错误
        
        Args:
            prompt: 提示词
            response_model: 响应模型
            max_retries: 最大重试次数（默认3次）
            
        Returns:
            解析后的响应
        """
        last_error = None
        
        for attempt in range(max_retries):
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
                
            except (ClientConnectorError, ConnectionResetError, ConnectionError, OSError) as e:
                # 网络连接错误，应该重试
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 指数退避：1秒、2秒、4秒
                    print(f"  网络连接错误，等待 {wait_time} 秒后重试 (尝试 {attempt + 1}/{max_retries})...")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    print(f"  重试 {max_retries} 次后仍然无法连接")
                    raise
                    
            except Exception as e:
                # 其他类型的错误，直接抛出
                raise
        
        # 如果所有重试都失败了
        if last_error:
            raise last_error

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

            line = f"{i}. Index: {i-1}, 时间: {event_time}, 事件: {event_name}"
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

    async def comprehensive_analyze(
            self,
            user_context: Dict,
            actions: List[Dict],
            history: Optional[Dict] = None,
    ) -> ComprehensiveIntentAnalysisOutput:
        """
        一次性完成：过滤有效行为 + 按意图分段 + 分析每个意图段
        
        Args:
            user_context: 用户上下文信息
            actions: 所有行为数据列表（包含索引）
            history: 历史意图分析结果（可选）
            
        Returns:
            合并分析结果
        """
        # 格式化行为数据
        actions_text = self.format_actions_for_prompt(actions)
        
        # 构建历史文本
        history_text = ""
        if history:
            history_text = f"""
历史意图分析:
- 之前意图: {history.get('intent', '无')}
- 之前得分: {history.get('confidence_score', history.get('score', 0.0))}
- 之前分析时间: {history.get('timestamp', '无')}
"""
        
        # 构建合并prompt
        prompt = jinja2.Template(COMPREHENSIVE_INTENT_ANALYSIS).render(
            user_context=user_context,
            actions_text=actions_text,
            actions_count=len(actions),
            history_text=history_text,
        )
        
        # 调用AI（只调用一次）
        result = await self.llm_request(prompt, ComprehensiveIntentAnalysisOutput)
        return result

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

                # 减少延迟时间，提高处理速度
                await asyncio.sleep(0.2)

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
        使用单次AI调用完成所有分析

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

        if len(user_df) == 0:
            print(f"  跳过: 无行为数据")
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
            user_df, session_timeout_minutes
        )
        print(f"  时间会话数: {len(time_sessions)}")

        # 对每个时间会话，使用单次AI调用完成：过滤 + 分段 + 分析
        all_session_results = []
        
        for session_idx, time_session in enumerate(time_sessions, 1):
            print(f"    时间会话 {session_idx}: {len(time_session)} 个行为")
            
            # 准备用户上下文
            session_df = pd.DataFrame(time_session)
            user_context = self.get_user_context(session_df)
            
            # 单次AI调用：完成过滤、分段和分析
            try:
                comprehensive_result = await self.comprehensive_analyze(
                    user_context,
                    time_session,
                    history=None,  # 每个时间会话独立分析，不考虑历史
                )
                
                # 提取有效行为数
                valid_count = len(comprehensive_result.valid_action_indices)
                print(f"      有效行为数: {valid_count}")
                print(f"      意图段数: {len(comprehensive_result.intent_segments)}")
                
                # 转换每个意图段为结果格式
                for seg in comprehensive_result.intent_segments:
                    intent_result = {
                        "intent": seg.intent,
                        "intent_category": seg.intent_category,
                        "confidence_score": seg.confidence_score,
                        "certainty_level": seg.certainty_level,
                        "evidence_quality": seg.evidence_quality,
                        "explored_feature": seg.explored_feature,
                        "exploration_purpose": seg.exploration_purpose,
                        "first_transaction_connection": seg.first_transaction_connection,
                        "baseline_trust": seg.baseline_trust,
                        "trust_indicators": seg.trust_indicators,
                        "concerns": [c.model_dump() for c in seg.concerns],
                        "psychological_reference": seg.psychological_reference.model_dump(),
                        "key_behaviors": seg.key_behaviors,
                        "reasoning": seg.reasoning,
                        "next_action_prediction": seg.next_action_prediction,
                        "session_index": seg.segment_index,
                        "session_size": len(seg.valid_action_indices),
                        "timestamp": datetime.now().isoformat(),
                    }
                    
                    # 如果需要运营建议，单独生成
                    if include_operation_recommendation:
                        intent_result = await self.generate_operation_recommendation(intent_result)
                    
                    all_session_results.append(intent_result)
                    
            except Exception as e:
                print(f"      分析失败: {e}")
                continue

        # 计算统计信息
        total_valid = sum(
            result.get("session_size", 0)
            for result in all_session_results
        )
        
        print(
            f"  最终分析的行为总数: {total_valid}/{original_count} (原始: {original_count})"
        )
        print()

        return uuid, {
            "user_uuid": uuid,
            "total_sessions": len(all_session_results),
            "total_actions_original": original_count,
            "total_actions_valid": total_valid,
            "total_actions_analyzed": total_valid,
            "sessions": all_session_results,
        }

    async def analyze_user_intent(
            self,
            csv_path: Optional[str] = None,
            user_uuids: Optional[List[str]] = None,
            session_timeout_minutes: int = 30,
            preloaded_df: Optional[pd.DataFrame] = None,
            include_operation_recommendation: bool = False,
            max_concurrent: int = 15,
    ) -> Dict[str, Any]:
        """
        分析用户意图（主入口）

        Args:
            csv_path: CSV文件路径（若已提供preloaded_df则可为空）
            user_uuids: 指定用户ID列表（如果为None，分析所有用户）
            session_timeout_minutes: 会话超时时间（分钟）
            preloaded_df: 预加载的数据DataFrame，避免重复读取
            include_operation_recommendation: 是否包含运营建议（默认False，只生成意图分析，可后续批量生成运营建议）
            max_concurrent: 最大并发处理用户数（默认15）

        Returns:
            分析结果字典
        """
        # 加载数据（优先使用预加载数据以避免重复读盘）
        if preloaded_df is not None:
            df = preloaded_df.copy()
        else:
            if not csv_path:
                return {"error": "缺少数据源(csv_path或preloaded_df)"}
            df = self.load_data(csv_path)

        # 如果指定了用户，只分析该用户
        if user_uuids is not None and len(user_uuids) > 0:
            df = df[df["user_uuid"].isin(user_uuids)]

        if len(df) == 0:
            return {"error": "没有找到用户数据"}

        # 按用户分组
        user_groups = list(df.groupby("user_uuid"))
        total_users = len(user_groups)
        print(f"共 {total_users} 个用户需要分析，最大并发数: {max_concurrent}")
        print("注意：本版本使用单次AI调用完成过滤、分段和分析，大幅减少API调用次数\n")

        # 使用信号量限制并发数
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_semaphore(uuid: str, user_df: pd.DataFrame):
            """带信号量限制的处理函数"""
            async with semaphore:
                return await self._process_single_user(
                    uuid, user_df, session_timeout_minutes, include_operation_recommendation
                )

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
    print("正在初始化分析器（单次调用版本）...")
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

