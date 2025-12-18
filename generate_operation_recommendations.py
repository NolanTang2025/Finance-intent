"""
批量生成运营建议
基于已有的意图分析结果，批量生成运营建议
"""

import asyncio
import os
import json
import sys
from intent_analyzer import IntentAnalyzer

async def main():
    """主函数"""
    # 检查API密钥
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("错误: 未找到 GEMINI_API_KEY 环境变量")
        print("请先设置: export GEMINI_API_KEY='your-api-key'")
        sys.exit(1)
    
    # 创建分析器
    print("正在初始化分析器...")
    analyzer = IntentAnalyzer(api_key)
    
    # 询问意图分析结果文件
    print("\n" + "="*60)
    print("批量生成运营建议")
    print("="*60)
    print("\n请输入意图分析结果JSON文件路径:")
    print("(例如: intent_result_00184160.json 或 intent_result_batch_10users.json)")
    
    input_file = input("\n文件路径: ").strip()
    if not input_file:
        print("错误: 文件路径不能为空")
        sys.exit(1)
    
    if not os.path.exists(input_file):
        print(f"错误: 找不到文件 {input_file}")
        sys.exit(1)
    
    # 询问输出文件
    output_file = input("\n输出文件路径（直接回车覆盖原文件）: ").strip()
    if not output_file:
        output_file = input_file
        print(f"将覆盖原文件: {input_file}")
    else:
        print(f"将保存到: {output_file}")
    
    confirm = input("\n确认开始批量生成运营建议？(yes/no): ").strip().lower()
    if confirm != 'yes':
        print("已取消")
        sys.exit(0)
    
    # 批量生成运营建议
    print("\n开始批量生成运营建议...")
    print("这可能需要一些时间，请耐心等待...\n")
    
    try:
        results = await analyzer.generate_operation_recommendations_batch(
            intent_results_file=input_file,
            output_file=output_file
        )
        
        print("\n" + "="*60)
        print("批量生成完成！")
        print("="*60)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())

