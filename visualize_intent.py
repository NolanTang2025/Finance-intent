"""
意图分析结果可视化工具
"""

import json
import sys
import os
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

def load_intent_results(file_path):
    """加载意图分析结果"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def visualize_intent_results(results, output_dir='intent_visualization'):
    """可视化意图分析结果"""
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 收集所有数据
    all_intents = []
    all_categories = []
    all_scores = []
    user_session_counts = []
    session_sizes = []
    
    for user_uuid, user_data in results.items():
        sessions = user_data.get('sessions', [])
        user_session_counts.append(len(sessions))
        
        for session in sessions:
            intent = session.get('intent', 'Unknown')
            category = session.get('intent_category', 'Unknown')
            score = session.get('confidence_score', 0)
            size = session.get('session_size', 0)
            
            all_intents.append(intent)
            all_categories.append(category)
            all_scores.append(score)
            session_sizes.append(size)
    
    # 1. 意图类别分布饼图
    category_counts = Counter(all_categories)
    if category_counts:
        plt.figure(figsize=(10, 8))
        plt.pie(category_counts.values(), labels=category_counts.keys(), autopct='%1.1f%%', startangle=90)
        plt.title('Intent Category Distribution', fontsize=16, fontweight='bold')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/intent_category_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成意图类别分布图: {output_dir}/intent_category_distribution.png")
    
    # 2. 置信度分布直方图
    if all_scores:
        plt.figure(figsize=(10, 6))
        plt.hist(all_scores, bins=20, edgecolor='black', alpha=0.7)
        plt.xlabel('Confidence Score', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Confidence Score Distribution', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/confidence_score_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成置信度分布图: {output_dir}/confidence_score_distribution.png")
    
    # 3. 每个用户的意图段数量
    if user_session_counts:
        plt.figure(figsize=(10, 6))
        plt.hist(user_session_counts, bins=min(20, max(user_session_counts)), edgecolor='black', alpha=0.7)
        plt.xlabel('Number of Intent Segments per User', fontsize=12)
        plt.ylabel('Number of Users', fontsize=12)
        plt.title('Intent Segments per User Distribution', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/segments_per_user.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成每用户意图段数量图: {output_dir}/segments_per_user.png")
    
    # 4. 每个意图段的行为数量
    if session_sizes:
        plt.figure(figsize=(10, 6))
        plt.hist(session_sizes, bins=min(30, max(session_sizes)), edgecolor='black', alpha=0.7)
        plt.xlabel('Number of Behaviors per Intent Segment', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Behaviors per Intent Segment Distribution', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/behaviors_per_segment.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成每意图段行为数量图: {output_dir}/behaviors_per_segment.png")
    
    # 5. 置信度 vs 意图类别
    if all_categories and all_scores:
        category_scores = defaultdict(list)
        for cat, score in zip(all_categories, all_scores):
            category_scores[cat].append(score)
        
        plt.figure(figsize=(12, 6))
        categories = list(category_scores.keys())
        scores_data = [category_scores[cat] for cat in categories]
        
        bp = plt.boxplot(scores_data, tick_labels=categories, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
        
        plt.xlabel('Intent Category', fontsize=12)
        plt.ylabel('Confidence Score', fontsize=12)
        plt.title('Confidence Score by Intent Category', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/confidence_by_category.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已生成类别置信度对比图: {output_dir}/confidence_by_category.png")
    
    # 6. 生成统计报告
    generate_statistics_report(results, output_dir)
    
    print(f"\n可视化完成！所有图表已保存到: {output_dir}/")

def generate_statistics_report(results, output_dir):
    """生成统计报告"""
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("Intent Analysis Statistics Report")
    report_lines.append("=" * 60)
    report_lines.append("")
    
    total_users = len(results)
    total_sessions = sum(len(user_data.get('sessions', [])) for user_data in results.values())
    
    report_lines.append(f"Total Users: {total_users}")
    report_lines.append(f"Total Intent Segments: {total_sessions}")
    report_lines.append(f"Average Segments per User: {total_sessions/total_users:.2f}" if total_users > 0 else "N/A")
    report_lines.append("")
    
    # 意图类别统计
    all_categories = []
    all_scores = []
    session_sizes = []
    
    for user_data in results.values():
        for session in user_data.get('sessions', []):
            all_categories.append(session.get('intent_category', 'Unknown'))
            all_scores.append(session.get('confidence_score', 0))
            session_sizes.append(session.get('session_size', 0))
    
    if all_categories:
        category_counts = Counter(all_categories)
        report_lines.append("Intent Category Distribution:")
        for category, count in category_counts.most_common():
            percentage = (count / len(all_categories)) * 100
            report_lines.append(f"  {category}: {count} ({percentage:.1f}%)")
        report_lines.append("")
    
    if all_scores:
        report_lines.append("Confidence Score Statistics:")
        report_lines.append(f"  Average: {sum(all_scores)/len(all_scores):.2f}")
        report_lines.append(f"  Min: {min(all_scores):.2f}")
        report_lines.append(f"  Max: {max(all_scores):.2f}")
        report_lines.append("")
    
    if session_sizes:
        report_lines.append("Session Size Statistics:")
        report_lines.append(f"  Average: {sum(session_sizes)/len(session_sizes):.2f}")
        report_lines.append(f"  Min: {min(session_sizes)}")
        report_lines.append(f"  Max: {max(session_sizes)}")
        report_lines.append("")
    
    # 用户详情
    report_lines.append("User Details:")
    report_lines.append("-" * 60)
    for user_uuid, user_data in list(results.items())[:10]:  # 只显示前10个用户
        sessions = user_data.get('sessions', [])
        report_lines.append(f"\nUser: {user_uuid[:16]}...")
        report_lines.append(f"  Total Segments: {len(sessions)}")
        
        for i, session in enumerate(sessions, 1):
            category = session.get('intent_category', 'Unknown')
            score = session.get('confidence_score', 0)
            size = session.get('session_size', 0)
            intent = session.get('intent', 'N/A')[:50]  # 截断长文本
            report_lines.append(f"  Segment {i}: {category} (score: {score:.2f}, size: {size})")
            report_lines.append(f"    Intent: {intent}...")
    
    if len(results) > 10:
        report_lines.append(f"\n... and {len(results) - 10} more users")
    
    # 保存报告
    report_text = "\n".join(report_lines)
    with open(f'{output_dir}/statistics_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"已生成统计报告: {output_dir}/statistics_report.txt")
    print("\n" + report_text)

def main():
    """主函数"""
    if len(sys.argv) < 2:
        # 如果没有指定文件，查找最新的结果文件
        json_files = [f for f in os.listdir('.') if f.startswith('intent_result') and f.endswith('.json')]
        if not json_files:
            print("错误: 未找到意图分析结果文件")
            print("使用方法: python3 visualize_intent.py <intent_result_file.json>")
            sys.exit(1)
        
        # 使用最新的文件
        result_file = sorted(json_files, key=os.path.getmtime, reverse=True)[0]
        print(f"使用最新的结果文件: {result_file}")
    else:
        result_file = sys.argv[1]
    
    if not os.path.exists(result_file):
        print(f"错误: 文件不存在: {result_file}")
        sys.exit(1)
    
    print(f"正在加载结果文件: {result_file}")
    results = load_intent_results(result_file)
    
    print(f"找到 {len(results)} 个用户的分析结果")
    print("正在生成可视化图表...\n")
    
    visualize_intent_results(results)

if __name__ == '__main__':
    main()

