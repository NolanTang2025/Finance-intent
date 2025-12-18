#!/usr/bin/env python3
"""
启动可视化网页服务器
在浏览器中打开 http://localhost:8000/intent_visualizer.html 查看可视化结果
"""

import http.server
import socketserver
import webbrowser
import os
import sys
import socket

PORT_START = 8000
PORT_RANGE = 10  # 尝试8000-8009

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # 添加CORS头，允许跨域访问
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

def is_port_available(port):
    """检查端口是否可用"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('', port))
            return True
        except OSError:
            return False

def find_available_port():
    """查找可用端口"""
    for port in range(PORT_START, PORT_START + PORT_RANGE):
        if is_port_available(port):
            return port
    return None

def main():
    # 切换到脚本所在目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # 检查HTML文件是否存在
    if not os.path.exists('intent_visualizer.html'):
        print("❌ 错误: 找不到 intent_visualizer.html 文件")
        sys.exit(1)
    
    # 查找可用端口
    port = find_available_port()
    if port is None:
        print(f"❌ 错误: 端口 {PORT_START}-{PORT_START + PORT_RANGE - 1} 都被占用了")
        print("   请关闭其他占用端口的程序，或手动指定端口")
        sys.exit(1)
    
    if port != PORT_START:
        print(f"⚠️  端口 {PORT_START} 被占用，使用端口 {port}")
    
    # 启动服务器
    with socketserver.TCPServer(("", port), MyHTTPRequestHandler) as httpd:
        url = f"http://localhost:{port}/intent_visualizer.html"
        print("=" * 60)
        print("🚀 可视化服务器已启动！")
        print("=" * 60)
        print(f"📊 访问地址: {url}")
        print(f"📁 工作目录: {os.getcwd()}")
        print("\n💡 使用说明:")
        print("   1. 在网页中点击 '选择意图分析结果文件' 按钮")
        print("   2. 选择你的 JSON 结果文件（如 intent_result_*.json）")
        print("   3. 系统会自动加载并可视化数据")
        print("\n按 Ctrl+C 停止服务器")
        print("=" * 60)
        
        # 自动打开浏览器
        try:
            webbrowser.open(url)
            print("✅ 已自动打开浏览器")
        except Exception as e:
            print(f"⚠️  无法自动打开浏览器: {e}")
            print(f"   请手动访问: {url}")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n👋 服务器已停止")

if __name__ == "__main__":
    main()

