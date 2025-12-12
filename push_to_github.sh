#!/bin/bash

# 推送到GitHub的脚本

echo "=========================================="
echo "推送到 GitHub"
echo "=========================================="
echo ""

# 检查是否已设置远程仓库
if git remote | grep -q "origin"; then
    echo "✓ 已检测到远程仓库:"
    git remote -v
    echo ""
    read -p "是否使用现有远程仓库? (y/n): " use_existing
    if [ "$use_existing" != "y" ]; then
        read -p "请输入GitHub仓库URL (例如: https://github.com/username/repo.git): " repo_url
        if [ -n "$repo_url" ]; then
            git remote set-url origin "$repo_url"
            echo "✓ 已更新远程仓库URL"
        fi
    fi
else
    echo "未检测到远程仓库，请设置GitHub仓库URL"
    read -p "请输入GitHub仓库URL (例如: https://github.com/username/repo.git): " repo_url
    if [ -n "$repo_url" ]; then
        git remote add origin "$repo_url"
        echo "✓ 已添加远程仓库"
    else
        echo "❌ 错误: 未提供仓库URL"
        exit 1
    fi
fi

echo ""
echo "正在推送到GitHub..."
echo ""

# 获取当前分支名
branch=$(git branch --show-current)

# 推送
git push -u origin "$branch"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 成功推送到GitHub!"
    echo ""
    echo "你的HTML页面现在可以在GitHub Pages上访问了:"
    echo "https://$(git remote get-url origin | sed 's/.*github.com[:/]\(.*\)\.git/\1/')/blob/$branch/intent_visualizer.html"
else
    echo ""
    echo "❌ 推送失败，请检查:"
    echo "1. GitHub仓库是否存在"
    echo "2. 是否有推送权限"
    echo "3. 网络连接是否正常"
fi

