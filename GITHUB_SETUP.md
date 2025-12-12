# 推送到 GitHub 的步骤

## 方法1: 使用脚本（推荐）

运行以下命令：

```bash
./push_to_github.sh
```

脚本会引导你输入GitHub仓库URL并自动推送。

## 方法2: 手动操作

### 1. 在GitHub上创建新仓库

1. 访问 https://github.com/new
2. 输入仓库名称（例如：`YUP-intent`）
3. 选择 Public 或 Private
4. **不要**初始化README、.gitignore或license（我们已经有了）
5. 点击 "Create repository"

### 2. 添加远程仓库并推送

```bash
# 添加远程仓库（替换为你的GitHub用户名和仓库名）
git remote add origin https://github.com/你的用户名/YUP-intent.git

# 或者使用SSH（如果你配置了SSH密钥）
git remote add origin git@github.com:你的用户名/YUP-intent.git

# 推送到GitHub
git push -u origin main
```

### 3. 如果遇到分支名问题

如果GitHub默认分支是 `main` 而你的本地是 `master`，可以：

```bash
# 重命名本地分支
git branch -M main

# 然后推送
git push -u origin main
```

## 启用 GitHub Pages（可选）

如果你想在线访问HTML页面：

1. 进入你的GitHub仓库
2. 点击 Settings → Pages
3. 在 Source 下选择分支（通常是 `main`）
4. 选择 `/ (root)` 文件夹
5. 点击 Save

然后可以通过以下URL访问：
`https://你的用户名.github.io/YUP-intent/intent_visualizer.html`

## 当前已提交的文件

- ✅ `intent_visualizer.html` - HTML可视化页面
- ✅ `.gitignore` - Git忽略文件配置

## 注意事项

- `.gitignore` 已配置，不会推送敏感数据（如API密钥、CSV文件等）
- 如果需要推送其他文件，可以使用 `git add <文件名>` 然后 `git commit`

