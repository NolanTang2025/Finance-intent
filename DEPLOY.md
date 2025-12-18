# Vercel 部署指南

这个可视化网页是完全静态的，可以直接部署到 Vercel。

## 快速部署

### 方法1: 通过 Vercel CLI（推荐）

1. **安装 Vercel CLI**
```bash
npm i -g vercel
```

2. **登录 Vercel**
```bash
vercel login
```

3. **部署**
```bash
vercel
```

4. **生产环境部署**
```bash
vercel --prod
```

### 方法2: 通过 GitHub + Vercel（推荐用于持续部署）

1. **将代码推送到 GitHub**
```bash
git init
git add intent_visualizer.html vercel.json .vercelignore
git commit -m "Add visualization page"
git remote add origin <your-github-repo-url>
git push -u origin main
```

2. **在 Vercel 中导入项目**
   - 访问 [vercel.com](https://vercel.com)
   - 点击 "New Project"
   - 选择你的 GitHub 仓库
   - Vercel 会自动检测配置并部署

### 方法3: 通过 Vercel Dashboard

1. 访问 [vercel.com](https://vercel.com)
2. 点击 "New Project"
3. 选择 "Upload" 或连接 Git 仓库
4. 上传 `intent_visualizer.html` 文件
5. 点击 "Deploy"

## 配置说明

### vercel.json
- 配置了路由规则，确保访问根路径时显示可视化页面
- 设置了 CORS 头，允许跨域访问

### .vercelignore
- 排除了不需要部署的文件（Python 脚本、数据文件等）
- 只部署必要的 HTML 文件

## 部署后的使用

部署完成后，你可以：

1. **访问你的 Vercel URL**（如 `https://your-project.vercel.app`）
2. **上传 JSON 文件**：点击页面上的 "选择意图分析结果文件" 按钮
3. **选择你的分析结果文件**（如 `intent_result_*.json`）
4. **查看可视化结果**

## 注意事项

1. **完全静态**：这个页面不需要后端服务器，所有处理都在浏览器中完成
2. **数据隐私**：上传的 JSON 文件只在浏览器中处理，不会上传到服务器
3. **文件大小**：建议 JSON 文件不超过 50MB，以确保良好的性能
4. **浏览器兼容性**：需要现代浏览器支持（Chrome、Firefox、Safari、Edge 最新版本）

## 自定义域名

部署后，你可以在 Vercel Dashboard 中：
1. 进入项目设置
2. 选择 "Domains"
3. 添加你的自定义域名

## 更新部署

如果使用 Git 方式部署，每次推送到主分支都会自动重新部署。

如果使用 CLI 方式，运行 `vercel --prod` 即可更新。

