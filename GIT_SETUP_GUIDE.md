# Git设置和上传指南

## 📋 前提条件

### 1. 安装Git
如果您还没有安装Git，请按照以下步骤：

**Windows用户：**
1. 访问 https://git-scm.com/download/windows
2. 下载并安装Git for Windows
3. 安装时选择默认选项即可

**验证安装：**
```bash
git --version
```

### 2. 配置Git用户信息
```bash
git config --global user.name "newhouse10086"
git config --global user.email "1914906669@qq.com"
```

## 🚀 GitHub仓库设置

### 1. 创建GitHub仓库
1. 登录到 https://github.com/newhouse10086
2. 点击右上角的 "+" 按钮，选择 "New repository"
3. 填写仓库信息：
   - **Repository name**: `ViT-CNN-crossview`
   - **Description**: `Advanced Deep Learning Framework for UAV Geo-Localization`
   - **Visibility**: Public（公开）
   - **不要**勾选 "Initialize this repository with a README"
   - **不要**添加 .gitignore 或 license（我们已经创建了）
4. 点击 "Create repository"

### 2. 获取仓库URL
创建后，GitHub会显示仓库URL：
```
https://github.com/newhouse10086/ViT-CNN-crossview.git
```

## 💻 本地Git设置和上传

### 1. 打开命令行
在项目根目录（ViT-CNN-crossview文件夹）中打开命令行：
- Windows: 在文件夹中按住Shift键右键，选择"在此处打开PowerShell窗口"
- 或者使用Git Bash

### 2. 初始化Git仓库
```bash
# 初始化Git仓库
git init

# 设置默认分支为main
git branch -M main
```

### 3. 添加文件到Git
```bash
# 添加所有文件
git add .

# 检查状态
git status
```

### 4. 创建初始提交
```bash
git commit -m "Initial commit: ViT-CNN-crossview framework

- Complete refactoring of FSRA project for PyTorch 2.1
- Hybrid ViT-CNN architecture for cross-view geo-localization
- Robust data handling with dummy dataset support
- Comprehensive metrics and visualization tools
- Professional project structure with full documentation
- Ready for research and production use"
```

### 5. 添加远程仓库
```bash
git remote add origin https://github.com/newhouse10086/ViT-CNN-crossview.git
```

### 6. 推送到GitHub
```bash
git push -u origin main
```

## 🔐 认证设置

### 如果遇到认证问题：

**方法1: 使用Personal Access Token（推荐）**
1. 访问 GitHub Settings > Developer settings > Personal access tokens > Tokens (classic)
2. 点击 "Generate new token (classic)"
3. 设置权限：勾选 `repo` 权限
4. 复制生成的token
5. 在推送时使用token作为密码

**方法2: 使用GitHub CLI**
```bash
# 安装GitHub CLI后
gh auth login
```

## 📝 完整命令序列

以下是完整的命令序列，您可以复制粘贴执行：

```bash
# 1. 进入项目目录
cd ViT-CNN-crossview

# 2. 初始化Git仓库
git init
git branch -M main

# 3. 配置用户信息（如果还没配置）
git config user.name "newhouse10086"
git config user.email "1914906669@qq.com"

# 4. 添加文件
git add .

# 5. 创建提交
git commit -m "Initial commit: ViT-CNN-crossview framework"

# 6. 添加远程仓库
git remote add origin https://github.com/newhouse10086/ViT-CNN-crossview.git

# 7. 推送到GitHub
git push -u origin main
```

## 🎯 验证上传成功

上传成功后，您可以：
1. 访问 https://github.com/newhouse10086/ViT-CNN-crossview
2. 确认所有文件都已上传
3. 检查README.md是否正确显示

## 📚 后续操作

### 添加仓库描述和标签
在GitHub仓库页面：
1. 点击设置图标（齿轮）
2. 添加描述：`Advanced Deep Learning Framework for UAV Geo-Localization`
3. 添加标签：`deep-learning`, `pytorch`, `computer-vision`, `geo-localization`, `uav`, `cross-view`, `vision-transformer`

### 设置仓库主页
1. 确保README.md显示正确
2. 可以添加项目徽章和截图
3. 更新项目链接和联系信息

## ❗ 常见问题

### 1. 推送失败
- 检查网络连接
- 确认GitHub仓库已创建
- 检查认证信息

### 2. 文件过大
- 检查.gitignore是否正确排除了大文件
- 使用Git LFS处理大文件

### 3. 权限问题
- 确认GitHub用户名和邮箱正确
- 使用Personal Access Token进行认证

## 🎉 完成！

完成上传后，您的ViT-CNN-crossview项目将在GitHub上公开可用，其他研究者可以：
- 查看和下载代码
- 提交问题和建议
- 贡献代码改进
- 引用您的工作

祝您上传成功！如有问题，请参考GitHub官方文档或联系技术支持。
