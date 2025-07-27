# Git上传指南 - ViT-CNN-crossview

## 🚀 快速上传步骤

### 方法1：使用自动化脚本（推荐）

```bash
# 运行Git设置脚本
python scripts/setup_git.py
```

### 方法2：手动步骤

#### 步骤1：配置Git
```bash
git config --global user.name "newhouse10086"
git config --global user.email "1914906669@qq.com"
```

#### 步骤2：初始化仓库
```bash
# 确保在项目根目录
cd ViT-CNN-crossview

# 初始化Git仓库
git init

# 添加所有文件
git add .

# 创建初始提交
git commit -m "Initial commit: ViT-CNN-crossview framework

- Complete refactoring of FSRA project for PyTorch 2.1
- Hybrid ViT-CNN architecture for cross-view geo-localization
- Advanced data handling with dummy dataset support
- Comprehensive metrics and visualization tools
- Professional project structure and documentation"
```

#### 步骤3：创建GitHub仓库

1. 访问 https://github.com/newhouse10086
2. 点击 "New repository"
3. 仓库名称：`ViT-CNN-crossview`
4. 描述：`Advanced Deep Learning Framework for UAV Geo-Localization`
5. 设置为 **Public** 仓库
6. **不要**勾选 "Initialize with README"、".gitignore" 或 "license"
7. 点击 "Create repository"

#### 步骤4：连接远程仓库
```bash
# 添加远程仓库
git remote add origin https://github.com/newhouse10086/ViT-CNN-crossview.git

# 设置主分支
git branch -M main

# 推送到GitHub
git push -u origin main
```

## 🔧 可能遇到的问题和解决方案

### 问题1：认证失败
如果推送时遇到认证问题：

**解决方案1：使用Personal Access Token**
1. 访问 GitHub Settings > Developer settings > Personal access tokens
2. 生成新的token，选择 `repo` 权限
3. 使用token作为密码：
```bash
git push -u origin main
# Username: newhouse10086
# Password: [输入你的Personal Access Token]
```

**解决方案2：使用SSH密钥**
```bash
# 生成SSH密钥
ssh-keygen -t rsa -b 4096 -C "1914906669@qq.com"

# 添加到SSH agent
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_rsa

# 复制公钥到GitHub
cat ~/.ssh/id_rsa.pub
# 将输出复制到 GitHub Settings > SSH and GPG keys

# 使用SSH URL
git remote set-url origin git@github.com:newhouse10086/ViT-CNN-crossview.git
git push -u origin main
```

### 问题2：文件太大
如果有大文件导致推送失败：

```bash
# 检查大文件
find . -size +100M -type f

# 使用Git LFS（如果需要）
git lfs install
git lfs track "*.pth"
git lfs track "*.pt"
git add .gitattributes
git commit -m "Add Git LFS tracking"
```

### 问题3：推送被拒绝
如果远程仓库有冲突：

```bash
# 拉取远程更改
git pull origin main --allow-unrelated-histories

# 解决冲突后重新推送
git push -u origin main
```

## 📝 推荐的提交信息格式

```bash
# 功能添加
git commit -m "feat: add community clustering module"

# 错误修复
git commit -m "fix: resolve PyTorch 2.1 compatibility issues"

# 文档更新
git commit -m "docs: update README with installation guide"

# 重构
git commit -m "refactor: modernize data loading pipeline"

# 性能优化
git commit -m "perf: optimize ViT-CNN forward pass"
```

## 🎯 上传后的GitHub仓库设置

### 1. 添加仓库描述和标签
在GitHub仓库页面：
- **Description**: `Advanced Deep Learning Framework for UAV Geo-Localization`
- **Website**: 可以留空或添加相关链接
- **Topics**: `deep-learning`, `pytorch`, `computer-vision`, `geo-localization`, `uav`, `cross-view`, `vision-transformer`, `cnn`

### 2. 创建Release
```bash
# 创建标签
git tag -a v1.0.0 -m "Initial release: ViT-CNN-crossview v1.0.0"
git push origin v1.0.0
```

### 3. 设置分支保护（可选）
在GitHub仓库设置中：
- Settings > Branches
- 添加规则保护 `main` 分支

## 📊 验证上传成功

上传成功后，您的仓库应该包含：

```
ViT-CNN-crossview/
├── 📁 src/                    # 源代码
├── 📁 config/                 # 配置文件
├── 📁 scripts/                # 脚本文件
├── 📁 data/                   # 数据目录（只有README）
├── 📄 train.py               # 主训练脚本
├── 📄 quick_start.py         # 快速启动
├── 📄 test_project.py        # 测试脚本
├── 📄 requirements.txt       # 依赖
├── 📄 environment.yml        # Conda环境
├── 📄 README.md             # 项目文档
├── 📄 PROJECT_SUMMARY.md    # 项目总结
├── 📄 LICENSE               # MIT许可证
└── 📄 .gitignore            # Git忽略文件
```

## 🎉 完成后的步骤

1. **验证仓库**：访问 https://github.com/newhouse10086/ViT-CNN-crossview
2. **检查README**：确保README.md正确显示
3. **测试克隆**：
   ```bash
   git clone https://github.com/newhouse10086/ViT-CNN-crossview.git
   cd ViT-CNN-crossview
   python quick_start.py test
   ```

## 📞 需要帮助？

如果遇到问题：
1. 检查网络连接
2. 确认GitHub账户权限
3. 查看Git错误信息
4. 参考GitHub官方文档

---

**恭喜！** 您的ViT-CNN-crossview项目现在已经成功上传到GitHub！🎉
