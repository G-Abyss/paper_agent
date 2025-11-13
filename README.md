# 📚 Paper Summarizer - 学术论文自动总结系统

一个基于 AI 的自动化工具，用于从 Google 学术邮件推送中提取、翻译和评审相关论文，并生成结构化的日报报告。

## ✨ 功能特性

- 📧 **自动邮件获取**：从 QQ 邮箱自动获取 Google 学术推送邮件
- 🔍 **智能筛选**：基于关键词自动筛选相关论文（遥操作、机器人动力学、力控等领域）
- 🤖 **AI 翻译**：使用本地 LLM（Ollama）进行专业论文翻译
- 📊 **专业评审**：自动生成结构化评审报告，包含多维度评分
- 📝 **日报生成**：自动生成 Markdown 格式的学术论文日报
- ⭐ **价值评估**：自动识别高价值论文（评分>4.0）

## 🛠️ 技术栈

- **Python 3.x**
- **CrewAI** - AI Agent 框架，用于论文翻译和评审
- **Ollama** - 本地大语言模型（支持 qwen2.5:32b 等模型）
- **IMAP** - 邮件协议
- **BeautifulSoup** - HTML 解析
- **YAML** - 配置文件

## 📋 前置要求

1. **Python 3.8+**
2. **Ollama** 已安装并运行（默认地址：`http://localhost:11434`）
3. **已下载所需模型**（如 `qwen2.5:32b`）
4. **QQ 邮箱账号**（需要开启 IMAP 服务）

## 🚀 安装步骤

### 1. 克隆仓库

```bash
git clone <your-repo-url>
cd <your-repo-name>
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

如果没有 `requirements.txt`，请安装以下依赖：

```bash
pip install crewai python-dotenv pyyaml beautifulsoup4 ollama
```

### 3. 配置环境变量

创建 `.env` 文件（不要提交到 Git）：

```env
# 邮箱配置
QMAIL_USER=your_qq_email@qq.com
QMAIL_PASSWORD=your_imap_password

# Ollama 配置
OLLAMA_MODEL=qwen2.5:32b
OLLAMA_BASE_URL=http://localhost:11434

# 其他配置
MAX_EMAILS=20
```

**注意**：QQ 邮箱需要使用授权码作为密码，不是登录密码。获取方式：
1. 登录 QQ 邮箱网页版
2. 设置 → 账户 → 开启 IMAP/SMTP 服务
3. 生成授权码

### 4. 配置关键词

编辑 `keywords.yaml` 文件，根据你的研究领域调整关键词：

```yaml
high_priority:
  - "teleoperation"
  - "robot dynamics"
  - "force control"
  # ... 更多关键词

related:
  - "haptic"
  - "bilateral control"
  # ... 更多相关关键词
```

## 📖 使用方法

### 运行程序

```bash
python run_summarizer.py
```

程序将：
1. 连接 QQ 邮箱
2. 获取最近 1 天的 Google 学术推送邮件
3. 提取论文信息并筛选相关论文
4. 使用 AI 进行翻译和评审
5. 生成日报报告并保存到 `reports/` 目录

### 输出文件

生成的报告保存在 `reports/` 目录下，文件名格式：
```
paper_report_YYYYMMDD_HHMMSS.md
```

报告包含：
- 🔥 **高价值论文**（评分>4.0，建议深入研究）
- 📖 **相关论文**（其他相关论文）
- 📊 **统计信息**（论文数量、平均评分等）

## 📁 项目结构

```
.
├── README.md              # 项目说明文档
├── run_summarizer.py      # 主程序文件
├── keywords.yaml          # 关键词配置文件
├── .env                   # 环境变量配置（不提交到 Git）
└── reports/               # 生成的报告目录
    └── paper_report_*.md  # 日报文件
```

## ⚙️ 配置说明

### 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `QMAIL_USER` | QQ 邮箱账号 | 必填 |
| `QMAIL_PASSWORD` | QQ 邮箱 IMAP 授权码 | 必填 |
| `OLLAMA_MODEL` | Ollama 模型名称 | `qwen2.5:32b` |
| `OLLAMA_BASE_URL` | Ollama 服务地址 | `http://localhost:11434` |
| `MAX_EMAILS` | 最大处理邮件数 | `20` |

### 关键词配置

`keywords.yaml` 包含三类关键词：
- **high_priority**：高优先级关键词（匹配权重更高）
- **related**：相关关键词
- **exclude**：排除关键词（暂未使用）

## 🔧 自定义配置

### 修改邮件搜索条件

在 `run_summarizer.py` 的 `fetch_scholar_emails` 函数中修改搜索条件：

```python
search_criteria = f'(FROM "your_email@gmail.com" SINCE {since_date}) AND (HEADER FROM "scholaralerts-noreply@google.com")'
```

### 修改评分阈值

在 `run_summarizer.py` 的 `process_paper_with_crewai` 函数中修改：

```python
'is_high_value': score_data.get('总分', 0.0) > 4.0  # 修改阈值
```

### 修改评审维度

在 `create_review_task` 函数中修改评审维度和格式。

## ⚠️ 注意事项

1. **隐私安全**：
   - 不要将 `.env` 文件提交到 Git
   - 邮箱授权码请妥善保管

2. **Ollama 模型**：
   - 确保 Ollama 服务正在运行
   - 确保已下载所需的模型（如 `qwen2.5:32b`）
   - 模型大小较大，请确保有足够的磁盘空间

3. **网络连接**：
   - 需要稳定的网络连接访问邮箱服务器
   - 首次运行可能需要较长时间下载模型

4. **处理时间**：
   - 每篇论文的处理时间取决于模型性能和论文长度
   - 建议在非高峰时段运行，避免影响其他工作

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本项目采用 MIT 许可证。

## 🙏 致谢

- [CrewAI](https://github.com/joaomdmoura/crewAI) - AI Agent 框架
- [Ollama](https://ollama.ai/) - 本地大语言模型
- [Google Scholar](https://scholar.google.com/) - 学术论文推送服务

---

**提示**：如果遇到问题，请检查：
1. Ollama 服务是否正常运行
2. 邮箱 IMAP 服务是否已开启
3. 环境变量是否正确配置
4. Python 依赖是否完整安装

