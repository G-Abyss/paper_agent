# 📚 Paper Summarizer - 学术论文自动总结系统

一个基于 AI 的自动化工具，用于从 Google 学术邮件推送中提取、翻译和评审相关论文，并生成结构化的日报报告。

## ✨ 功能特性

- 📧 **自动邮件获取**：从 QQ 邮箱自动获取 Google 学术推送邮件，支持自定义日期范围
- 🔍 **智能筛选**：基于关键词自动筛选相关论文（遥操作、机器人动力学、力控、灵巧手等领域）
- 🤖 **AI 翻译**：使用 CrewAI 框架和本地 LLM（Ollama）进行专业论文翻译
- 📊 **专业评审**：自动生成结构化评审报告，包含多维度评分（创新性、技术深度、相关性、实用性、研究质量）
- 📝 **日报生成**：自动生成 Markdown 格式的学术论文日报
- ⭐ **价值评估**：自动识别高价值论文（评分>4.0），并单独导出到 Excel
- 📊 **Excel 导出**：高价值论文自动导出为 Excel 表格，方便管理和追踪
- 💾 **备份支持**：支持将报告自动备份到指定目录

## 🛠️ 技术栈

- **Python 3.x**
- **CrewAI** - AI Agent 框架，用于论文翻译和评审
- **Ollama** - 本地大语言模型（支持 qwen2.5:32b 等模型）
- **IMAP** - 邮件协议
- **BeautifulSoup** - HTML 解析
- **YAML** - 配置文件
- **Pandas** - 数据处理
- **OpenPyXL** - Excel 文件生成

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
pip install crewai python-dotenv pyyaml beautifulsoup4 ollama pandas openpyxl
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
MAX_EMAILS=30

# 日期范围配置（可选）
# START_DAYS: 开始日期（前START_DAYS天，例如START_DAYS=3表示从前3天开始）
# END_DAYS: 结束日期（前END_DAYS天，例如END_DAYS=0表示到今天，END_DAYS=1表示到昨天）
START_DAYS=1
END_DAYS=0

# 备份目录配置（可选）
# 如果设置了此路径，报告会同时保存到该路径
BACKUP_DIR=
```

**注意**：QQ 邮箱需要使用授权码作为密码，不是登录密码。获取方式：
1. 登录 QQ 邮箱网页版
2. 设置 → 账户 → 开启 IMAP/SMTP 服务
3. 生成授权码

### 4. 配置关键词

编辑 `keywords.yaml` 文件，根据你的研究领域调整关键词。文件包含三类关键词：

- **high_priority**：高优先级关键词（匹配权重更高，权重×2）
  - 遥操作相关：teleoperation、遥操作、remote manipulation 等
  - 机器人动力学相关：robot dynamics、dynamic identification、参数辨识 等
  - 力控相关：force control、impedance control、力控 等
  - 灵巧手相关：dexterous manipulation、dexterous hand、灵巧手、抓取 等

- **related**：相关关键词（权重×1）
  - 触觉相关：haptic、haptics、触觉、tactile 等
  - 机器学习相关：reinforcement learning、deep learning、强化学习、深度强化学习 等
  - 其他相关领域关键词

- **exclude**：排除关键词（暂未使用）

论文相关性分数 = 高优先级关键词匹配数 × 2 + 相关关键词匹配数

## 📖 使用方法

### 运行程序

```bash
python run_summarizer.py
```

程序将：
1. 连接 QQ 邮箱
2. 根据配置的日期范围获取 Google 学术推送邮件（默认：前1天到今天）
3. 提取论文信息并基于关键词筛选相关论文
4. 使用 CrewAI 框架和 AI 进行专业翻译和评审
5. 生成日报报告并保存到 `reports/` 目录
6. 将高价值论文（评分>4.0）导出到 Excel 文件

### 输出文件

生成的报告保存在 `reports/` 目录下，文件名格式：
```
Robotics_Academic_Daily_YYYYMMDD.md
```

报告包含：
- 🔥 **高价值论文**（评分>4.0，建议深入研究）
  - 包含完整的评审内容（核心贡献、技术方法、相关性分析、技术价值、值得关注的原因）
  - 包含详细的评分信息（JSON 格式）
- 📖 **相关论文**（其他相关论文）
  - 包含评审内容和评分信息
- 📊 **统计信息**（论文数量、平均评分等）

同时会生成 Excel 文件：
```
高价值论文_YYYYMMDD.xlsx
```

Excel 文件包含高价值论文的详细信息：
- 论文标题
- 论文链接
- 总分及各维度评分（创新性、技术深度、相关性、实用性、研究质量）
- 评分理由

## 📁 项目结构

```
.
├── README.md                    # 项目说明文档
├── run_summarizer.py            # 主程序文件
├── keywords.yaml                # 关键词配置文件
├── requirements.txt             # Python 依赖列表
├── .env                         # 环境变量配置（不提交到 Git）
└── reports/                     # 生成的报告目录
    ├── Robotics_Academic_Daily_*.md  # 日报文件（Markdown 格式）
    └── 高价值论文_*.xlsx        # 高价值论文 Excel 文件
```

## ⚙️ 配置说明

### 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `QMAIL_USER` | QQ 邮箱账号 | 必填 |
| `QMAIL_PASSWORD` | QQ 邮箱 IMAP 授权码 | 必填 |
| `OLLAMA_MODEL` | Ollama 模型名称 | `qwen2.5:32b` |
| `OLLAMA_BASE_URL` | Ollama 服务地址 | `http://localhost:11434` |
| `MAX_EMAILS` | 最大处理邮件数 | `30` |
| `START_DAYS` | 开始日期（前START_DAYS天） | `1` |
| `END_DAYS` | 结束日期（前END_DAYS天，0表示今天） | `0` |
| `BACKUP_DIR` | 备份目录路径（可选） | 空（不备份） |

### 关键词配置

`keywords.yaml` 包含三类关键词：
- **high_priority**：高优先级关键词（匹配权重×2）
  - 包括：遥操作、机器人动力学、力控、灵巧手等核心领域关键词
- **related**：相关关键词（匹配权重×1）
  - 包括：触觉、机器学习、强化学习等相关领域关键词
- **exclude**：排除关键词（暂未使用）

论文相关性分数计算：`高优先级关键词匹配数 × 2 + 相关关键词匹配数`

## 🔧 自定义配置

### 修改日期范围

通过环境变量配置：
```env
START_DAYS=3  # 从前3天开始
END_DAYS=0    # 到今天结束
```

或在代码中直接修改：
```python
START_DAYS = 1  # 从前1天开始
END_DAYS = 0    # 到今天结束
```

### 修改邮件搜索条件

在 `run_summarizer.py` 的 `fetch_scholar_emails` 函数中修改搜索条件：

```python
search_criteria = f'(FROM "scholaralerts-noreply@google.com" SINCE {start_date_str} BEFORE {end_date_str})'
```

### 修改评分阈值

在 `run_summarizer.py` 的 `process_paper_with_crewai` 函数中修改：

```python
'is_high_value': score_data.get('总分', 0.0) > 4.0  # 修改阈值
```

### 修改评审维度

在 `create_review_task` 函数中修改评审维度和格式。当前评审维度包括：
- 创新性（0.0-1.0）
- 技术深度（0.0-1.0）
- 相关性（0.0-1.0）
- 实用性（0.0-1.0）
- 研究质量（0.0-1.0）
- 总分（0.0-5.0）

### 配置备份目录

在 `.env` 文件中设置：
```env
BACKUP_DIR=/path/to/backup/directory
```

报告将同时保存到 `reports/` 目录和备份目录。

## ⚠️ 注意事项

1. **隐私安全**：
   - 不要将 `.env` 文件提交到 Git
   - 邮箱授权码请妥善保管
   - 备份目录中的报告也包含敏感信息，请注意保护

2. **Ollama 模型**：
   - 确保 Ollama 服务正在运行
   - 确保已下载所需的模型（如 `qwen2.5:32b`）
   - 模型大小较大（32B 模型约 20GB+），请确保有足够的磁盘空间和内存
   - 建议使用性能较好的 GPU 以加快处理速度

3. **网络连接**：
   - 需要稳定的网络连接访问邮箱服务器
   - 首次运行可能需要较长时间下载模型

4. **处理时间**：
   - 每篇论文的处理时间取决于模型性能和论文长度
   - 使用 CrewAI 框架，每篇论文需要经过翻译和评审两个步骤
   - 建议在非高峰时段运行，避免影响其他工作
   - 处理大量论文时可能需要较长时间（每篇约 1-3 分钟）

5. **CrewAI 配置**：
   - 程序已禁用 CrewAI 遥测功能
   - 确保 CrewAI 版本 >= 0.1.0
   - 如果遇到连接问题，检查 `OLLAMA_BASE_URL` 配置是否正确

6. **Excel 导出**：
   - 需要安装 `openpyxl` 库
   - Excel 文件会自动调整列宽以便阅读
   - 如果导出失败，检查是否有写入权限

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

