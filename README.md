# 🎬 DYY SRT 智能翻译工具 (术语协同版)

这是一项基于 **Streamlit** 和 **DeepSeek (OpenAI API 兼容)** 构建的高级影视字幕翻译解决方案。它不仅能够实现高质量的英中翻译，还特别针对影视翻译中的"角色断代"、"术语不统一"以及"翻译成本高"等痛点进行了深度优化。

## ✨ 核心特性

- **🧠 智能语境感知**：采用滑动窗口机制，翻译当前批次时会自动带入前一幕的对白作为参考，确保对话逻辑连贯。
- **📚 术语一致性保障**：
    - **自动提取**：AI 会预先阅读前 150 句对白，自动提取核心人名、地名及专有名词。
    - **手动干预**：支持用户自定义影片背景、剧情风格及术语对照表。
- **💰 降本增效逻辑**：
    - **动态切片**：根据字幕条数和时间戳停顿（换场）智能分批，最大化单次 API 请求的信息量，大幅节省 Token 消耗。
    - **零容错校验**：严格校验 AI 返回的 JSON 格式，若发生漏翻自动触发重试。
- **🛡️ 断点续传与缓存**：基于文件内容和配置生成 MD5 哈希缓存。即使程序中断或网页刷新，也能从上次停止的地方继续，无需重复扣费。
- **⚡ 并发加速**：支持多线程并发请求，显著提升长视频字幕的翻译速度。

## 📂 项目结构

```
├── app.py                     # Streamlit Web UI 入口
├── src/
│   ├── __init__.py
│   ├── core/                  # 核心数据模型与通用工具
│   │   ├── __init__.py
│   │   ├── constants.py       # 共享配置常量（批次大小、重试次数等）
│   │   ├── models.py          # SubtitleEntry 数据类定义
│   │   ├── text_processor.py  # 中文检测、智能换行、ASS 标签处理
│   │   └── chunker.py         # 场景感知字幕分批算法
│   ├── parsers/               # 字幕文件解析层
│   │   ├── __init__.py
│   │   ├── srt_parser.py      # SRT 文件解析与序列化
│   │   └── ass_parser.py      # ASS/SSA 文件解析与序列化
│   └── translator/            # 翻译业务逻辑层
│       ├── __init__.py
│       ├── prompts.py         # System / User Prompt 构建器
│       ├── engine.py          # API 调用、JSON 校验、重试与单行隔离降级
│       ├── cache.py           # MD5 本地缓存（断点续传）
│       └── glossary.py        # AI 自动提取专有名词术语表
├── .srt_cache/                # 自动生成，存放本地缓存文件
├── requirements.txt
└── README.md
```

### 模块职责速查

| 模块 | 职责 |
|------|------|
| `src/core/models` | `SubtitleEntry` 数据类，所有模块共享的领域模型 |
| `src/core/chunker` | 将字幕列表切分为 API 安全批次（大小上限 + 场景断点） |
| `src/core/text_processor` | 中文检测、智能换行、ASS 标签遮罩/还原 |
| `src/parsers/srt_parser` | SRT 文件解析、序列化、编码检测 |
| `src/parsers/ass_parser` | ASS 文件解析（含标签提取）、ASS 输出重建 |
| `src/translator/engine` | 调用 DeepSeek API → 解析 JSON → 校验 → 重试 → 降级 |
| `src/translator/prompts` | System Prompt / User Prompt 模板构建 |
| `src/translator/cache` | MD5 缓存路径生成、进度持久化（`.srt_cache/`） |
| `src/translator/glossary` | AI 阅读前 N 句自动提取专有名词对照表 |

## 🛠️ 安装指南

1. **克隆仓库**：
   ```bash
   git clone https://github.com/你的用户名/你的仓库名.git
   cd 你的仓库名
   ```

2. **安装依赖**：
   建议使用 Python 3.9+ 环境。
   ```bash
   pip install -r requirements.txt
   ```
   *注：主要依赖包括 `streamlit`, `openai`, `pathlib` 等。*

3. **配置 API Key**：
   在项目根目录下创建 `.streamlit/secrets.toml` 文件（或直接在 Web 界面输入）：
   ```toml
   DEEPSEEK_API_KEY = "你的_deepseek_api_key"
   ```

## 🚀 使用流程

1. **启动应用**：
   ```bash
   streamlit run app.py
   ```
2. **上传文件**：上传需要翻译的 `.srt` 或 `.ass` 格式字幕。
3. **设定背景**：
   - 点击 **"✨ AI 自动提取"** 获取初步术语表。
   - 在"专有名词对照表"中修正关键角色译名。
4. **开始翻译**：点击"🚀 开始翻译"，实时查看进度及剩余时间预测。
5. **下载结果**：翻译完成后，一键下载带 `_zh` 后缀的中文字幕文件。

## ⚙️ 底层技术细节

*   **智能分批算法**：结合了 `max_batch_size`（数量上限）和 `gap_threshold_ms`（时间空隙）。当检测到视频两行字幕之间有较长停顿（如换场）时，会自动进行切割，以保持语义的相对独立。
*   **鲁棒性设计**：
    *   提供 **"原文兜底"** 机制：若 AI 在多次尝试后仍遗漏某条字幕，系统将保留原英文，确保 SRT 时间轴完整不乱。
    *   **缓存机制**：所有已完成的批次实时存入本地 `.srt_cache/` 目录。

## 📩 用户反馈

工具内置了 SMTP 异步反馈系统。如果您在使用过程中遇到问题，可以通过侧边栏的反馈框直接向开发者发送邮件。

---

### 📝 免责声明
本项目仅供学习和研究使用。使用 DeepSeek 等商业 API 产生的费用由用户自行承担。建议在翻译长篇影视前，先用小段落测试翻译效果。

---
**Power by DeepSeek & Streamlit**
