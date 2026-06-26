# 设计文档：级联 vs 端到端语音理解基准评测

**日期:** 2026-06-24
**状态:** 已批准
**作者:** User + Claude Code

---

## 1. 项目概览

### 1.1 项目名称
**级联 vs 端到端：语音理解架构的鲁棒性感知基准评测**

### 1.2 核心研究问题
"搭积木"式的级联架构（ASR → 文本 LLM）能否匹敌端到端语音 LLM？各自在什么条件下占优？我们的核心假设是：**语音 LLM 的真正价值不在于干净音频上的原始准确率，而在于真实声学退化场景下的鲁棒性** — 我们通过实验来验证这一点。

### 1.3 交付物
| 交付物 | 说明 |
|--------|------|
| **Jupyter Notebook** | 完整实验管线：数据加载 → 两条推理路径 → 评估 → 可视化 |
| **Gradio 演示** | 轻量 Web 界面，实时并排对比两种方案 |
| **英文书面报告** | 6-8 页学术风格报告，含方法论、结果和分析 |
| **深度案例研究** | 3-5 个"解剖式"音频样本，展示每种方案的失败模式 |

### 1.4 范围与时间线
- **周期:** 8-10 周（本科暑期科研）
- **难度:** 中级 — 涉及 API 集成、本地模型推理、评估设计与可视化
- **硬件:** NVIDIA RTX 5070 8GB VRAM

---

## 2. 架构设计

### 2.1 三条流水线

```
方案 A: 级联（ASR → 文本 LLM）—— "搭积木"
  音频 ──→ faster-whisper large-v3 ──→ 转录文本 ──→ GPT-4o-mini ──→ 四个任务输出
            (本地 GPU, 免费)                        (API, 便宜)

方案 B: 端到端（语音 LLM）—— "一步到位"
  音频 ────────────────────────────→ GPT-4o Audio 模式 ──────────→ 四个任务输出
            (无中间文本)                  (API, 原生音频)

方案 C:（可选——轻量参考）
  音频 ──→ Qwen2-Audio-7B (INT4, 本地 GPU)
            (开源语音 LLM，仅干净场景对比)
```

### 2.2 四个评估任务

| 任务 | Prompt 模板要点 | 评估指标 |
|------|----------------|----------|
| **摘要** | "用 3-5 句话总结以下内容..." | ROUGE-L, BERTScore, 人工评分(1-5) |
| **情感分析** | "将情感分类为正面/负面/中性，给出置信度..." | 准确率, F1, 混淆矩阵 |
| **关键词提取** | "提取 5-10 个最重要的关键词/短语..." | 精确率/召回率/F1 vs 参考答案 |
| **意图识别** | "说话人的主要意图是什么？(告知/说服/娱乐/提问/描述)..." | 准确率, 置信度校准 |

### 2.3 鲁棒性测试层

| 退化类型 | 参数 | 模拟场景 |
|----------|------|----------|
| **多人嘈杂噪声** | SNR = 20, 10, 0 dB | 拥挤环境 |
| **白噪声** | SNR = 20, 10, 0 dB | 基线声学退化 |
| **混响** | RT60 = 0.5s, 1.0s, 1.5s | 大房间声学 |

每段音频通过所有退化级别。测量两种架构的**性能退化曲线**。

### 2.4 ASR 错误传播分析（级联专有）

- 计算 ASR 输出与标准转录之间的 WER
- 散点图：WER（x 轴）vs 下游任务得分（y 轴）
- 量化**错误放大系数** — WER 每增加 1%，摘要质量下降多少？

---

## 3. 数据

### 3.1 主数据集：TED-LIUM v3
- **规模:** ~452 小时 TED 演讲，含转录文本
- **优势:** 语音干净、话题多样、结构清晰适合摘要
- **选取:** 50-100 个片段（1-3 分钟/段），覆盖不同说话人和话题

### 3.2 辅助数据集：AMI Corpus（会议子集）
- **规模:** ~100 小时真实会议，含转录和标注
- **优势:** 自然对话，有重叠、犹豫、非正式语言
- **选取:** 20-30 个会议片段（2-5 分钟/段）

### 3.3 数据划分
| 划分 | TED-LIUM | AMI | 用途 |
|------|----------|-----|------|
| 开发集 | 10 段 | 5 段 | Prompt 调试、管线验证 |
| 测试-干净 | 30 段 | — | 干净音频基准评测 |
| 测试-会议 | — | 15 段 | 自然语音基准评测 |
| 测试-鲁棒性 | 10 段 × 9 条件 | — | 噪声退化实验 |

---

## 4. 组件设计

### 4.1 项目结构
```
speech-benchmark/
├── notebooks/
│   ├── 01_data_preparation.ipynb      # 数据集加载、预处理、划分
│   ├── 02_cascade_pipeline.ipynb      # faster-whisper → GPT-4o-mini 推理
│   ├── 03_direct_pipeline.ipynb       # GPT-4o Audio 模式推理
│   ├── 04_evaluation.ipynb            # 指标计算、统计检验
│   ├── 05_visualization.ipynb         # 雷达图、退化曲线、案例研究
│   └── 06_deep_case_studies.ipynb     # 失败案例集、深度剖析
├── src/
│   ├── __init__.py
│   ├── data.py                        # 数据加载、音频处理、噪声注入
│   ├── cascade.py                     # faster-whisper + GPT-4o-mini 管线
│   ├── direct.py                      # GPT-4o Audio 模式管线
│   ├── evaluation.py                  # 各任务指标、统计检验
│   ├── visualization.py               # 可视化函数
│   └── config.py                      # API 密钥、路径、常量
├── app/
│   └── gradio_app.py                  # 交互演示
├── data/
│   ├── raw/                           # 原始数据集
│   ├── processed/                     # 处理后的片段和元数据
│   └── results/                       # 推理输出、评估分数
├── requirements.txt
├── README.md
└── report/
    └── report.md                      # 书面报告源文件
```

### 4.2 关键依赖
```
# 本地推理
faster-whisper>=1.0.0
torch>=2.4.0
transformers>=4.45.0
bitsandbytes>=0.43.0        # INT4 量化（可选）

# API
openai>=1.50.0

# 音频处理
librosa>=0.10.0
soundfile>=0.12.0
pydub>=0.25.0
audiomentations>=0.35.0     # 噪声注入

# 评估
rouge-score>=0.1.2
bert-score>=0.3.13
scipy>=1.14.0
scikit-learn>=1.5.0

# 可视化
matplotlib>=3.9.0
plotly>=5.23.0
seaborn>=0.13.0
ipywidgets>=8.1.0

# 演示
gradio>=4.40.0

# 工具
pandas>=2.2.0
numpy>=1.26.0
tqdm>=4.66.0
python-dotenv>=1.0.0
```

### 4.3 配置文件骨架 (`src/config.py`)

```python
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"

# ASR 模型
WHISPER_MODEL = "large-v3"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "float16"  # 显存紧张时用 "int8_float16"

# 文本 LLM（级联）
TEXT_LLM_MODEL = "gpt-4o-mini"

# 语音 LLM（端到端）
SPEECH_LLM_MODEL = "gpt-4o-audio-preview"

# 开源语音 LLM（可选）
QWEN2_AUDIO_MODEL = "Qwen/Qwen2-Audio-7B-Instruct"

# 音频参数
SAMPLE_RATE = 16000
MAX_AUDIO_SECONDS = 300

# 噪声条件
NOISE_CONDITIONS = {
    "clean": None,
    "babble_20db": {"type": "babble", "snr": 20},
    "babble_10db": {"type": "babble", "snr": 10},
    "babble_0db":  {"type": "babble", "snr": 0},
    "white_20db":  {"type": "white",  "snr": 20},
    "white_10db":  {"type": "white",  "snr": 10},
    "white_0db":   {"type": "white",  "snr": 0},
    "reverb_0.5s": {"type": "reverb", "rt60": 0.5},
    "reverb_1.0s": {"type": "reverb", "rt60": 1.0},
    "reverb_1.5s": {"type": "reverb", "rt60": 1.5},
}

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
```

---

## 5. 数据流

### 5.1 实验执行流程

```
  加载数据集 ──→ 噪声注入（1段→10变体）──┬──→ 级联管线 ──┐
                                        ├──→ 端到端管线 ──┤
                                        └──→ Qwen2-Audio ──┘
                                                    │
                                                    ▼
                                          汇总所有输出结果
                                                    │
                                                    ▼
                                          评估层（4任务 + 鲁棒性）
                                                    │
                                                    ▼
                                          统计分析（t检验 + bootstrap + 相关性）
                                                    │
                                                    ▼
                                          可视化（雷达图、退化曲线、波形对齐）
```

### 5.2 API 调用设计

**级联（文本 LLM 调用）:**
```python
def cascade_inference(transcript: str, task: str) -> dict:
    prompts = {
        "summarization": "Summarize the following transcript in 3-5 sentences...",
        "sentiment": "Classify sentiment as positive/negative/neutral...",
        "keywords": "Extract 5-10 key phrases from this transcript...",
        "intent": "Identify primary intent: inform/persuade/entertain/question/describe...",
    }
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompts[task]},
            {"role": "user", "content": transcript},
        ],
        temperature=0.0,
    )
    return {"task": task, "output": response.choices[0].message.content}
```

**端到端（语音 LLM 调用）:**
```python
def direct_inference(audio_base64: str, task: str) -> dict:
    prompts = { ... }
    response = openai.chat.completions.create(
        model="gpt-4o-audio-preview",
        messages=[
            {"role": "system", "content": prompts[task]},
            {"role": "user", "content": [
                {"type": "input_audio", "input_audio": audio_base64},
            ]},
        ],
        temperature=0.0,
    )
    return {"task": task, "output": response.choices[0].message.content}
```

---

## 6. 评估设计

### 6.1 各任务指标

| 任务 | 自动指标 | 人工评估 |
|------|---------|----------|
| 摘要 | ROUGE-1/2/L, BERTScore (F1) | 1-5 分：完整性、简洁性、事实准确性 |
| 情感分析 | 准确率, Macro-F1, 混淆矩阵 | 与参考答案一致性 |
| 关键词提取 | Precision@k, Recall@k, F1@k (k=5,10) | 每个关键词相关性评分(1-3) |
| 意图识别 | 准确率, 置信度校准 (ECE) | 与参考答案一致性 |

### 6.2 统计检验
- **配对 t 检验**（非正态时用 Wilcoxon）：每个指标上级联 vs 端到端
- **Bootstrap 95% 置信区间**：所有均值差异
- **Cohen's d**：效应量
- **Pearson r**：WER → 下游任务得分相关性（误差传播）

### 6.3 鲁棒性评估
- **退化曲线:** 指标得分(y) vs SNR(x) — 两条线（级联/端到端）叠在同一图上
- **鲁棒性指数:** 退化曲线下面积（越高越鲁棒）
- **拐点分析:** 每种方案在哪个 SNR 跌出"可接受"阈值？

---

## 7. 可视化（"出彩"层）

### 7.1 核心图表
1. **雷达图** — 4 任务 × 2 架构，蜘蛛网叠层
2. **退化曲线** — SNR 为 x 轴，指标为 y 轴，级联 vs 端到端
3. **错误传播散点图** — WER vs 下游得分，附回归线和 r 值
4. **混淆矩阵** — 级联 vs 端到端情感分析并排对比
5. **柱状图** — 成本/延迟对比

### 7.2 音频对齐可视化
- **波形 + 转录对齐:** `librosa.display.waveshow()` + 下方标注转录，ASR 错误红色高亮
- **逐句对比:** 对一段 2 分钟 TED 片段，逐句展示：标准转录 | ASR 输出 | 级联摘要 | 端到端摘要 — 四行垂直对齐

### 7.3 交互式（Plotly）
- 所有主图表同时渲染 Plotly 版本（可缩放、悬停显示数值）
- 雷达图悬停显示精确值 + 95% CI

---

## 8. Gradio 演示界面设计

```
┌──────────────────────────────────────────────────────────┐
│  🎤 语音理解架构对比                                       │
│                                                          │
│  ┌─────────────────────┐  ┌─────────────────────┐        │
│  │ 📁 上传音频           │  │ 🎛️ 噪声级别           │        │
│  │ [拖拽或点击上传]       │  │ ○ 干净  ○ 20dB     │        │
│  │ [🔴 实时录音]         │  │ ○ 10dB   ○ 0dB      │        │
│  └─────────────────────┘  └─────────────────────┘        │
│                                                          │
│  [▶ 开始对比]                                             │
│                                                          │
│  ┌──────────────────────┐ ┌──────────────────────┐       │
│  │ 级联 (ASR+LLM)        │ │ 端到端 (GPT-4o Audio)  │       │
│  │ 📝 转录: ...          │ │ 🎯 摘要: ...           │       │
│  │ 📊 摘要: ...          │ │ 😊 情感: ...           │       │
│  │ 😊 情感: ...          │ │ 🔑 关键词: ...         │       │
│  │ 🔑 关键词: ...        │ │ 🎯 意图: ...           │       │
│  │ 🎯 意图: ...          │ │                        │       │
│  │ ⏱️ 2.3s 💰 $0.004    │ │ ⏱️ 3.1s 💰 $0.018    │       │
│  └──────────────────────┘ └──────────────────────┘       │
│                                                          │
│  📈 性能对比 [雷达图] [退化曲线]                             │
└──────────────────────────────────────────────────────────┘
```

核心交互：左右两栏并排对比是 UX 的核心。噪声开关可实时测试鲁棒性。底部图表引用预计算的基准评测结果。

---

## 9. 风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| GPT-4o Audio API 不可用或太贵 | 中 | 高 | 预留预录制结果作为备选；API 预算上限 $30 |
| faster-whisper 在 RTX 5070 上显存溢出 | 低 | 中 | 使用 `int8_float16` 或降级到 `medium` 模型 |
| Qwen2-Audio 8GB 显存放不下 | 高 | 低 | 这是可选项——量化失败就跳过 |
| 参考答案标注不足（情感/关键词/意图） | 中 | 中 | 用 GPT-4（文本）作为 "oracle" 标注器补标 |
| OpenAI API 速率限制 | 低 | 低 | 指数退避重试 + 请求间隔 1s |

---

## 10. 成功标准

- [ ] 两条流水线端到端跑通：级联 + 端到端
- [ ] 4 个任务在 TED-LIUM 和 AMI 上完成评估
- [ ] 3 种噪声 × 3 级别 = 9 条退化曲线
- [ ] 统计检验显著（或能解释为什么不显著）
- [ ] 失败案例集中至少含 3 个深度剖析案例
- [ ] Gradio 演示：上传任意音频 → 看到并排对比
- [ ] 书面报告：6-8 页英文，包含全部图表
- [ ] 全部代码、prompt 和结果可复现（种子固定 + temperature=0）

---

## 11. 实现阶段预览

| 阶段 | 周次 | 重点 |
|------|------|------|
| 1. 脚手架 | 1-2 | 项目搭建、数据下载、环境配置 |
| 2. 级联管线 | 2-3 | faster-whisper 集成、GPT-4o-mini prompt 工程 |
| 3. 端到端管线 | 3-4 | GPT-4o Audio 模式集成、prompt 适配 |
| 4. 评估 | 4-5 | 指标实现、统计检验 |
| 5. 鲁棒性 | 5-6 | 噪声注入、退化实验 |
| 6. 可视化 | 6-7 | 全部图表、音频对齐视图、交互式图表 |
| 7. 案例研究 | 7-8 | 失败案例集、深度剖析 |
| 8. Gradio 演示 | 8-9 | Web 界面实现 |
| 9. 报告 | 9-10 | 书面报告、打磨、演练 |

---

*设计文档完。本文档将作为实现计划的输入。*
