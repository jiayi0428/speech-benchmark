# 语音理解基准

[English](README.md)

**级联路径（ASR + 文本 LLM）与端到端路径（语音 LLM）的初步比较**

[![Tests](https://img.shields.io/badge/tests-31%20passed-brightgreen)](tests/)
[![Python](https://img.shields.io/badge/python-3.14-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Repo](https://img.shields.io/badge/repo-github.com/jiayi0428/speech--benchmark-lightgrey)](https://github.com/jiayi0428/speech-benchmark)

> *去掉转写瓶颈，能否提升语音理解能力？*

这是 Jiayi Li、Liu Luofei（刘洛菲）和 Zhang Yuchen（张予辰）于 2026 年
开展的本科暑期研究项目。项目在四项语音理解任务上比较多条路径，并使用
人工真值评估和噪声鲁棒性测试。

---

## 架构

```text
 音频
     级联路径（“积木式”）：
   [faster-whisper large-v3] → 转写文本 → [DeepSeek-chat API] → 输出
   本地 GPU + API；约 $0.0005/任务；约 16 秒

     Direct 路径（“端到端”）：
   [Qwen2-Audio-7B INT4] → 输出
   本地 GPU 推理；结构化任务使用文本 API 后处理
   原始同环境运行：约 256 秒
```

---

## 主要发现（初步，N=8）

| 维度 | Cascade | Direct | 结论 |
|---|---|---|---|
| **速度（原始同环境运行）** | 约 16 秒 | 约 256 秒 | Cascade 快约 16 倍 |
| **语音模型 API 成本** | 约 $0.0005/任务 | 本地推理 | Direct |
| **原始结构化输出** | **100% 合法 JSON** | 30% | Cascade |
| **噪声鲁棒性（0 dB）** | 相对自身基线变化更平坦 | 摘要均值更高 | 无确定赢家 |
| **情绪/韵律信息** | 转写时丢失 | 理论上可利用 | 尚未直接验证 |

> **这是基于 8 个 TTS 样本的先导实验。** 研究边界见[局限性](report/report.zh-CN.md#63-局限性)。

---

## 一键复现

```bash
git clone <your-repo-url> speech-benchmark
cd speech-benchmark
pip install -r requirements.txt
cp .env.example .env  # 填入 DEEPSEEK_API_KEY
python run_all.py     # 数据 → Cascade → Direct → 噪声 → 评估 → 图表
```

输出包括：`data/results/` 中的实验结果、`report/figures/` 中的图表，以及 `report/report.md` 和 `report/report.zh-CN.md`。

---

## 项目结构

```text
speech-benchmark/
 run_all.py                   一键复现入口
 src/                         核心流程（6 个模块）
   cascade.py               # faster-whisper + DeepSeek-chat
   direct_qwen.py           # Qwen2-Audio-7B（本地 INT4）
   data.py                  # 音频读写、加噪、数据集准备
   evaluation.py            # WER、ROUGE-L、F1、准确率、t 检验、bootstrap
   visualization.py         # 雷达图、退化曲线、误差传播图
   config.py                # 自动识别 DeepSeek/OpenAI/Gemini/Qwen
 notebooks/                   交互式分析（5 个 notebook）
 app/gradio_app.py            两路径实时并排比较演示
 report/                      论文式报告与图表
   report.md                # 英文完整报告
   report.zh-CN.md          # 中文完整报告
   figures/                 # 雷达图、延迟、成本、退化曲线
 data/
   ground_truth.json        # 人工标注的参考标签
   results/                 # 实验输出
 tests/                       # 自动化测试
```

---

## 任务与指标

| 任务 | 指标 | 说明 |
|---|---|---|
| **摘要生成** | ROUGE-L | 生成摘要与人工参考摘要的内容重合度 |
| **情感分析** | 准确率 | 与 positive / negative / neutral 真值比较 |
| **关键词提取** | Precision、Recall、F1 | 与人工关键词短语的重合度 |
| **意图识别** | 准确率 | inform / persuade / entertain / question / describe |

---

## 鲁棒性测试

| 噪声类型 | 强度 | 模拟场景 |
|---|---|---|
| 白噪声 | 10 dB、0 dB SNR | 基础声学退化 |
| 多人说话噪声 | 框架已准备 | 拥挤环境 |
| 混响 | 框架已准备 | 大空间声学环境 |

---

## 真实录音先导实验

项目进一步在 8 段包含不可控环境声和观众声的真实人声录音上评估 Cascade 与 Direct。

| 任务 | Cascade | Direct |
|---|---:|---:|
| 摘要 ROUGE-L | 0.2807 | 0.2388 |
| 情感准确率 | 75.0% | 62.5% |
| 关键词精确短语 F1 | 0.4428 | 0.4167 |
| 意图准确率 | 62.5% | 25.0% |

Cascade 在四项指标上的均值都更高，但 N=8 下所有配对 bootstrap 区间都跨过 0。这不是受控噪声实验，而且两条路径来自不同执行环境，因此不能横向解释延迟。

工作流、配对结果、API 用量和局限性见：

- [`experiments/HUMAN_SPEECH_V1.md`](experiments/HUMAN_SPEECH_V1.md)
- [`report/report.zh-CN.md`](report/report.zh-CN.md)

B/C/D 消融发现，Qwen 转写的规范化 WER 高于 Whisper（0.0696 对
0.0338），但两条“转写 → DeepSeek”路径的情感和意图准确率完全相同。
在这次 N=8 先导实验中，Qwen 转写后交给 DeepSeek 的 C 路径在摘要和
意图上也高于 Qwen 直接理解的 D 路径。详见
[`experiments/HUMAN_SPEECH_ABLATION_C.md`](experiments/HUMAN_SPEECH_ABLATION_C.md)。

---

## 原始 TTS 的 Qwen 转写消融

原先 8 条干净 TTS 音频已经通过“Qwen 转写 -> DeepSeek 四项任务”路径运行。
8 条转写和 32 次任务调用全部成功，清洗后的规范化 WER 为 0.0102。C 路径
在摘要、情感、关键词和意图上的结果为 0.3815、75.0%、0.3694 和 87.5%；
D 路径对应为 0.4600、62.5%、0.3378 和 87.5%。N=8 下所有配对区间都跨过
0，只能说明初步趋势。由于现有 B 文件缺少结构化任务结果和 Whisper 转写，
B/C/D 只能比较摘要，不能比较完整四任务或 Whisper/Qwen WER。详见
[`experiments/TTS_QWEN_TRANSCRIPT_V1.md`](experiments/TTS_QWEN_TRANSCRIPT_V1.md)。

---

## 新增 TTS12 四路径实验

新增 12 条干净 TTS 音频使用同一套真值，对比 A（Oracle 真值转写）、
B（Whisper 级联）、C（Qwen 转写）和 D（Qwen 直接理解）。

| 路径 | 摘要 | 情感 | 关键词 F1 | 意图 |
|---|---:|---:|---:|---:|
| A：Oracle | 0.3528 | 92% | 0.4500 | 100% |
| B：Whisper | 0.3448 | 92% | 0.4500 | 100% |
| C：Qwen 转写 | 0.3479 | 83.3% | 0.4298 | 100% |
| D：Qwen 直接理解 | 0.3324 | 41.7% | 0.4286 | 16.7% |

A/B/C 的结果接近；D 的摘要和关键词仍接近前三条路径，但出现明显的意图
分类失败。N=12 仍只能作描述性分析。详见
[`experiments/TTS12_CD_V1.md`](experiments/TTS12_CD_V1.md)。

独立 D 复现逐字、逐分重现了全部 48 个结果。三次新 C/D 摘要重复中，
C 每次都是 7 胜、D 每次都是 5 胜，而且每条样本的胜负方向都保持不变；
但所有区间仍跨过 0。详见
[`experiments/TTS12_REPEAT_STABILITY_V1.md`](experiments/TTS12_REPEAT_STABILITY_V1.md)。

---

## 原始 TTS 实验结果摘要

| 指标 | Cascade | Direct |
|---|---:|---:|
| 情感准确率 | 88% | 38% |
| 意图准确率 | 88% | 62% |
| 关键词 F1 | 0.36 | 0.29 |
| 摘要 ROUGE-L | 0.402 | 0.448 |

完整指标见 `data/results/final_summary.json`。

---

## 快速启动演示

```bash
cd app && python gradio_app.py
# 打开 http://127.0.0.1:7860，上传音频并实时比较两条路径
```

---

## 局限性

1. **每个数据集 N=8 或 N=12：** 这些是先导研究，不是大规模评估。
2. **主基准使用 TTS：** 补充真实人声实验仍只有 8 个样本，并带有不可控录音条件。
3. **每种范式只测试一个模型组合：** Qwen2-Audio-7B INT4 与 Whisper + DeepSeek-chat。
4. **没有人工质量评分：** 使用人工真值上的自动指标，没有人工评价连贯性、可读性或事实性。
5. **受控噪声类型单一：** 已运行白噪声；多人说话噪声和混响仅有框架，尚未正式执行。

完整局限性与未来工作见[中文研究报告](report/report.zh-CN.md#63-局限性)。

真实人声配对结果已合并进[中文主报告](report/report.zh-CN.md)。

---

## 作者

**Jiayi Li（李佳宜）· Liu Luofei（刘洛菲）· Zhang Yuchen（张予辰）**
2026 年本科暑期研究

项目使用 Python 3.14、faster-whisper、DeepSeek API、Qwen2-Audio-7B、PyTorch、Gradio 和 Matplotlib 构建。
