# Human Speech V1：Cascade 与 Direct 配对报告

[English](human_speech_v1_report.md)

**日期：** 2026-07-02  
**作者：** Jiayi Li（李佳宜）、Liu Luofei（刘洛菲）、Zhang Yuchen（张予辰）
**范围：** 8 个成对的真实英文人声样本，4 项任务  
**性质：** 初步描述性先导实验

## 执行摘要

Cascade 在四项指标上的平均分都高于 Direct。观察到的最大差异来自意图识别：Cascade 为 62.5%，Direct 为 25.0%。在摘要任务中，Cascade 也赢得了 8 个逐样本比较中的 6 个。

这些结果没有统计确定性。Direct 减 Cascade 的所有配对 bootstrap 95% 区间都跨过 0，而且 N=8 太小，不能宣称统计显著。录音包含不可控的环境声和观众声，也没有匹配的干净版本，因此不能单独估计噪声影响。

## 方法

- 样本：`altitude`、`calm`、`depression`、`happiness`、`healthy`、`menopause`、`speak`、`thanks`
- 人工真值：逐字稿、摘要、情感、关键词和意图
- Cascade：外部提供的 faster-whisper + 文本 LLM 结果，包含转写
- Direct：本地 Qwen2-Audio-7B INT4
- Direct 结构化任务：使用与现有白噪声实验相同的 DeepSeek 后处理
- 指标：ROUGE-L、分类准确率、精确短语关键词 F1、WER

外部 Cascade 文件与 8 个样本名称和内容匹配，但没有音频哈希，因此无法独立验证它与 Direct 输入是否逐字节相同。

## 汇总结果

![真实人声路径比较](figures/human_speech_path_comparison.png)

| 任务 | 指标 | Cascade | Direct | Direct - Cascade |
|---|---|---:|---:|---:|
| 摘要 | ROUGE-L | 0.2807 | 0.2388 | -0.0419 |
| 情感 | 准确率 | 75.0%（6/8） | 62.5%（5/8） | -12.5 pp |
| 关键词 | 精确短语 F1 | 0.4428 | 0.4167 | -0.0261 |
| 意图 | 准确率 | 62.5%（5/8） | 25.0%（2/8） | -37.5 pp |

## 配对结果

| 任务 | Cascade 胜 | Direct 胜 | 平局 | Direct - Cascade 的 95% bootstrap 区间 |
|---|---:|---:|---:|---:|
| 摘要 | 6 | 2 | 0 | [-0.0882, 0.0211] |
| 情感 | 2 | 1 | 5 | [-0.5000, 0.2500] |
| 关键词 | 4 | 2 | 2 | [-0.1565, 0.1038] |
| 意图 | 4 | 1 | 3 | [-0.8750, 0.1250] |

所有区间都跨过 0，因此表格只能支持初步趋势，不能证明架构差异。

## 逐样本结果

`C/D` 表示先列 Cascade、再列 Direct。分类任务中 1 表示正确，0 表示错误。

| 样本 | 摘要 C/D | 情感 C/D | 关键词 F1 C/D | 意图 C/D | Cascade WER |
|---|---:|---:|---:|---:|---:|
| altitude | 0.2857 / 0.1875 | 1 / 0 | 0.0000 / 0.0000 | 1 / 0 | 0.0455 |
| calm | 0.5424 / 0.4615 | 1 / 1 | 0.8571 / 0.8571 | 1 / 0 | 0.0889 |
| depression | 0.1860 / 0.3333 | 0 / 0 | 0.3750 / 0.7143 | 0 / 1 | 0.0000 |
| happiness | 0.1944 / 0.1579 | 1 / 1 | 0.2500 / 0.3077 | 0 / 0 | 0.1600 |
| healthy | 0.1538 / 0.1639 | 0 / 1 | 0.3750 / 0.1818 | 1 / 0 | 0.0645 |
| menopause | 0.2308 / 0.1277 | 1 / 1 | 0.5714 / 0.5455 | 1 / 1 | 0.1522 |
| speak | 0.3137 / 0.2373 | 1 / 1 | 0.3636 / 0.0000 | 1 / 0 | 0.0727 |
| thanks | 0.3390 / 0.2414 | 1 / 0 | 0.7500 / 0.7273 | 0 / 0 | 0.0667 |

Cascade 平均 WER 为 0.0813。项目使用简单空格分词，因此标点和 token 切分会影响该数值。

## 结构化输出与 API 用量

- Cascade 经文本 LLM 后的严格 JSON：24/24
- Qwen 原始严格 JSON：0/24
- Direct 经 DeepSeek 后处理后的严格 JSON：24/24
- Direct 成功后处理调用：24 次
- API 用量：1,832 个输入 tokens、478 个输出 tokens，共 2,310
- 按项目历史估算约为 0.012 美元；这不是实际账单

## 延迟

Cascade 的记录任务均值为 16.239 至 18.290 秒，Direct 为 1.462 至 4.481 秒。不能把这些数字解释为架构速度差异，因为：

- 两条路径在不同机器上运行；
- 计时边界无法确认完全一致；
- Direct 计时不包含模型加载。

## 结果解释

在这 8 个具体真实人声样本上，Cascade 呈现更高的平均质量，意图识别差异尤其明显。Direct 仍在个别样本上胜出，例如 `depression` 的摘要、关键词和意图。

精确短语指标可能掩盖语义质量。例如 Cascade 在 `altitude` 上给出的关键词与内容相关，但由于措辞没有与人工关键词完全一致，F1 仍为 0。ROUGE-L 同样主要奖励词汇重合，不直接衡量事实正确性和可读性。

因此，不能把路径差异归因于环境噪声，不能宣称统计显著，也不能把当前延迟顺序推广到两种架构。

## B/C/D 消融实验

新增路径 C 使用 Qwen2-Audio 做逐字转写，再采用与 Whisper Cascade 相同的 DeepSeek 任务提示词。

| 路径 | 摘要 ROUGE-L | 情感 | 关键词 F1 | 意图 |
|---|---:|---:|---:|---:|
| B：Whisper 转写 | 0.2807 | 75.0% | 0.4428 | 62.5% |
| C：Qwen 转写 | 0.3064 | 75.0% | 0.3708 | 62.5% |
| D：Qwen 直接理解 | 0.2388 | 62.5% | 0.4167 | 25.0% |

规范化 WER 中，Whisper 为 0.0338，Qwen 为 0.0696。尽管 Qwen 的 WER 更高，B 与 C 的情感和意图正确性完全相同；C 的摘要和意图也高于 D。这说明 WER 和是否存在中间转写，都不能单独解释系统差异。

C 与 D 仍是近似比较，因为 C 由 DeepSeek 完成语义任务，而 D 由 Qwen 完成语义任务。Direct 减 C 的摘要 bootstrap 区间为 [-0.1506, -0.0010]，刚好没有跨 0，但 N=8 和多重比较要求进一步复现。

正式 C 结果使用 32 个唯一调用和 4,615 tokens。由于超时进程继续运行并与人工续跑重叠，完整审计包含 52 次调用和 7,541 tokens。按项目历史口径，全部调用估算约 0.026 美元。

## Git 复现文件

- `experiments/human_speech_v1.json`
- `experiments/HUMAN_SPEECH_V1.md`
- `data/ground_truth_human_v1.json`
- `data/processed/human_speech_v1/audio_manifest.json`
- `data/processed/human_speech_v1/as_recorded/*.wav`
- `data/results/human_speech_v1/cascade_raw.json`
- `data/results/human_speech_v1/direct_raw.jsonl`
- `data/results/human_speech_v1/direct_postprocessed.jsonl`
- `data/results/human_speech_v1/path_comparison_scores.csv`
- `data/results/human_speech_v1/path_comparison_summary.json`
- `data/results/human_speech_v1/qwen_transcription_raw.jsonl`
- `data/results/human_speech_v1/qwen_transcript_cascade_raw.jsonl`
- `data/results/human_speech_v1/bcd_ablation_summary.json`
- `compare_human_paths.py`
- `compare_human_bcd_ablation.py`
