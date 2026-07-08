# 级联式与端到端语音理解：N=50 TTS 与 N=50 真实人声双实验基准

[English](report.md)

**作者：** Jiayi Li（李佳宜）、Liu Luofei（刘洛菲）、Zhang Yuchen（张予辰）  
**日期：** 2026 年 6 月至 8 月  
**课程：** 浙江大学本科暑期研究  
**代码仓库：** `github.com/jiayi0428/speech-benchmark`

## 摘要

本报告比较两种语音理解架构：**级联式**（ASR → 文本 LLM）和**端到端式**（音频直接输入语音大模型）。最终采用两个等规模数据集：**50 条 TTS 合成语音**和**50 条真实人声录音**。四条路径完全对齐：A Oracle、B Whisper 级联、C Qwen 转写级联、D Qwen Direct。

核心结论不是“某一路径永远赢”，而是：**级联路径是当前四个语义任务的默认最优选择，Direct 在语气/表演性强的情感判断中有局部价值。** TTS 中，B 达到 **78% 情感准确率**和 **98% 意图准确率**，D 为 **72%** 和 **80%**。真实人声中，A/B/C 总体仍强于 D，但在 `entertain` 子集上，D 的情感准确率达到 **50%**，高于 B 的 **25%** 和 C 的 **37.5%**。这说明端到端音频模型确实可能捕捉到转写文本丢失的语气、讽刺、节奏和表演线索。

## 1. 方法

所有实验使用同四项任务：摘要（ROUGE-L）、情感分类、关键词提取（F1）和意图分类。四条路径如下：

| 路径 | 信息来源 | 下游推理 |
|---|---|---|
| A Oracle | 人工真值转写 | DeepSeek-chat |
| B Cascade | faster-whisper large-v3 转写 | DeepSeek-chat |
| C Qwen-ASR | Qwen2-Audio 转写 | DeepSeek-chat |
| D Qwen-Direct | Qwen2-Audio 直接音频理解 | 结构化任务再由 DeepSeek 整理格式 |

人声 N=50 来自现有 N=66 人声集合：剔除 16 条非 v5 的 `describe` 样本；v5 全部保留，并用另一台电脑更新后的 `rensheng_results.json` 覆盖原 v5 结果。TTS N=50 使用 `TTS_50_results.json`。

人声意图分布：`{'describe': 15, 'entertain': 8, 'inform': 17, 'persuade': 4, 'question': 6}`。  
TTS 意图分布：`{'describe': 8, 'entertain': 8, 'inform': 19, 'persuade': 9, 'question': 6}`。

## 2. TTS 基准（N=50）

| 路径 | 摘要 ROUGE-L | 情感准确率 | 关键词 F1 | 意图准确率 |
|---|---:|---:|---:|---:|
| A Oracle（人工文本→DeepSeek） | 0.3291 | 72.0% | 0.3079 | 98.0% |
| B Cascade（Whisper→DeepSeek） | 0.3160 | 78.0% | 0.3188 | 98.0% |
| C Qwen转写→DeepSeek | 0.3299 | 68.0% | 0.2978 | 94.0% |
| D Direct（Qwen音频直推） | 0.3251 | 72.0% | 0.3124 | 80.0% |

![TTS N=50 四路径任务得分](figures/final_n50_tts_metrics.png)

TTS 是更受控的条件：语音干净、发音稳定，声学信息很少提供文本之外的额外价值。因此结果很清楚：B/C 很强，D 在摘要上接近，但在意图识别上明显落后。B 与 D 的 18 个百分点意图差距，是 TTS 实验最重要的信号。

| 任务 | D-B | D-C | 解释 |
|---|---:|---:|---|
| 摘要 | 0.0091 | -0.0048 | Direct 落后 |
| 情感 | -0.0600 | 0.0400 | Direct 落后 |
| 关键词 | -0.0064 | 0.0146 | Direct 落后 |
| 意图 | -0.1800 | -0.1400 | Direct 落后 |

![Direct 相对级联路径差值](figures/final_n50_direct_deltas.png)

## 3. 真实人声基准（N=50）

| 路径 | 摘要 ROUGE-L | 情感准确率 | 关键词 F1 | 意图准确率 |
|---|---:|---:|---:|---:|
| A Oracle（人工文本→DeepSeek） | 0.2601 | 74.0% | 0.3914 | 64.0% |
| B Cascade（Whisper→DeepSeek） | 0.2621 | 68.0% | 0.3325 | 64.0% |
| C Qwen转写→DeepSeek | 0.2564 | 72.0% | 0.3421 | 54.0% |
| D Direct（Qwen音频直推） | 0.1854 | 62.0% | 0.2765 | 50.0% |

![人声 N=50 四路径任务得分](figures/final_n50_human_metrics.png)

真实人声中，A/B/C 仍然是更强的通用方案。D 在摘要上下降明显，说明自然噪声、观众声、停顿和录音差异会挑战 Qwen2-Audio 的直接理解能力。但均值表也会掩盖一个重要分组信号：Direct 在语气和表演性真正重要的场景里有价值。

| 任务 | D-B | D-C | 解释 |
|---|---:|---:|---|
| 摘要 | -0.0767 | -0.0710 | Direct 落后 |
| 情感 | -0.0600 | -0.1000 | Direct 落后 |
| 关键词 | -0.0559 | -0.0656 | Direct 落后 |
| 意图 | -0.1400 | -0.0400 | Direct 落后 |

## 4. 按意图分组的发现

Direct 最值得关注的是 `entertain` 情感判断：

| 路径（entertain） | 摘要 | 情感 | 关键词 | 意图 |
|---|---:|---:|---:|---:|
| B_whisper_cascade | 0.2003 | 25.0% | 0.2679 | 62.5% |
| C_qwen_transcript | 0.2062 | 37.5% | 0.3232 | 25.0% |
| D_qwen_direct | 0.1552 | 50.0% | 0.1483 | 25.0% |

![人声按真实意图的情感准确率](figures/final_n50_human_sentiment_by_intent.png)

![TTS 按真实意图的情感准确率](figures/final_n50_tts_sentiment_by_intent.png)

![人声按真实意图的 Direct 信号热力图](figures/final_n50_human_intent_heatmap.png)

Direct 不是最好的通用语音理解路径，但它也不是“没有用”。它赢在最符合直觉的位置：真实娱乐/脱口秀类语音的情感判断。笑点、讽刺、语速、停顿、观众反应和说话方式会携带文本转写无法完整保留的情绪线索。

## 5. TTS 与真实人声的差异

![真实人声相对 TTS 的性能变化](figures/final_n50_cross_benchmark_drop.png)

TTS 会高估真实部署的容易程度。B 的情感准确率从 TTS 的 78% 降到人声的 68%；D 的摘要 ROUGE-L 从 0.325 降到 0.185。这不只是随机波动，而是自然录音条件、环境声、表演性和真实语音不规整性共同造成的难度上升。

## 6. 架构解释：不同任务下的路径差异

实验结果不支持把某一条路径视为所有任务的唯一答案。更准确的解释是：不同路径保留的信息不同，因此任务类型会影响相对表现：

| 条件 | 更强的证据指向 |
|---|---|
| 清晰 TTS 或事实性语音 | B Whisper 级联作为默认 |
| 摘要 | 优先 B/C；D 可作为补充观察 |
| 关键词 | 优先 B/C，不建议 D 作为主路径 |
| 意图 | B 最稳，尤其在 TTS 中 |
| 娱乐、脱口秀、讽刺类情感判断 | 增加 D 作为情感辅助路径 |
| 成本/隐私优先的探索性场景 | 可用 D 做轻量首轮，但必须提示准确率代价 |

因此，本项目的研究启示是：内容语义任务更适合级联路径；当语气、情绪、讽刺、节奏和现场反应会影响标签时，Direct 的音频线索具有独立价值。

## 7. 局限性

N=50 已经比早期 N=8/N=12 稳定很多，但仍是项目规模基准。每个架构族只测试了一组模型：Whisper large-v3、Qwen2-Audio-7B 和 DeepSeek-chat。人工标注没有正式计算多标注者一致性。按 intent 分组后的样本不均衡：人声 `entertain` 只有 8 条，`persuade` 只有 4 条，因此子组结论应作为明确趋势和后续验证假设，而不是无限泛化的定律。

## 8. 结论

当前四个语义任务下，**级联路径仍然是默认更可靠的语音理解架构**。它在意图和关键词上尤其稳，在 TTS 中优势更明显。Direct 的价值更窄，但真实存在：它能在娱乐/讽刺/表演性强的人声情感判断中利用语气和声学线索。最好的系统不是单一路径，而是一个路由器：内容任务走级联，强语气情感任务引入 Direct。

## 输出文件

- 人声 N=50 汇总：`data/results/human_speech_final_n50/summary.json`
- 人声 N=50 明细：`data/results/human_speech_final_n50/scores.csv`
- TTS N=50 汇总：`data/results/tts_speech_final_n50/summary.json`
- TTS N=50 明细：`data/results/tts_speech_final_n50/scores.csv`
