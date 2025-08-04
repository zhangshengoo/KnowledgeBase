### **<font style="color:rgb(26, 28, 30);">背景：</font>**
KimiAudio是语音大模型，能够实现语音识别任务，但预训练模型并没有针对热词能力进行指令训练，导致该指令遵循能力有待挖掘，因此想要通过在Prompt 加入热词作为上下文信息，来提升热词识别准确性。





### 数据：
#### 数据语料
**phase1: 1000 hours LibriSpeech**

**phase2: 1w+ hours, 多样性**

****

**正样本构建**

+ **核心语料**：包含目标热词的真实用户交互数据（10K+样本）
+ **TTS合成数据**：基于多说话人TTS生成带热词的语音（50K+样本） [arXiv](https://arxiv.org/html/2407.18879v1)[arxiv](https://arxiv.org/html/2407.18879v1)

![](https://cdn.nlark.com/yuque/0/2025/png/50590451/1753886712390-6b0fd0b8-5e29-4268-8fe1-72e31b320917.png)

+ **数据增强**：语速变化(0.8x-1.2x)、音调变换(±2半音)加噪

**负样本策略**

+ **Silence类**：背景噪声和静音片段（20%） [arXiv](https://ar5iv.labs.arxiv.org/html/1804.03209)
+ **Unknown类**：不包含任何热词的日常对话（60%）
+ **Confusing words**：声学相似的混淆词（20%） [arXiv](https://arxiv.org/abs/2011.01460)
+ **正负比例**：1:8（参考CTC论文经验） [arXiv +2](https://arxiv.org/html/2411.06437v1)

| | Total hours | spk num | language |
| --- | --- | --- | --- |
| AIShell1 | | | |
| AIShell2 | | | |
| WenetSpeech | | | |
| LibriSpeech | | | |




#### 标签格式
```json
{
  "audio_path": "path/to/audio.wav",
  "transcription": "你好请帮我播放周杰伦的青花瓷",
  "hotwords": ["周杰伦", "青花瓷", "播放"],
  "hotword_candidates": ["周杰伦", "青花瓷", "林俊杰", "王力宏", "稻香", "播放", "暂停", ...],  // 包含干扰词
  "domain": "music"  // 可选：领域标签
}
```



#### 数据训练策略
动态生成热词列表：

+ 每个样本有`P_keep=0.5`的概率使用热词
+ 从当前batch的transcription中随机抽取1-5个n-gram作为正样本热词
+ 添加干扰词（从其他样本或预定义词表中随机选择）
+ 20%的样本不包含任何热词（训练模型的鲁棒性）



### 模型推理设计
#### 1 热词Prompt模板
```plain
# 基础模板
"请将语音转换为文字。"

# 含热词模板
"请将语音转换为文字。可能包含以下关键词：{hotwords}。"

# 领域特定模板（可选）
"请将语音转换为文字。这是一段关于{domain}的对话，可能包含：{hotwords}。"
```





#### 2 动态热词数量设计
+ **推理时热词数量**：0/50/100/500/1000/2000个（根据实际场景）
+ **无热词处理**：30%的推理请求不包含热词，测试基础识别能力
+ **热词分级**：根据置信度或频率对热词排序





### 3. 模型训练
#### 1 损失函数设计
python

```python
# 总损失 = ASR损失 + 热词召回损失
L_total = L_asr + λ * L_hotword_recall

# ASR损失：标准的交叉熵损失
L_asr = CrossEntropy(predicted_text, ground_truth)

# 热词召回损失：鼓励模型正确识别热词
L_hotword_recall = Σ -log(P(hotword_i | audio, context))
```

建议λ=0.1-0.3，需要根据验证集调整。

+ **<font style="color:rgb(26, 28, 30);">是否需要增加热词召回损失？</font>**
    - **<font style="color:rgb(26, 28, 30);">建议初期不要增加</font>**<font style="color:rgb(26, 28, 30);">。</font>
    - **<font style="color:rgb(26, 28, 30);">原因</font>**<font style="color:rgb(26, 28, 30);">：</font>
        * **<font style="color:rgb(26, 28, 30);">增加复杂性</font>**<font style="color:rgb(26, 28, 30);">：需要额外设计损失函数，并平衡其与主损失的权重，调参困难。</font>
        * **<font style="color:rgb(26, 28, 30);">潜在风险</font>**<font style="color:rgb(26, 28, 30);">：可能会导致模型“死记硬背”，倾向于无脑地把Prompt里的热词复制到输出中，而忽略了音频本身的实际内容，损害模型的泛化能力。</font>
    - **<font style="color:rgb(26, 28, 30);"></font>**<font style="color:rgb(26, 28, 30);">先用最简单、最经典的交叉熵损失进行训练。如果微调后发现热词识别率仍然不理想，再考虑将其作为一种高级优化手段进行尝试。</font>



#### 2 KimiAudio冻结策略
两种选择

1. 全量微调adapter层，冻结KimiAudio Encoder和LLM
2. LoRA训练 LLM最后基层



### 训练策略
#### 1. 数据增强
+ **热词替换**：随机替换transcription中的词为同音词/近音词
+ **热词遮蔽**：随机遮蔽部分热词，训练模型的推理能力
+ **顺序打乱**：打乱热词列表顺序，避免位置偏见
+ **热词干扰： **部分热词和target transcription中的词同音/近音，但语义有较大差异





#### 3.3.3 训练超参数
+ Batch size: 16-32
+ Learning rate: 1e-5（adapter）, 1e-6（主模型）
+ Warmup steps: 1000
+ Total steps: 50k-100k
+ Gradient accumulation: 16





## 4. 评估指标
+ **整体WER**：标准词错误率
+ **B-WER**：热词的词错误率
+ **U-WER**：非热词的词错误率
+ **热词召回率**：正确识别的热词比例
+ **抗干扰能力**：干扰词被错误识别的比例



## 5. 实施建议
1. 先在小规模数据上验证方案可行性
2. 逐步增加数据规模和模型解冻程度
3. 重点关注热词召回和整体识别的平衡
4. 可以考虑多任务学习，同时训练热词检测任务

<font style="color:rgb(26, 28, 30);"></font>

