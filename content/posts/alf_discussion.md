+++
date = '2026-08-22T00:42:10+08:00'
draft = false
title = '关于 ALF 中 Bias 引起的 MoE 路由翻转的简要讨论'
categories = ['Learning']
tags = ['LLM','MoE','ALF']
+++

# 问题

在 Auxiliary-Loss-Free（ALF）方法中，采用 $\operatorname{TopK}_{s+b}$ 来选择 expert。

correction bias $\mathbf{b}$ 的学习可以改善负载均衡。但是在后续的 gated output 中，仍然采用原始 score $s_i$ 来加权。这就会自然引入一个问题：是否会因为某个 expert 相对于其他 expert 的 bias 差 $b_i-b_j$ 很大，使一个原本 $s_i$ 很小的 expert i 被激活？换言之，明明模型计算出的 $s_i$ 认为 expert i 不应该被选入，它是否仍会由于负载均衡的要求而被选入？

# TL;DR

- 只要 $\mathbf{b}$ 在 expert 之间不完全相同，Top-K 就可能相对于 raw-score Top-K 发生翻转；是否频繁取决于 score gap 与 bias 差的联合分布。**仅凭 ALF 机制本身，无法保证翻转概率的上界，也无法保证最大翻转程度 $b_i-b_j$ 的上界。**
- 在 Moonlight-16B-A3B 的真实文本激活样本中，普通翻转很常见：56.24% 的 token-layer 改变了 Top-K 集合；**但接近零贡献（$s_i\approx 0$）的事件很少，最后的加权输出中，占比小于 0.01 和 0.001 的 expert 比例分别为 0.1926% 和 0.00063%。**
- 这种换入的影响还有待评估，可能有危害，也可能完全没有危害。**直觉上，低权重换入的 expert 自身贡献很小，主要对应潜在计算浪费；但它可能同时排除一个重要的高 raw-score expert**，因此整次翻转对模型输出的影响不一定小。当前小规模实验没有显示严重的总体性能问题，但还不足以给出普适的质量结论。

# 对开源 checkpoint 的实际诊断

## bias range 与翻转的确定性关系

先考虑普通 global Top-K。raw-score Top-K 集合记为 $R=\operatorname{TopK}_s$，实际集合记为 $A=\operatorname{TopK}_{s+b}$。假设 $e_{\mathrm{in}}$ 是被 bias 换入的 expert，即 $e_{\mathrm{in}}$ 属于 $A$ 但不属于 $R$；对应地，存在一个被换出的 expert $e_{\mathrm{out}}$，即 $e_{\mathrm{out}}$ 属于 $R$ 但不属于 $A$。由 Top-K 选择规则可得

$$
s_{\mathrm{in}}+b_{\mathrm{in}}\ge s_{\mathrm{out}}+b_{\mathrm{out}}.
$$

因此

$$
s_{\mathrm{out}}-s_{\mathrm{in}}
\le b_{\mathrm{in}}-b_{\mathrm{out}}
\le b_{\max}-b_{\min}=B.
$$

其中 $B=\operatorname{range}(b)$ 是层内有效 bias 范围。又因为 $e_{\mathrm{out}}$ 属于 raw-score Top-K，所以 $s_{\mathrm{out}}\ge s_{(K)}$，从而

$$
s_{\mathrm{in}}\ge s_{(K)}-B.
$$

这是一个严格的 raw-score 损失上界，但它不能单独限制换入 expert 的 raw rank 或归一化 mixture weight，但是可以方便我们后续诊断分析实际的 checkpoint。主流模型使用 Sigmoid 产生 $s_i$，其值在 0 到 1 之间（DeepSeek作为ALF的开山鼻祖，在V4版本主动启用Sigmoid, 使用 $\sqrt{\operatorname{softplus}}$计算路由分数，因此其 $\Delta b$ 数值不宜与 Sigmoid 模型直接横比）。

## checkpoint 中的 bias 有效范围

对每一个 learned-router 层 $\ell$，定义有效 bias 范围

$$
\Delta b_\ell=\max_i b_{\ell,i}-\min_i b_{\ell,i}.
$$

下表直接从一些开源 checkpoint 的 router bias tensor 逐层计算并报告 $\Delta b_\ell$。

| 模型 | score 激活 | $N$ | $K$ | $\Delta b$ 中位数 | $\Delta b$ 最大值 |
|---|:---:|---:|---:|---:|---:|
| Moonlight-16B-A3B | sigmoid | 64 | 6 | 0.19226 | 0.24463 |
| DeepSeek V3/R1 family | sigmoid | 256 | 8 | 0.04520 | 0.22078 |
| GLM 4.5 Air | sigmoid | 128 | 8 | 0.18620 | 0.37493 |
| GLM 4.7 Flash | sigmoid | 64 | 4 | 0.16091 | 0.26885 |
| GLM 4.5/4.7 | sigmoid | 160 | 8 | 0.26385 | 0.57069 |
| GLM 4.6 | sigmoid | 160 | 8 | 0.26546 | 0.57069 |
| GLM 5/5.1/5.2 | sigmoid | 256 | 8 | 0.32037 | 0.64270 |
| Kimi K2 | sigmoid | 384 | 8 | 0.27525 | 0.78318 |
| Kimi K2.5 | sigmoid | 384 | 8 | 0.33179 | 0.77349 |
| Kimi K3 | sigmoid | 896 | 16 | 0.34118 | 0.84703 |

**表 1：checkpoint 静态审计：逐层 correction-bias 有效范围。**

从中位数来看，$\Delta b$ 的值相对来说可以接受；但是 GLM 和 Kimi 的最大值仍然可能产生显著的换入换出效应。值得注意的是，DeepSeek V3 系列的 $\Delta b$ 最大值甚至小于 GLM 和 Kimi 的中位数，其中动力学成因倒是有待勘探。

## 翻转情况的统计

Moonlight-16B-A3B 具有 64 个 routed experts、Top-6 和 26 个 MoE 层。实验使用 32 条、16 个领域的真实文本进行完整模型前向，不使用随机 hidden state。本文将实际集合 $A=\operatorname{TopK}_{s+b}$ 中有、但 raw-score 集合 $R=\operatorname{TopK}_s$ 中没有的 expert，即 $A\setminus R$，称为 incoming expert（换入 expert）。相对于每个真实 hidden state 上的 global raw Top-6，得到如下总体统计：

| 指标 | 实测值 | 分母或解释 |
|---|---:|---|
| token-layer 数 | 26,312 | 1,012 tokens $\times$ 26 MoE 层 |
| routed-expert dispatch 数 | 157,872 | 26,312 $\times$ Top-6 |
| Top-K 集合改变 | 14,798（56.24%） | 全部 token-layer |
| 换入 expert 数 | 19,654 | 平均 0.747 个/token-layer |
| incoming raw rank $>32$ | 817（4.16%） | 全部 19,654 个 incoming |
| score 损失 $\ge 0.1$ | 338（1.72%） | $s_{(K)}-s_i$，全部 incoming |
| mixture weight $<0.01$ | 304（0.1926%） | 全部 dispatch；其中 303 个为 incoming |
| mixture weight $<0.001$ | 1（0.00063%） | 全部 dispatch；该事件为 incoming |

**表 2：Moonlight 真实激活的总体翻转与低贡献事件统计。**

只看“Top-K 是否改变”会把正常的边界重排序和真正低贡献的翻转混在一起。下面进一步报告换入 expert 的 raw score、raw rank、相对 raw Top-K 边界的 score 损失，以及实际 mixture weight 分布：

| 指标 | p01 | p05 | p50 | p95 | 最大 |
|---|---:|---:|---:|---:|---:|
| incoming raw score | 0.0083 | 0.0414 | 0.1476 | 0.9564 | 0.9832 |
| incoming raw rank | 7 | 7 | 8 | 30 | 64 |
| $s_{(K)}-s_i$ | 0.0004 | 0.0023 | 0.0237 | 0.0808 | 0.1899 |
| incoming mixture weight | 0.0063 | 0.0217 | 0.0628 | 0.1656 | 0.1709 |
| raw K/K+1 gap | 0.0002 | 0.0011 | 0.0198 | 0.1047 | 0.4415 |

**表 3：Moonlight 换入 expert 与 raw Top-K 边界的分布统计。**

低贡献事件并非均匀分布在所有层。第 3、5 和 26 层具有最高的 weight 小于 0.01 dispatch 比例：

| 层 | Top-K 改变 | 换入/token | rank p95 | weight p01 | $w<0.01$ |
|---:|---:|---:|---:|---:|---:|
| 3 | 58.60% | 0.842 | 39 | 0.00166 | 1.515% |
| 5 | 53.95% | 0.754 | 61 | 0.00952 | 0.3458% |
| 26 | 43.68% | 0.623 | 53 | 0.00218 | 2.652% |
| 全部层 | 56.24% | 0.747 | 30 | 0.00626 | 0.1926% |

**表 4：低贡献尾部最集中的 Moonlight 层。**

因此，在当前 Moonlight 样本中，普通翻转是常态，但绝大多数换入 expert 仍具有正常的 mixture contribution；真正接近零贡献的事件是稀少且层集中的尾部。当然，该结论来自 Moonlight 16B-A3B 的真实激活，不能直接外推到更大的 DeepSeek-V3、Kimi K3、GLM-5。

# 不严谨的数学解释

为什么 ALF 本身没有无条件的概率保证，但在实际 checkpoint 诊断中又没有发现大量接近零贡献的翻转？下面给出两个不算特别严谨的直观解释，权作理解，不太能够深究。

## $\mathbf{b}$ 的对称初始化和负反馈更新

$\mathbf{b}$ 的更新规则为：

$$
\mathbf{b}\leftarrow\mathbf{b}-\gamma\operatorname{sign}(\mathbf{F}-\mathbf{Q}).
$$

其中 $\mathbf{F}$ 表示按全部 routed assignments 归一化后的实际负载分配，$\mathbf{Q}=(1/N,1/N,\ldots,1/N)$ 表示理想的负载分配。若 $F_i$ 定义为每个 token 选中 expert i 的概率，则相应目标应写成 $K/N$。

这里的 sign 起到了负反馈作用：一旦 $F_i>1/N$，就尝试减小 $b_i$；一旦 $F_i<1/N$，就尝试增加 $b_i$。$\mathbf{b}$ 通常从 $\mathbf{0}$ 对称初始化（在 ALF 的原始 paper 中明确提及）。虽然训练会打破这种对称性，但是从直觉上来看，负反馈机制仍能够在某种程度上避免 $\mathbf{b}$ 偏离对称状态太多。另外，DeepSeek 给出的 $\gamma$ 推荐参数是 $10^{-3}$，相比于 Sigmoid 0 到 1 的范围较小，也不至于突然偏离太狠。

## 考虑负载均衡的平衡条件

假设在 token 总体分布上达到了理想的边际负载均衡。令 $\tau(h)$ 表示 token $h$ 的第 $K$ 大 corrected score，则严格的平衡条件是

$$
\Pr\bigl(s_i(h)+b_i\ge\tau(h)\bigr)\approx\frac{K}{N}.
$$

这里 $\tau(h)$ 依赖该 token 的全部 expert scores，并且与 $s_i(h)$ 相关，若进一步采用 mean-field 近似，假设 $\tau(h)$ 可以用近似常数 $\tau$ 代替，并忽略它与 $s_i$ 的相关性，则有

$$
\Pr(s_i\le\tau-b_i)\approx 1-\frac{K}{N}.
$$

假设 $s_i$ 的分布函数是 $F_i$，我们有：

$$
F_i(\tau-b_i)\approx 1-\frac{K}{N}.
$$

整理一下有：

$$
b_i\approx\tau-F_i^{-1}\left(1-\frac{K}{N}\right).
$$

对于 DeepSeek-V3，$K=8,N=256$，所以 $1-K/N=0.96875\approx 0.97$，从而

$$
b_i\approx\tau-F_i^{-1}(0.97).
$$

如果系统保持理想的负载均衡，可以猜想各个专家的分布函数差距不大，因此 $b_i$ 的差距也不会很大。

# Open Questions

- 如何想办法量化 ALF 方法下，这种 expert 换入换出的影响程度？真的有害吗？
- 有没有最后 $\Delta b$ 差距不显著更加严谨的数学解释？
- 如何解释不同厂商的模型 $\Delta b$ 范围的差异？这是因为数据、架构还是训练参数的设定？比如 Kimi 系列、GLM 系列和 DeepSeek 系列各自的 $\Delta b$ 范围相对接近，但是不同厂商的模型差距比较明显；另外，$\Delta b$ 看起来和模型尺寸正相关。
