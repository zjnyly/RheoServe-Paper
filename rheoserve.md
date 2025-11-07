# RheoServe: A Heterogeneous Acceleration Engine for Sparse Attention Models

# Introduction

# Related Work
## Sparse Attention 
Sparse and approximate attention mechanisms have been proposed to alleviate the O(n2) computational burden and O(n) memory access, thereby accelerating LLM inference.  Early approaches, including StreamingLLM (Xiao et al., 2023) and LM-Infinite (Han et al., 2023),
 employ simple and fixed sparsity patterns, such as attention sinks and sliding windows, to reduce attention
 complexity. More advanced methods (Ge et al., 2023; Jiang et al., 2024; Xu et al., 2025; Xiao et al.,
 2024b; Zhang et al., 2025a;d) introduce richer sparsity structures for improved task performance, often
 10
Working in Progress
 combined with other optimizations like low-bit quantization (Zhang et al., 2025b;c; Yang et al., 2024).
 Among these methods, query-aware sparsification (selecting only a subset of query-related KV pairs) is a
 widely adopted strategy (Li et al., 2024; Zhang et al., 2023; Liu et al., 2023). In contrast, query-agnostic
 selection methods (Huang et al., 2024; Kim et al., 2024; Yao et al., 2024) focus on selecting essential KV
 pairs without queries, but suffer from performance degradation when directly applied to large-scale LLMs.
 More recent research demonstrates that integrating attention sparsity directly into model training can yield
 nearly lossless performance while maintaining substantial sparsity and speedup. NSA (Yuan et al., 2025),
 SeerAttention (Gao et al., 2024), and MoBA (Lu et al., 2025) introduce trainable block-sparse patterns
 tailored for long-context scenarios, though these approaches tend to slow down short-context inference.
 InfLLM-V2 (Zhao et al., 2025) addresses this limitation by unifying multiple sparsity patterns within a
 single attention kernel, enabling dynamic adaptation between short and long contexts. FSA (Yan et al.,
 2025) further improves kernel efficiency for smaller query-head number, while DSA (DeepSeek-AI, 2025)
 introduces element-wise trainable sparsity suitable for ultra-large LLMs. DMA (Shi et al., 2025) extends
 this line of work by incorporating eviction-based sparse selection. Beyond natural language processing,
 trainable sparse attention has also been successfully applied to long-video generation (Zhang et al., 2025e;
 Zhan et al., 2025), signal processing (Wang et al., 2025), and computational biology (Yoshai et al., 2025).

 
## LLM Accelerating Systems

