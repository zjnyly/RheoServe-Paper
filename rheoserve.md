# RheoServe: A Heterogeneous Acceleration Engine for Sparse Attention Models

# Introduction

Large Language Models has deeply integrated into our daily life, and its capability keeps growing following the scalaing law. 

Although the scaling law slows down on models size and pretrained scale, a new trend of scaling: test time scaling /compute time scaling becames a spotlight. Encoupled with reinformcement learning,  The new scaling method encourages LLM to discover more possibilities though its own chain of thoughts. 

followes an auto-regressive way
means long decoding
kv cache, however, it goes longer

softmax technoloy, dampens most of the attentions, 

so sparse is a viable way. Main stream models deepseek enploy dynamic sparse attention in deepseek v3.2

most works focus on keeping a small range of data, althou offloading, the bottle neck is fixed by transter speed like PCIE

however, if we manage a cache pool, the bottel neck is closely related to cache miss rate, if the missrate is low, than it is good


vllm although xxx  but it mainly focus on xxx

however, the offload bandwidth is also vital. LSE-softmax, enbles parallel processing of hetetogenies devices. So it is possible to utilize ... for preson platform with double channle DDR5, the possible is , much higher than PCIE 4.0 


Nvidia GTC 2025, the new vera rubin arch, also a possibility. 


to sumarize , a , b, c, and for these ..., we make xxx improvements
xxx
xxx
xxx



可以加一个热力图，表示attention的稀疏度
![alt text](image.png)


# Related Work
## Efficient Sparse Attention 

Sparsity is a natural of softmax bast attention mechenism.
method classification:

drop: StreamingLLM, H2O, snapKV

drop based face missing important tokens

full dynamic: quest, MagicPIG, hashattention, seerattention

directly trained model: NSA, MOBA, DSA. 

However, these method keeps whole bounch of kv cache, which require 


There are also some focus on long context reasonning, 
A key focus point is for long reasoning tasks, 

Our system, unifies the evict and retrive system, using retrive for xxx use evict for ...



 
## LLM Acceleration Systems
The key point for LLM serving is the management and interaction with kv cache.

vanilla serving system
vllm
sglang
flashinfer all support paged attention 
flashattention with online softmax (above)

flashinfer with lse attention

Dynamic Attention Kernel Support:
Quest, SeerAttention

offloading system
 hgca, sparseserve focus on pcie transfer, magipig offload computation with cpu, but it lacks page management, also, it only perform one calculation, which will result in context drift.

 Our system, on the other hand.

however, 






 Early approaches, including StreamingLLM (Xiao et al., 2023) and LM-Infinite (Han et al., 2023),
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
 1)    further improves kernel efficiency for smaller query-head number, while DSA (DeepSeek-AI, 2025)
 introduces element-wise trainable sparsity suitable for ultra-large LLMs. DMA (Shi et al., 2025) extends
 this line of work by incorporating eviction-based sparse selection. Beyond natural language processing,
 trainable sparse attention has also been successfully applied to long-video generation (Zhang et al., 2025e;
 Zhan et al., 2025), signal processing (Wang et al., 2025), and computational biology (Yoshai et al., 2025).



fast retrival / scheduling