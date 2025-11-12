# RheoServe: A Heterogeneous Acceleration Engine for Sparse Attention Models

# Introduction

Large Language Models has deeply integrated into our daily life, and its capability keeps growing following the scalaing law. 

Although the scaling law slows down on models size and pretrained scale, a new trend of scaling: test time scaling /compute time scaling becames a spotlight. Encoupled with reinformcement learning,  The new scaling method encourages LLM to discover more possibilities though its own chain of thoughts. As a result, the context length requirement keeps growing, from 4k to 16k, 32k, and even 128k. KV Cache technology, which stores the key and value vectors of past tokens to avoid redundant computation, becomes a critical bottleneck for LLM serving systems. However, the KV cache size grows linearly with the context length, leading to significant memory consumption and bandwidth requirements. 


To reduce the inference cost of long-context LLMs, previous works [7, 22, 24, 27, 36, 39, 40, 45] have explored methods to compress the KV cache required during generation. They have demonstrated
 that a small portion of critical tokens largely determines the generated token, and the criticalities
 of the tokens vary with different query tokens. Specifically, during self-attention computation, the
 attention scores (referred to QTK in this paper) of these critical tokens are significantly larger than
 those of other tokens. As a result, the attention computation for each query token can be accurately
 approximated by involving only the KV cache of its critical tokens.

Based on the observation, dynamic sparse attention algorithms (DSAes) [7, 36, 39] for KV cache
 have been proposed to conduct the attention process in a select-then-compute manner. To be specific,
 DSAes manage the KV cache at the block level, where each block contains consecutive tokens. For
 each KV block, DSAes maintain small metadata derived from the token keys to represent the tokens
 in the block. During inference, DSAes first estimate the criticalities of all KV blocks using their
 metadata and the query token. They then select the top-k most critical KV blocks to perform the
 approximate attention. The value of k is predefined to balance efficiency and accuracy (e.g., 64, 128).
 By offloading non-critical KV blocks to host memory and loading only the selected KV blocks into
 GPUs, DSAes significantly reduce the GPU memory consumption, thereby enabling the efficient
 processing of long sequences


Softmax technoloy, dampens most of the attentions, 

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


problem



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




fast retrival / scheduling



可以加一个热力图，表示attention的稀疏度 （分区域，retrival 区域，window区域，sink 区域）
![alt text](image.png)

可以加一个解释head 组织形式的图




一张大图，用黑色符号标注出来对应文字描述的算法/算法的行流程
fastcache


介绍一下cache recorder的作用
介绍一下budget control的核心作用


要想办法将 retrival + offload 这个故事能够结合起来，同时围绕一个核心就是page table


介绍一下算法流程


介绍一下head粒度的pagetable 组织
介绍一下cpu 的 GQA 特性，然后等效带宽的概念

介绍一下介绍一下gpu端的page table 模拟








================================================================================
Cache Hit Rate Statistics (Averaged over all decode iterations)
================================================================================

Transaction 0 (Total samples: 1914912):
  Overall Average: 0.9140 (91.40%)
  Per Layer Average:
    Layer  0: 0.9167 (91.67%)
    Layer  1: 0.9012 (90.12%)
    Layer  2: 0.9256 (92.56%)
    Layer  3: 0.9239 (92.39%)
    Layer  4: 0.9309 (93.09%)
    Layer  5: 0.9327 (93.27%)
    Layer  6: 0.9233 (92.33%)
    Layer  7: 0.8969 (89.69%)
    Layer  8: 0.9087 (90.87%)
    Layer  9: 0.8985 (89.85%)
    Layer 10: 0.9225 (92.25%)
    Layer 11: 0.9167 (91.67%)
    Layer 12: 0.9125 (91.25%)
    Layer 13: 0.8980 (89.80%)
    Layer 14: 0.9147 (91.47%)
    Layer 15: 0.9025 (90.25%)
    Layer 16: 0.9128 (91.28%)
    Layer 17: 0.9029 (90.29%)
    Layer 18: 0.9083 (90.83%)
    Layer 19: 0.9130 (91.30%)
    Layer 20: 0.9022 (90.22%)
    Layer 21: 0.9074 (90.74%)
    Layer 22: 0.9187 (91.87%)
    Layer 23: 0.9071 (90.71%)
    Layer 24: 0.9081 (90.81%)
    Layer 25: 0.9179 (91.79%)
    Layer 26: 0.9141 (91.41%)
    Layer 27: 0.9230 (92.30%)
    Layer 28: 0.9122 (91.22%)
    Layer 29: 0.9165 (91.65%)
    Layer 30: 0.9209 (92.09%)
    Layer 31: 0.9222 (92.22%)
    Layer 32: 0.9181 (91.81%)
    Layer 33: 0.9214 (92.14%)
    Layer 34: 0.9209 (92.09%)
    Layer 35: 0.9129 (91.29%)
  Per Head Average:
    Head  0: 0.9193 (91.93%)
    Head  1: 0.9107 (91.07%)
    Head  2: 0.9130 (91.30%)
    Head  3: 0.9112 (91.12%)
    Head  4: 0.9154 (91.54%)
    Head  5: 0.9149 (91.49%)
    Head  6: 0.9165 (91.65%)
    Head  7: 0.9114 (91.14%)

Transaction 1 (Total samples: 1914912):
  Overall Average: 0.9174 (91.74%)
  Per Layer Average:
    Layer  0: 0.9167 (91.67%)
    Layer  1: 0.9009 (90.09%)
    Layer  2: 0.9285 (92.85%)
    Layer  3: 0.9296 (92.96%)
    Layer  4: 0.9344 (93.44%)
    Layer  5: 0.9356 (93.56%)
    Layer  6: 0.9284 (92.84%)
    Layer  7: 0.8984 (89.84%)
    Layer  8: 0.9124 (91.24%)
    Layer  9: 0.9059 (90.59%)
    Layer 10: 0.9227 (92.27%)
    Layer 11: 0.9162 (91.62%)
    Layer 12: 0.9142 (91.42%)
    Layer 13: 0.9056 (90.56%)
    Layer 14: 0.9174 (91.74%)
    Layer 15: 0.9081 (90.81%)
    Layer 16: 0.9159 (91.59%)
    Layer 17: 0.9104 (91.04%)
    Layer 18: 0.9138 (91.38%)
    Layer 19: 0.9172 (91.72%)
    Layer 20: 0.9108 (91.08%)
    Layer 21: 0.9125 (91.25%)
    Layer 22: 0.9211 (92.11%)
    Layer 23: 0.9106 (91.06%)
    Layer 24: 0.9152 (91.52%)
    Layer 25: 0.9199 (91.99%)
    Layer 26: 0.9173 (91.73%)
    Layer 27: 0.9254 (92.54%)
    Layer 28: 0.9166 (91.66%)
    Layer 29: 0.9182 (91.82%)
    Layer 30: 0.9221 (92.21%)
    Layer 31: 0.9240 (92.40%)
    Layer 32: 0.9200 (92.00%)
    Layer 33: 0.9246 (92.46%)
    Layer 34: 0.9240 (92.40%)
    Layer 35: 0.9133 (91.33%)
  Per Head Average:
    Head  0: 0.9214 (92.14%)
    Head  1: 0.9148 (91.48%)
    Head  2: 0.9167 (91.67%)
    Head  3: 0.9156 (91.56%)
    Head  4: 0.9186 (91.86%)
    Head  5: 0.9182 (91.82%)
    Head  6: 0.9192 (91.92%)
    Head  7: 0.9151 (91.51%)

Transaction 2 (Total samples: 1914912):
  Overall Average: 0.9159 (91.59%)
  Per Layer Average:
    Layer  0: 0.9161 (91.61%)
    Layer  1: 0.9033 (90.33%)
    Layer  2: 0.9305 (93.05%)
    Layer  3: 0.9295 (92.95%)
    Layer  4: 0.9347 (93.47%)
    Layer  5: 0.9359 (93.59%)
    Layer  6: 0.9297 (92.97%)
    Layer  7: 0.8976 (89.76%)
    Layer  8: 0.9104 (91.04%)
    Layer  9: 0.9014 (90.14%)
    Layer 10: 0.9234 (92.34%)
    Layer 11: 0.9174 (91.74%)
    Layer 12: 0.9139 (91.39%)
    Layer 13: 0.9008 (90.08%)
    Layer 14: 0.9158 (91.58%)
    Layer 15: 0.9043 (90.43%)
    Layer 16: 0.9151 (91.51%)
    Layer 17: 0.9054 (90.54%)
    Layer 18: 0.9109 (91.09%)
    Layer 19: 0.9129 (91.29%)
    Layer 20: 0.9051 (90.51%)
    Layer 21: 0.9090 (90.90%)
    Layer 22: 0.9193 (91.93%)
    Layer 23: 0.9093 (90.93%)
    Layer 24: 0.9106 (91.06%)
    Layer 25: 0.9192 (91.92%)
    Layer 26: 0.9168 (91.68%)
    Layer 27: 0.9232 (92.32%)
    Layer 28: 0.9152 (91.52%)
    Layer 29: 0.9177 (91.77%)
    Layer 30: 0.9200 (92.00%)
    Layer 31: 0.9228 (92.28%)
    Layer 32: 0.9188 (91.88%)
    Layer 33: 0.9223 (92.23%)
    Layer 34: 0.9229 (92.29%)
    Layer 35: 0.9129 (91.29%)
  Per Head Average:
    Head  0: 0.9207 (92.07%)
    Head  1: 0.9126 (91.26%)
    Head  2: 0.9147 (91.47%)
    Head  3: 0.9136 (91.36%)
    Head  4: 0.9173 (91.73%)
    Head  5: 0.9170 (91.70%)
    Head  6: 0.9181 (91.81%)
    Head  7: 0.9136 (91.36%)

Transaction 3 (Total samples: 1914912):
  Overall Average: 0.9154 (91.54%)
  Per Layer Average:
    Layer  0: 0.9168 (91.68%)
    Layer  1: 0.9009 (90.09%)
    Layer  2: 0.9314 (93.14%)
    Layer  3: 0.9322 (93.22%)
    Layer  4: 0.9370 (93.70%)
    Layer  5: 0.9373 (93.73%)
    Layer  6: 0.9315 (93.15%)
    Layer  7: 0.8994 (89.94%)
    Layer  8: 0.9120 (91.20%)
    Layer  9: 0.9068 (90.68%)
    Layer 10: 0.9232 (92.32%)
    Layer 11: 0.9172 (91.72%)
    Layer 12: 0.9148 (91.48%)
    Layer 13: 0.9020 (90.20%)
    Layer 14: 0.9171 (91.71%)
    Layer 15: 0.9027 (90.27%)
    Layer 16: 0.9126 (91.26%)
    Layer 17: 0.9068 (90.68%)
    Layer 18: 0.9098 (90.98%)
    Layer 19: 0.9126 (91.26%)
    Layer 20: 0.9052 (90.52%)
    Layer 21: 0.9080 (90.80%)
    Layer 22: 0.9151 (91.51%)
    Layer 23: 0.9095 (90.95%)
    Layer 24: 0.9126 (91.26%)
    Layer 25: 0.9134 (91.34%)
    Layer 26: 0.9113 (91.13%)
    Layer 27: 0.9187 (91.87%)
    Layer 28: 0.9140 (91.40%)
    Layer 29: 0.9151 (91.51%)
    Layer 30: 0.9191 (91.91%)
    Layer 31: 0.9211 (92.11%)
    Layer 32: 0.9154 (91.54%)
    Layer 33: 0.9206 (92.06%)
    Layer 34: 0.9224 (92.24%)
    Layer 35: 0.9088 (90.88%)
  Per Head Average:
    Head  0: 0.9185 (91.85%)
    Head  1: 0.9135 (91.35%)
    Head  2: 0.9145 (91.45%)
    Head  3: 0.9148 (91.48%)
    Head  4: 0.9163 (91.63%)
    Head  5: 0.9158 (91.58%)
    Head  6: 0.9163 (91.63%)
    Head  7: 0.9135 (91.35%)
================================================================================


================================================================================
Cache Hit Rate Statistics (Averaged over all decode iterations)
================================================================================

Transaction 0 (Total samples: 1090368):
  Overall Average: 0.8483 (84.83%)
  Per Layer Average:
    Layer  0: 0.8463 (84.63%)
    Layer  1: 0.8288 (82.88%)
    Layer  2: 0.8297 (82.97%)
    Layer  3: 0.8365 (83.65%)
    Layer  4: 0.8318 (83.18%)
    Layer  5: 0.8239 (82.39%)
    Layer  6: 0.8282 (82.82%)
    Layer  7: 0.8382 (83.82%)
    Layer  8: 0.8466 (84.66%)
    Layer  9: 0.8587 (85.87%)
    Layer 10: 0.8265 (82.65%)
    Layer 11: 0.8349 (83.49%)
    Layer 12: 0.8444 (84.44%)
    Layer 13: 0.8503 (85.03%)
    Layer 14: 0.8365 (83.65%)
    Layer 15: 0.8524 (85.24%)
    Layer 16: 0.8513 (85.13%)
    Layer 17: 0.8640 (86.40%)
    Layer 18: 0.8574 (85.74%)
    Layer 19: 0.8556 (85.56%)
    Layer 20: 0.8652 (86.52%)
    Layer 21: 0.8550 (85.50%)
    Layer 22: 0.8525 (85.25%)
    Layer 23: 0.8568 (85.68%)
    Layer 24: 0.8663 (86.63%)
    Layer 25: 0.8515 (85.15%)
    Layer 26: 0.8603 (86.03%)
    Layer 27: 0.8573 (85.73%)
    Layer 28: 0.8516 (85.16%)
    Layer 29: 0.8603 (86.03%)
    Layer 30: 0.8553 (85.53%)
    Layer 31: 0.8526 (85.26%)
    Layer 32: 0.8541 (85.41%)
    Layer 33: 0.8604 (86.04%)
    Layer 34: 0.8523 (85.23%)
    Layer 35: 0.8444 (84.44%)
  Per Head Average:
    Head  0: 0.8446 (84.46%)
    Head  1: 0.8526 (85.26%)
    Head  2: 0.8494 (84.94%)
    Head  3: 0.8491 (84.91%)
    Head  4: 0.8485 (84.85%)
    Head  5: 0.8470 (84.70%)
    Head  6: 0.8460 (84.60%)
    Head  7: 0.8489 (84.89%)

Transaction 1 (Total samples: 1090368):
  Overall Average: 0.8422 (84.22%)
  Per Layer Average:
    Layer  0: 0.8422 (84.22%)
    Layer  1: 0.8251 (82.51%)
    Layer  2: 0.8329 (83.29%)
    Layer  3: 0.8380 (83.80%)
    Layer  4: 0.8377 (83.77%)
    Layer  5: 0.8279 (82.79%)
    Layer  6: 0.8323 (83.23%)
    Layer  7: 0.8339 (83.39%)
    Layer  8: 0.8396 (83.96%)
    Layer  9: 0.8466 (84.66%)
    Layer 10: 0.8233 (82.33%)
    Layer 11: 0.8306 (83.06%)
    Layer 12: 0.8352 (83.52%)
    Layer 13: 0.8421 (84.21%)
    Layer 14: 0.8324 (83.24%)
    Layer 15: 0.8417 (84.17%)
    Layer 16: 0.8405 (84.05%)
    Layer 17: 0.8500 (85.00%)
    Layer 18: 0.8484 (84.84%)
    Layer 19: 0.8462 (84.62%)
    Layer 20: 0.8484 (84.84%)
    Layer 21: 0.8434 (84.34%)
    Layer 22: 0.8461 (84.61%)
    Layer 23: 0.8476 (84.76%)
    Layer 24: 0.8578 (85.78%)
    Layer 25: 0.8475 (84.75%)
    Layer 26: 0.8481 (84.81%)
    Layer 27: 0.8510 (85.10%)
    Layer 28: 0.8411 (84.11%)
    Layer 29: 0.8508 (85.08%)
    Layer 30: 0.8509 (85.09%)
    Layer 31: 0.8483 (84.83%)
    Layer 32: 0.8504 (85.04%)
    Layer 33: 0.8515 (85.15%)
    Layer 34: 0.8462 (84.62%)
    Layer 35: 0.8433 (84.33%)
  Per Head Average:
    Head  0: 0.8405 (84.05%)
    Head  1: 0.8448 (84.48%)
    Head  2: 0.8428 (84.28%)
    Head  3: 0.8433 (84.33%)
    Head  4: 0.8433 (84.33%)
    Head  5: 0.8406 (84.06%)
    Head  6: 0.8405 (84.05%)
    Head  7: 0.8417 (84.17%)

Transaction 2 (Total samples: 1090368):
  Overall Average: 0.8550 (85.50%)
  Per Layer Average:
    Layer  0: 0.8521 (85.21%)
    Layer  1: 0.8360 (83.60%)
    Layer  2: 0.8536 (85.36%)
    Layer  3: 0.8569 (85.69%)
    Layer  4: 0.8563 (85.63%)
    Layer  5: 0.8473 (84.73%)
    Layer  6: 0.8516 (85.16%)
    Layer  7: 0.8427 (84.27%)
    Layer  8: 0.8538 (85.38%)
    Layer  9: 0.8610 (86.10%)
    Layer 10: 0.8350 (83.50%)
    Layer 11: 0.8461 (84.61%)
    Layer 12: 0.8535 (85.35%)
    Layer 13: 0.8600 (86.00%)
    Layer 14: 0.8455 (84.55%)
    Layer 15: 0.8586 (85.86%)
    Layer 16: 0.8529 (85.29%)
    Layer 17: 0.8624 (86.24%)
    Layer 18: 0.8575 (85.75%)
    Layer 19: 0.8506 (85.06%)
    Layer 20: 0.8660 (86.60%)
    Layer 21: 0.8560 (85.60%)
    Layer 22: 0.8499 (84.99%)
    Layer 23: 0.8586 (85.86%)
    Layer 24: 0.8714 (87.14%)
    Layer 25: 0.8522 (85.22%)
    Layer 26: 0.8645 (86.45%)
    Layer 27: 0.8603 (86.03%)
    Layer 28: 0.8589 (85.89%)
    Layer 29: 0.8650 (86.50%)
    Layer 30: 0.8607 (86.07%)
    Layer 31: 0.8623 (86.23%)
    Layer 32: 0.8537 (85.37%)
    Layer 33: 0.8628 (86.28%)
    Layer 34: 0.8570 (85.70%)
    Layer 35: 0.8459 (84.59%)
  Per Head Average:
    Head  0: 0.8514 (85.14%)
    Head  1: 0.8595 (85.95%)
    Head  2: 0.8556 (85.56%)
    Head  3: 0.8560 (85.60%)
    Head  4: 0.8569 (85.69%)
    Head  5: 0.8517 (85.17%)
    Head  6: 0.8526 (85.26%)
    Head  7: 0.8560 (85.60%)

Transaction 3 (Total samples: 1090368):
  Overall Average: 0.8508 (85.08%)
  Per Layer Average:
    Layer  0: 0.8450 (84.50%)
    Layer  1: 0.8287 (82.87%)
    Layer  2: 0.8347 (83.47%)
    Layer  3: 0.8420 (84.20%)
    Layer  4: 0.8349 (83.49%)
    Layer  5: 0.8265 (82.65%)
    Layer  6: 0.8320 (83.20%)
    Layer  7: 0.8388 (83.88%)
    Layer  8: 0.8461 (84.61%)
    Layer  9: 0.8588 (85.88%)
    Layer 10: 0.8282 (82.82%)
    Layer 11: 0.8346 (83.46%)
    Layer 12: 0.8451 (84.51%)
    Layer 13: 0.8547 (85.47%)
    Layer 14: 0.8386 (83.86%)
    Layer 15: 0.8554 (85.54%)
    Layer 16: 0.8536 (85.36%)
    Layer 17: 0.8651 (86.51%)
    Layer 18: 0.8606 (86.06%)
    Layer 19: 0.8573 (85.73%)
    Layer 20: 0.8672 (86.72%)
    Layer 21: 0.8570 (85.70%)
    Layer 22: 0.8579 (85.79%)
    Layer 23: 0.8610 (86.10%)
    Layer 24: 0.8700 (87.00%)
    Layer 25: 0.8546 (85.46%)
    Layer 26: 0.8632 (86.32%)
    Layer 27: 0.8620 (86.20%)
    Layer 28: 0.8564 (85.64%)
    Layer 29: 0.8641 (86.41%)
    Layer 30: 0.8566 (85.66%)
    Layer 31: 0.8554 (85.54%)
    Layer 32: 0.8579 (85.79%)
    Layer 33: 0.8660 (86.60%)
    Layer 34: 0.8563 (85.63%)
    Layer 35: 0.8431 (84.31%)
  Per Head Average:
    Head  0: 0.8461 (84.61%)
    Head  1: 0.8551 (85.51%)
    Head  2: 0.8528 (85.28%)
    Head  3: 0.8521 (85.21%)
    Head  4: 0.8512 (85.12%)
    Head  5: 0.8494 (84.94%)
    Head  6: 0.8480 (84.80%)
    Head  7: 0.8518 (85.18%)
================================================================================