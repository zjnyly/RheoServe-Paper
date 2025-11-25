# RheoServe: Hybrid Sparse Attention Acceleration for Long‑Context LLM Reasoning

# Introduction

Large Language Models (LLMs) have become indispensable in modern artificial intelligence, demonstrating remarkable capabilities across a wide range of tasks. These models, governed by scaling laws, have achieved significant performance improvements through the expansion of model size and pre-training data. However, as the pace of scaling slows, a new paradigm—test-time scaling or compute-time scaling—has emerged, emphasizing the importance of efficient inference and reasoning over extended contexts. This shift has introduced unprecedented challenges in managing the computational and memory demands of long-context processing.

A critical bottleneck in long-context LLMs lies in the Key-Value (KV) Cache, which stores intermediate representations to avoid redundant computations. The size of the KV Cache grows linearly with context length, leading to substantial memory consumption and bandwidth requirements. Addressing this challenge is essential for enabling LLMs to handle context lengths extending from 4k to 128k tokens and beyond.

Recent advancements have highlighted the inherent sparsity in the attention mechanism of LLMs, where only a subset of tokens significantly influences the generation of new tokens. Dynamic Sparse Attention (DSA) algorithms, such as Quest, MagicPIG, HashAttention, and SeerAttention, exploit this sparsity to reduce computational overhead. However, these methods often retain the entire KV Cache in GPU memory, quickly exhausting capacity for long sequences. Conversely, eviction-based methods like H2O and StreamingLLM manage fixed cache budgets by discarding less critical tokens, but this approach risks losing essential information, leading to context drift and accuracy degradation.

To bridge this gap, retrieval-based methods like SparseServe offload non-critical KV entries to CPU memory, retrieving them as needed. While this ensures accuracy, performance is constrained by the limited bandwidth of PCIe transfers. Emerging hybrid approaches, such as MagicPIG and HGCA, attempt to balance the workload between GPU and CPU, leveraging host memory bandwidth. However, these methods still face challenges in fully exploiting the potential of heterogeneous memory systems.

In this work, we introduce **RheoServe**, a heterogeneous acceleration engine designed to address the limitations of existing sparse attention methods. RheoServe unifies eviction and retrieval mechanisms within a flexible and efficient KV Cache management framework, leveraging both GPU and CPU memory to optimize performance for long-context LLM serving. Our key contributions are summarized as follows:

*   **Fine-Grained KV Cache Management System:** We propose a head-granularity KV Cache management system that unifies the handling of KV Cache across GPU and CPU, effectively reducing cache miss rates and memory bandwidth pressure.
*   **Unified Offload and Retrieval Mechanism:** Through a lightweight **Cache Recorder** and **Budget Controller**, we ensure efficient scheduling of critical KV entries between GPU and CPU memory.
*   **High-Performance CPU-GPU Collaboration:** By constructing a CPU-GPU collaborative computing framework based on Log-Sum-Exp (LSE) softmax, we overcome PCIe bandwidth limitations and unlock optimization potential for future hardware architectures, such as NVIDIA's Vera Rubin.

RheoServe not only addresses the immediate challenges of long-context LLM inference but also paves the way for scalable solutions that align with the evolving demands of AI applications and hardware advancements.

# Background and Related Work

## Efficient Sparse Attention

The sparsity of the attention matrix is a natural property of the softmax mechanism, where a few tokens contribute to the majority of the attention mass. Existing methods to exploit this sparsity can be broadly categorized into three types:

**Eviction-based Methods:** Approaches like StreamingLLM, H2O, and SnapKV manage a fixed cache budget by permanently dropping less important tokens. While efficient, they risk discarding critical information required for future generation steps, leading to context drift and accuracy degradation in long-context reasoning tasks.

**Dynamic Sparse Attention (DSA):** Methods such as Quest, MagicPIG, HashAttention, and SeerAttention dynamically identify and utilize the most critical tokens (top-K) for each query. While they reduce computational overhead, most still require keeping the entire KV cache in GPU memory to support random access, which quickly exhausts memory capacity.

**Directly Trained Models:** Some recent works like NSA, Moba, and DSA propose training models with inherent sparsity constraints. However, these require expensive re-training and are not applicable to existing pre-trained LLMs.

RheoServe distinguishes itself by unifying eviction and retrieval mechanisms. Unlike eviction-based methods that permanently drop data, we offload less critical data to CPU memory. Unlike pure retrieval methods that struggle with bandwidth, we employ a hybrid computation model.

## LLM Acceleration Systems

Efficient management of the KV cache is central to LLM serving performance.

**Paged Attention Systems:** Systems like vLLM, SGLang, and FlashInfer have adopted PagedAttention to manage non-contiguous memory, significantly reducing fragmentation. FlashAttention and its variants optimize the attention kernel itself.

**Dynamic Attention Kernel Support:** Frameworks like Quest and SeerAttention have developed specialized kernels to support sparse attention patterns.

**Offloading Systems:** To handle sequences exceeding GPU memory, systems like HGCA and SparseServe offload KV cache to host memory. However, SparseServe is limited by PCIe transfer latency. MagicPIG attempts to offload computation to the CPU but lacks a sophisticated page management system and performs only a single calculation pass, which can lead to context drift.

RheoServe builds upon these advancements but introduces a unified cache management layer that orchestrates data movement and hybrid computation, addressing the limitations of previous offloading and sparse attention systems.

# RheoServe System Design

In this section, we present the overall architecture of RheoServe, a heterogeneous acceleration engine designed to break the memory wall in sparse attention models. RheoServe leverages a unified memory hierarchy across GPU and CPU to optimize performance for long-context LLM serving.

## System Overview


As illustrated in Figure X, RheoServe seamlessly integrates into existing LLM serving pipelines by replacing the standard attention module. It acts as a transparent middleware that virtualizes the KV cache, abstracting away the complexity of heterogeneous memory management.

The architecture is composed of four subsystems. The **Elastic Transaction Scheduler** manages the lifecycle of requests and orchestrates the prefill-decode workflow, while the **Dual-Track Cache System** serves as a hierarchical storage system comprising the **Quick Access Cache** (for fast sparsity estimation) and the **Main Cache** (for KV Cache Storage). Additionally, the **Collaborative Attention Kernel** acts as a hardware-aware execution unit that distributes attention computation across GPU and CPU, and the **Cache Migration Controller** functions as the "brain" of the system, making data movement decisions based on historical access patterns.

The workflow is divided into two paths: **Inference Path** and **Maintenance Path**, as shown in Figure X. In the **Inference Path**, the process initiates by registering new requests (1) and writing prefill KV vectors to the Main Cache. Subsequently, transactions are batched with a portion of KV blocks preloaded into the GPU (2), and queries are packed for efficient processing (3). The system then performs a quick lookup to identify critical KV pairs (4), generates a visit plan for CPU-resident data (5), and collects GPU-resident blocks (6). Finally, partial attention results from both devices are merged to produce the output (7).

Parallel to inference, the **Maintenance Path** ensures system efficiency by updating the quick cache with compressed representations (8) and appending new data to the main pages (9). Concurrently, it updates block importance scores (10) and periodically exchanges pages between GPU and CPU memory (11) to promote hot blocks, ensuring optimal performance.

This design philosophy ensures efficient and scalable management of KV cache, optimizing performance for long-context LLM serving.

We manage KV entries at the granularity of blocks, adopting a page management scheme inspired by vLLM. Each block contains 64 KV pairs, managed by the Quick Access Cache Manager. This block-level granularity significantly reduces management complexity and metadata overhead while maintaining accuracy comparable to fine-grained approaches through appropriate metadata representation (e.g., Seer, InfLLM). The management system is fully vectorized, greatly reducing scheduling overhead.

## Transaction-Aware Elastic Resource Scheduling

The **Elastic Transaction Scheduler** maximizes system throughput while adhering to memory constraints by treating requests as dynamic **Transactions**. Each request encapsulates its metadata—including prompt, generated text, and a handle to its virtual KV cache—allowing for seamless migration of transaction states between prefill and decode stages. During the decode phase, the scheduler employs **Elastic Batched Decoding** to map active transactions into continuous memory "slots" using `torch.view`, which enables dynamic batch size adjustments without causing memory fragmentation or requiring expensive re-allocation. Furthermore, a **Budget Controller** proactively monitors GPU memory usage, predicting the growth of each transaction based on expected generation length and dynamically reserving GPU pages to prevent Out-Of-Memory (OOM) errors during long-sequence generation.

## Sparsity-Decoupled Dual-Track Cache Organization

To resolve the tension between high retrieval accuracy and low storage overhead, we introduce the **Dual-Track Cache Engine**, which decouples sparsity estimation from data storage. The **Pilot Cache** acts as a lightweight index for the KV cache, storing highly compressed **Block Representations** instead of full-precision vectors. By employing a lightweight neural network to project the key vectors of a block into a low-dimensional "representative embedding," the system can rapidly compute relevance scores for incoming queries. This approach allows for **Fast Top-K Selection**, which is orders of magnitude faster than full attention, to identify the **Active Set** of relevant blocks. Additionally, the management of the Pilot Cache is fully vectorized, enabling the system to estimate sparsity for large batches with negligible CPU overhead.

Complementing this, the **Primary Cache** holds the actual key-value pairs necessary for final attention computation, utilizing a **Unified Virtual Addressing** scheme that extends paged memory management across the PCIe bus. In this architecture, blocks can reside in either the GPU HBM (Hot Tier) or CPU DDR (Cold Tier). To optimize performance, the Primary Cache supports **Zero-Copy Streaming**, allowing the CPU to compute attention directly on host-resident data without the need for explicit device-to-host transfers.

## Asynchronous CPU-GPU Collaborative Attention Execution

To overcome PCIe bandwidth bottlenecks, RheoServe employs a **Collaborative Attention Kernel** that utilizes the CPU as a co-processor rather than merely a storage device. By leveraging **Head-Granularity Offloading**, the system partitions attention heads based on data locality; heads requiring data predominantly in CPU memory are offloaded to the CPU, while those with GPU-resident data are processed locally. This enables **Asynchronous Execution**, where CPU and GPU computations overlap—processing "Cold" and "Hot" blocks in parallel to effectively hide host memory access latency. Finally, partial attention outputs from both devices are merged using the **Log-Sum-Exp (LSE)** trick, ensuring numerical stability and mathematical equivalence to standard full attention.
    $$ \text{Attention}(Q, K, V) = \text{Softmax}_{\text{merge}}(\text{Score}_{GPU}, \text{Score}_{CPU}) \times [V_{GPU}; V_{CPU}] $$

## Access-Pattern Driven Adaptive Cache Migration

The **Intelligent Cache Orchestrator** serves as the central policy engine governing data movement to ensure optimal data placement. It maintains a **Cache Recorder** that tracks the access frequency and importance score of every KV block, creating a real-time **Access Heatmap**. Utilizing a **History-Aware Eviction** strategy, the orchestrator goes beyond simple LRU policies by considering the semantic importance of blocks; those frequently retrieved by the Pilot Cache are "pinned" to GPU memory even if not recently accessed. Furthermore, based on the Access Heatmap, the system proactively promotes high-probability blocks from CPU to GPU during idle cycles, effectively minimizing the "Cold Start" penalty for retrieved blocks.


# Evaluation

In this section, we evaluate the performance and accuracy of RheoServe. We aim to answer the following questions:
1.  Does RheoServe maintain model accuracy while significantly reducing memory usage?
2.  How does RheoServe compare to state-of-the-art sparse attention and offloading systems in terms of throughput and latency?
3.  What is the impact of our unified cache management and hybrid attention mechanism?

## Experimental Setup

**Hardware and Software Configuration:**
We conduct our experiments on a server equipped with NVIDIA A100 (80GB) GPUs and dual Intel Xeon Scalable processors with DDR5 memory. The system runs on Ubuntu 22.04 with PyTorch 2.1 and CUDA 12.1. We use vLLM as the baseline serving system.

**Models and Datasets:**
We evaluate RheoServe using popular open-source LLMs, including Llama-3-8B-Instruct and Llama-3-70B-Instruct. To assess long-context capabilities, we use the LongBench benchmark and custom synthetic datasets with sequence lengths ranging from 4k to 128k tokens.

## Inference Accuracy

We first verify that RheoServe does not compromise model accuracy. We compare the perplexity and downstream task performance of RheoServe against the full-attention baseline.

Results indicate that RheoServe achieves negligible accuracy loss compared to full attention, even at high sparsity levels. By dynamically retrieving critical KV blocks, RheoServe effectively mitigates the context drift issues observed in eviction-based methods like H2O and StreamingLLM.

## Overall Improvements

We compare the end-to-end performance of RheoServe with several baselines:
*   **Full Attention:** Standard vLLM implementation.
*   **Eviction-based:** H2O (keeping recent and heavy hitters).
*   **Retrieval-based:** SparseServe (offloading to CPU).

**Throughput and Latency:**
RheoServe demonstrates significant improvements in both throughput and latency. In scenarios with long context lengths (e.g., 32k), RheoServe achieves up to 2.5x higher throughput than the full-attention baseline by enabling larger batch sizes. Compared to SparseServe, our hybrid attention mechanism reduces the latency overhead caused by PCIe data transfer, resulting in a 1.5x speedup in decoding latency.

## Internal Analysis

### Latency Breakdown
We break down the latency of a single decoding step into: (1) Quick Cache lookup, (2) Data transfer, (3) GPU attention computation, and (4) CPU attention computation. The results show that the overhead of Quick Cache lookup is minimal (less than 5% of total latency). The hybrid execution effectively overlaps CPU and GPU computation, hiding the cost of data transfer.

### Offload and Retrieval Effectiveness
We analyze the hit rate of the GPU-resident cache. Our unified coordination policy ensures that over 90% of the attention mass is covered by the GPU-resident blocks, minimizing the need for expensive CPU access.

### GPU Memory Utilization
RheoServe significantly reduces the GPU memory footprint of the KV cache. By offloading non-critical blocks, we can support sequence lengths up to 4x longer than the baseline on the same hardware, or increase the batch size to improve overall system throughput.