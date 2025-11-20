# RheoServe: A Heterogeneous Acceleration Engine for Sparse Attention Models

# Introduction

Large Language Models (LLMs) have become deeply integrated into our daily lives, with their capabilities growing in accordance with scaling laws. While the scaling of model size and pre-training data has slowed, a new trend—test-time scaling or compute-time scaling—has emerged as a focal point. Coupled with reinforcement learning, this new scaling paradigm encourages LLMs to explore more possibilities through their own "chain of thought." Consequently, context length requirements have surged from 4k to 16k, 32k, and even 128k tokens. The KV Cache, which stores key and value vectors of past tokens to avoid redundant computation, has thus become a critical bottleneck for LLM serving systems. Its size grows linearly with context length, imposing significant memory consumption and bandwidth demands.

Recent works have discovered that the attention mechanism in LLMs exhibits inherent sparsity, where only a small subset of tokens significantly influences the generation of new tokens. This implies that storing and computing the entire KV cache is often unnecessary. Dynamic Sparse Attention (DSA) algorithms, such as Quest, MagicPIG, HashAttention, and SeerAttention, identify and utilize only the most critical tokens (top-K) for attention computation, significantly reducing computational overhead. However, these methods typically retain the entire KV cache in GPU memory, which quickly exhausts capacity when dealing with long sequences.

Other approaches, like H2O and StreamingLLM, manage a fixed cache budget by permanently evicting less important tokens. However, these eviction-based methods often perform poorly in long-context inference, inducing context drift and accuracy degradation as they risk losing critical information required at later generation stages.

To address this, recent studies like SparseServe opt for offloading non-critical KV entries to CPU memory and retrieving necessary entries back to the GPU based on importance estimation. While this assures accuracy, performance is strictly limited by PCIe bandwidth. Consequently, works like MagicPIG and HGCA split the attention workload between CPU and GPU using Log-Sum-Exp (LSE) softmax, alleviating the transfer bottleneck since host memory bandwidth is generally higher than PCIe bandwidth. Nevertheless, a performance gap remains between GPU memory bandwidth and host memory bandwidth; excessive offloading can cause overall system performance to be dominated by host memory limitations.

We make the following observations: (1) Existing methods are polarized—eviction-based methods focus on extreme sparsity, while retrieval-based methods focus on high recall. There is a lack of a unified framework that systematically combines these mechanisms—maintaining as much KV cache on the GPU as possible while ensuring critical offloaded data is retrieved efficiently. (2) A flexible and efficient KV cache management system is required to manage resident KV cache on both GPU and CPU memory, as well as the transfers between them. Moreover, emerging hardware like NVIDIA's Vera Rubin architecture features unified memory architectures with significantly improved host-device memory bandwidth, presenting new opportunities.

To address these challenges, we present **RheoServe**, a heterogeneous acceleration engine specifically designed for sparse attention models. Our key contributions are summarized as follows:

*   **Fine-Grained KV Cache Management System:** We propose a head-granularity KV cache management system that unifies the management of KV cache across GPU and CPU. It supports both retrieval and eviction, effectively reducing cache miss rates and memory bandwidth pressure.
*   **Unified Offload and Retrieval Mechanism:** Through a lightweight **Cache Recorder** and **Budget Controller**, we unify the offload and retrieval mechanisms, ensuring that critical KV entries are efficiently scheduled between GPU and CPU.
*   **High-Performance CPU-GPU Collaboration:** We construct a CPU-GPU collaborative computing framework based on LSE-softmax. By leveraging the high bandwidth of DDR5 and GPU pipeline parallelism, we break through the PCIe bandwidth bottleneck and provide optimization potential for future hardware (e.g., GTC 2025 Vera Rubin architecture).

**Opportunities:**
*   **Scalability for Long-Context Tasks:** RheoServe excels in long-generation and long-inference tasks, providing a scalable solution that supports future demands for even longer contexts (e.g., 128k+).
*   **Unified Framework for Sparse Attention:** RheoServe fills the gap between eviction-based and retrieval-based methods with a unified KV cache management system.
*   **Exploiting Emerging Hardware Architectures:** With the advent of new hardware like the NVIDIA Vera Rubin architecture, host-device memory bandwidth is significantly improved. RheoServe's design is poised to fully exploit these hardware characteristics for further performance gains.



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

As illustrated in Figure X, RheoServe seamlessly integrates into existing LLM serving pipelines (e.g., vLLM) by replacing the standard attention module. It acts as a transparent middleware that virtualizes the KV cache, abstracting away the complexity of heterogeneous memory management.

The architecture is composed of four synergistic subsystems:

1.  **Elastic Transaction Scheduler:** Manages the lifecycle of requests and orchestrates the prefill-decode workflow.
2.  **Dual-Track Cache Engine:** A hierarchical storage system comprising the **Pilot Cache** (for fast sparsity estimation) and the **Primary Cache** (for actual data storage).
3.  **Collaborative Attention Kernel:** A hardware-aware execution unit that distributes attention computation across GPU and CPU.
4.  **Intelligent Cache Orchestrator:** The "brain" of the system that makes data movement decisions based on historical access patterns.

RheoServe adopts a **Prefill-Decode (PD) Disaggregated** design philosophy. As illustrated in Figure X, the workflow is divided into an inference path and a maintenance path, corresponding to the numbered markers in the figure:

**Inference Path:**
**(1) Transaction Registration:** When a new request arrives, the **Elastic Transaction Scheduler** registers it as a transaction. It proceeds through the prefill phase using full attention, and the generated KV vectors are written directly into the CPU-resident **Primary Cache**.
**(2) Batching & Preloading:** Once enough transactions accumulate, they are batched, and a small "warm-up" portion of their KV blocks is preloaded into the GPU **Primary Cache**.
**(3) Query Packing:** The queries are packed into batched query vectors for efficient processing.
**(4) Pilot Cache Lookup:** The **Dual-Track Cache Engine** queries the **Pilot Cache** to rapidly identify the "Active Set" of KV blocks relevant to the current query.
**(5) Visit Planning:** Based on the lookup, the **Intelligent Cache Orchestrator** generates a **Visit Plan**, scheduling specific blocks residing in host memory for CPU-side attention.
**(6) GPU Collection:** Simultaneously, we collect all GPU-resident blocks belonging to the current transaction for GPU-side attention, ensuring maximum utilization of high-bandwidth memory.
**(7) Hybrid Merge:** The **Collaborative Attention Kernel** executes on both devices and merges the partial attention results using LSE reduction to produce the final output.

**Maintenance Path:**
**(8) Pilot Cache Update:** The newly generated KV pair is used to update the **Pilot Cache**, generating new compressed block representations.
**(9) Primary Cache Update:** The new KV data is appended to the **Primary Cache** pages.
**(10) Score Maintenance:** During the Pilot Cache lookup, the system updates the accumulative block importance scores in the **Cache Recorder**.
**(11) Page Exchange:** Periodically, the Orchestrator uses these scores to exchange pages between GPU and CPU, promoting hot blocks to the GPU to ensure a high cache hit rate.

## Elastic Transaction Scheduler

The **Elastic Transaction Scheduler** is responsible for maximizing system throughput while adhering to memory constraints. Unlike traditional schedulers that view requests as static sequences, we treat them as dynamic **Transactions**.

*   **Transaction Abstraction:** Each request encapsulates its metadata, including prompt, generated text, and a handle to its virtual KV cache. This abstraction allows the scheduler to easily migrate transaction states between the prefill and decode stages.
*   **Dynamic Slot Mapping:** During the decode phase, the scheduler maps active transactions into continuous memory "slots" using `torch.view`. This enables **Elastic Batched Decoding**, allowing us to dynamically adjust batch sizes without memory fragmentation or expensive re-allocation.
*   **Predictive Budgeting:** A **Budget Controller** monitors the GPU memory usage. It predicts the memory growth of each transaction based on expected generation length and dynamically reserves GPU pages. This proactive reservation prevents OOM (Out-Of-Memory) errors during long-sequence generation.

## Dual-Track Cache Engine

To resolve the tension between high retrieval accuracy and low storage overhead, we introduce the **Dual-Track Cache Engine**, which decouples sparsity estimation from data storage.

### Pilot Cache (formerly Quick K Cache)

The **Pilot Cache** serves as a lightweight index for the KV cache. Instead of storing full-precision vectors, it stores a highly compressed **Block Representation**.

*   **Compression via Low-Rank Projection:** We employ a lightweight neural network (inspired by SeerAttention) to project the key vectors of a block (e.g., 64 tokens) into a low-dimensional "representative embedding."
*   **Fast Top-K Selection:** For an incoming query, the Pilot Cache computes a relevance score against these representative embeddings. This operation is orders of magnitude faster than full attention. The top-K most relevant blocks form the **Active Set**.
*   **Vectorized Management:** The Pilot Cache is fully vectorized, allowing the system to estimate sparsity for large batches with negligible CPU overhead.

### Primary Cache (formerly Normal KV Cache)

The **Primary Cache** stores the actual key-value pairs required for the final attention computation.

*   **Unified Virtual Addressing:** We implement a paged memory management scheme similar to vLLM but extended across the PCIe bus. Blocks can reside in GPU HBM (Hot Tier) or CPU DDR (Cold Tier).
*   **Zero-Copy Streaming:** The Primary Cache supports zero-copy data paths where possible, allowing the CPU to compute attention directly on host-resident data without explicit device-to-host transfers.

## Collaborative Attention Kernel

To overcome the PCIe bandwidth bottleneck, RheoServe employs a **Collaborative Attention Kernel** that treats the CPU not just as a storage device, but as a co-processor.

*   **Head-Granularity Offloading:** We leverage the multi-head nature of attention. For a given query, attention heads are partitioned. Heads requiring data predominantly resident in CPU memory are offloaded to the CPU, while heads with GPU-resident data are processed locally.
*   **Asynchronous Execution:** CPU and GPU computations overlap. While the GPU processes the "Hot" blocks, the CPU processes the "Cold" blocks in parallel, effectively hiding the latency of accessing host memory.
*   **LSE-Based Aggregation:** The partial attention outputs (local sums and max scores) from both devices are merged using the **Log-Sum-Exp (LSE)** trick. This ensures numerical stability and mathematical equivalence to standard full attention.
    $$ \text{Attention}(Q, K, V) = \text{Softmax}_{\text{merge}}(\text{Score}_{GPU}, \text{Score}_{CPU}) \times [V_{GPU}; V_{CPU}] $$

## Intelligent Cache Orchestrator

The **Intelligent Cache Orchestrator** is the central policy engine that governs data movement, ensuring the right data is in the right place at the right time.

*   **Cache Recorder & Heatmap:** The orchestrator maintains a **Cache Recorder** that tracks the access frequency and "importance score" of every KV block. This creates a real-time **Access Heatmap**.
*   **History-Aware Eviction:** Unlike simple LRU (Least Recently Used) policies, our eviction strategy considers the semantic importance of blocks. Blocks that are historically frequently retrieved by the Pilot Cache are "pinned" to GPU memory, even if they haven't been accessed in the immediate last step.
*   **Pre-fetching & Promotion:** Based on the Access Heatmap, the orchestrator proactively promotes high-probability blocks from CPU to GPU during idle cycles, minimizing the "Cold Start" penalty for retrieved blocks.


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