# vLLM Talent Pool - Engineering Highlights

## 🧠 Multi-Turn Bi-Level Optimization (HRL)
- **Hierarchical In-Context RL (HICRL)**: Architected a bi-level optimization framework where a **High-Level (HL)** context generator and a **Low-Level (LL)** task solver are optimized via alternating RL loops.
- **Context Utilization Engineering**: Developed a KL-Maximization objective to force "forced dependency" on generated demonstrations, ensuring the LL utilizes the HL feedback rather than bypassing it.
- **Multi-Turn Trajectory Management**: Engineered complex state-tracking for 3-turn RL loops (Initial Solve → Feedback → Refinement) with specialized advantage estimation.

## ⚡ High-Performance vLLM & Distributed Systems
- **vLLM-Native Rollouts**: Deeply integrated vLLM as the backbone for high-throughput RL rollouts. Leveraged **PagedAttention** and **Batching** to handle the compounding sequence lengths of hierarchical prompts (Context + Reasoning Trace).
- **Asynchronous RL Serving**: Orchestrated asynchronous generation across distributed **Ray** actors, minimizing latency in the bi-level sampling process.
- **Inference-Aware Reward Shaping**: Implemented real-time reward calculation using vLLM-generated log-probabilities (logp) to track and optimize model confidence and context dependency.
- **Backend Interop**: Managed weight synchronization and sharding between training backends (**Megatron-LM**, **FSDP**) and vLLM inference engines for zero-redundancy model updates.



