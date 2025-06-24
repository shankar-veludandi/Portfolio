# Cotten4Rec for Efficient Sequential Recommendation

## Project Summary

Cotten4Rec is an efficient sequential recommender system designed to tackle the heavy computational overhead found in Transformer-based models like BERT4Rec. While BERT4Rec achieves strong recommendation accuracy with self-attention, it suffers from significant runtime and memory costs in practice. The softmax dot-product attention in BERT4Rec requires forming large intermediate matrices with quadratic growth in sequence length, making such models inefficient to train and scale on typical hardware. Cotten4Rec addresses this inefficiency by rethinking the attention mechanism for sequential recommendation.

Instead of standard softmax attention, Cotten4Rec uses a linear cosine similarity attention mechanism fused into a single CUDA kernel. This optimized design computes attention scores via cosine similarity and aggregates values in one pass, eliminating the need to materialize the full s×s attention matrix or any large auxiliary buffers. By streaming the computations in one GPU kernel, Cotten4Rec drastically cuts down on memory footprint and kernel-launch overhead. This makes Cotten4Rec especially scalable for industry scenarios with typical short-to-moderate sequence lengths.

Empirical results show that Cotten4Rec offers substantial efficiency gains over both BERT4Rec and LinRec. It consistently uses 23% less GPU memory than either baseline across benchmark datasets. Cotten4Rec also speeds up training by 4% on Amazon Beauty and 20% on MovieLens-20M when compared to both BERT4Rec and LinRec. The performance tradeoff is that Cotten4Rec's recommendation quality (NDCG@10 and HIT@10) is within 2% of both BERT4Rec and LinRec. Overall, Cotten4Rec provides a practical, scalable alternative for sequential recommendation, delivering near SOTA accuracy with significantly lower resource requirements. It is well-suited for real-world deployments where faster training and reduced memory usage are as critical as recommendation quality.

## Prequisites & Installation

### 1. System Requirements
- **OS:** Linux or macOS
- **GPU & CUDA:** NVIDIA GPU with CUDA 12.2 installed
- **Python:** 3.9

### 2. Clone Repositories 
```bash
# Clone Mongaras' Cottention_Transformer repo for fused kernel
git clone https://github.com/gmongaras/Cottention_Transformer.git
# Clone this repo for main code and patched kernel
git clone https://github.com/shankar-veludandi/Portfolio/cotten4rec.git
```

### 3. Copy the Patched CUDA kernel 
```bash
cp Portfolio/cotten4rec/patched_combined_kernel_general.cu \
   Cottention_Transformer/Cuda_Kernel/combined_kernel_general.cu
```

## Acknowledgements
Inspired by the BERT4Rec (Sun et al., 2019) and Cottention (Mongaras et al., 2024) papers

## Contact
Shankar Veludandi - [shankar.veludandi.02@gmail.com](mailto:shankar.veludandi.02@gmail.com)
