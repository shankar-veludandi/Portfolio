# Cotten4Rec for Efficient Sequential Recommendation

## Project Summary

Cotten4Rec is an efficient sequential recommender system designed to tackle the heavy computational overhead found in Transformer-based models like BERT4Rec. While BERT4Rec achieves strong recommendation accuracy with self-attention, it suffers from significant runtime and memory costs in practice. The softmax dot-product attention in BERT4Rec requires forming large intermediate matrices with quadratic growth in sequence length, making such models inefficient to train and scale on typical hardware. Cotten4Rec addresses this inefficiency by rethinking the attention mechanism for sequential recommendation.

Instead of standard softmax attention, Cotten4Rec uses a linear cosine similarity attention mechanism fused into a single CUDA kernel. This optimized design computes attention scores via cosine similarity and aggregates values in one pass, eliminating the need to materialize the full s×s attention matrix or any large auxiliary buffers. By streaming the computations in one GPU kernel, Cotten4Rec drastically cuts down on memory footprint and kernel-launch overhead. This makes Cotten4Rec especially scalable for industry scenarios with typical short-to-moderate sequence lengths.

Empirical results show that Cotten4Rec offers substantial efficiency gains over both BERT4Rec and LinRec. It consistently uses 23% less GPU memory than either baseline across benchmark datasets. Cotten4Rec also speeds up training by 4% on Amazon Beauty and 20% on MovieLens-20M when compared to both BERT4Rec and LinRec. The performance tradeoff is that Cotten4Rec's recommendation quality (NDCG@10 and HIT@10) is within 2% of both BERT4Rec and LinRec. Overall, Cotten4Rec provides a practical, scalable alternative for sequential recommendation, delivering near SOTA accuracy with significantly lower resource requirements. It is well-suited for real-world deployments where faster training and reduced memory usage are as critical as recommendation quality.

## Publication Status

> 🚧 **Under review**  
> This work has been submitted as a Feature Article to *IEEE Intelligent Systems Magazine* and is currently under review.  
> You can read the full submitted manuscript here:  [On the Efficiency of Sequentially Aware Recommender Systems: Cotten4Rec](On_the_Efficiency_of_Sequentially_Aware_Recommender_Systems_Cotten4Rec.pdf) 📄

## Installation & Setup
These steps will get Cotten4Rec up and running, whether on a local GPU machine or via an HPC cluster.

### 1. Prerequisites
- **OS:** Linux or macOS
- **GPU:** NVIDIA
- **CUDA** CUDA 12.2 (Driver 535.104.05) installed
- **Python:** 3.9

### 2. Clone repositories:
Clone Mongaras' [Cottention_Transformer](https://github.com/gmongaras/Cottention_Transformer/tree/main) repo for the fused kernel.
```bash
git clone https://github.com/gmongaras/Cottention_Transformer.git
```

Clone *this* repo for the main code and the patched fused kernel.
```bash
git clone https://github.com/shankar-veludandi/Portfolio/cotten4rec.git
```

### 3. Copy the patched CUDA kernel:
```bash
cp Portfolio/cotten4rec/patched_combined_kernel_general.cu Cottention_Transformer/Cuda_Kernel/combined_kernel_general.cu
```

### 4. Build & install the fused CUDA kernel:
```bash
cd Cottention_Transformer
pip install -r requirements.txt
cd Cuda_Kernel
python -m pip uninstall FastAttention -y
python setup.py install
```

### 5. Create & activate your conda environment:
```bash
cd ../..
cd Portfolio/cotten4rec
conda create -n nenv python=3.9 -y
conda activate nenv
pip install -r requirements.txt
```

### 6. Run experiments with Jupyter:
Install Jupyter: `conda install -c conda-forge jupyterlab -y`

If you have a GPU on your local machine, run `jupyter lab` and open the printed URL in your browser (`http://localhost:8888/?token=<token>`)

If you're on an HPC system where you request a GPU node, you can still run notebooks in your browser by SSH port-forwarding.

1. Allocate & SSH into your GPU node (example with SLURM):
   ```bash
   salloc -N 1 --gres=gpu:1 -t 60
   ssh <node>
   ```

2. Activate your environment & launch Jupyter
   ```bash
   conda activate nenv
   jupyter notebook --no-browser --ip=0.0.0.0 --port=8888
   ```

3. Port-forward back to your local machine in a separate terminal on your local machine:
   ```bash
   ssh -L 8888:<node>:8888 <username>@<landing-pad>
   ```

4. In your local browser, go to the printed URL:
   ```bash
   http://localhost:8888/?token=<token>
   ```

Now run the Cotten4Rec, BERT4Rec, and LinRec jupyter notebooks across the ML-1M, ML-20M, and Amazon Beauty datasets in your brower!

## Acknowledgements
Based on the [BERT4Rec](https://arxiv.org/abs/1904.06690) (Sun et al., 2019) and [Cottention](https://arxiv.org/abs/2409.18747) (Mongaras et al., 2024) papers

## Contact
Shankar Veludandi - [shankar.veludandi.02@gmail.com](mailto:shankar.veludandi.02@gmail.com)
