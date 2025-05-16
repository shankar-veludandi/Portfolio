# Advanced Sequence Modeling & Embeddings

This notebook explores three distinct sequence‐modeling tasks using neural nets:

1. **Character‐Level Language Modeling** (RNN vs. LSTM)  
2. **Time‐Series Forecasting** of daily minimum temperatures (RNN vs. LSTM vs. GRU)  
3. **Contextual Word Embeddings & Similarity** with BERT

---

## 1. Character‐Level Language Modeling

**Problem:**  
Generate “Tiny Shakespeare”–style text one character at a time.

**Approach:**  
- **Data:** Raw text from _tinyshakespeare.txt_  
- **Models:**  
  - Vanilla RNN (hidden_size=128, n_layers=2)  
  - LSTM  (hidden_size=128, n_layers=2)  
- **Training:**  
  - Trained separately for 5, 50, and 500 epochs  
  - Saved checkpoints (`.pt`) and sampled text via `generate.py`

**Results:**  
- **5 epochs:** Mostly gibberish, scant structure  
- **50 epochs:** RNN learns local patterns; LSTM yields more coherent snippets  
- **500 epochs:** Both produce fluent Shakespeare-like phrases; LSTM captures longer dependencies with fewer nonsensical fragments

---

## 2. Time‐Series Forecasting of Daily Minimum Temperatures

**Problem:**  
Predict next‐day minimum temperature in Melbourne from a univariate time series.

**Approach:**  
- **Data:** 1981–1990 daily minima  
- **Dataset:**  
  - Sliding windows of 30 days → next‐day label  
  - Train/Val/Test splits  
- **Models:**  
  - Vanilla RNN (tanh)  
  - LSTM  
  - GRU  
- **Training:**  
  - 10 epochs, batch size 32, MSE loss, Adam optimizer  

**Results:**  
| Model         | Test RMSE |
|--------------:|----------:|
| Vanilla RNN   | 3.35      |
| LSTM          | 2.99      |
| GRU           | 3.62      |

> Both LSTM and GRU outperform vanilla RNN; LSTM achieves the lowest validation and test RMSE by capturing longer‐range temporal dependencies.

---

## 3. Contextual Word Embeddings & Similarity (BERT)

**Problem:**  
Extract and visualize BERT embeddings; define a custom “dissimilarity” metric combining cosine similarity and Euclidean distance.

**Approach:**  
1. **Embedding Extraction:**  
   - Load `bert-base-uncased` model & tokenizer  
   - For each word, take the last hidden state’s mean across tokens  
2. **Visualization:**  
   - **PCA** → 2D scatter of selected words  
3. **Similarity Metrics:**  
   - **Cosine similarity** (semantic orientation)  
   - **Euclidean distance** (absolute magnitude)  
   - **Custom dissimilarity**:  
     ```python
     D = α·(1 – cos_sim) + (1–α)·(norm_dist)
     ```
4. **Plots:**  
   - Bar charts ranking nearest neighbors by each metric  
   - Heatmaps comparing cosine vs. custom across word pairs  

**Results:**  
- **PCA 2D** shows logical clusters (e.g. “king”↔“queen”, “apple” near “orange”)  
- **Cosine vs. Custom:**  
  - Cosine favors semantic siblings (high cat↔lion)  
  - Custom highlights both semantic closeness and magnitude differences  
  - Heatmaps reveal nuanced contrasts between metrics
