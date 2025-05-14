# Wildlife Classifier Pipeline

**Goal:**  
Identify zebra and giraffe presence in camera-trap images (multi-label classification).

**Key Steps:**  
1. **Data Preparation**:  
   - Stratified 70/15/15 split on {neither, zebra-only, giraffe-only, both}.  
   - Custom PyTorch `CameraTrapsDataset` with resizing, normalization, and augmentations.

2. **Model Training**:  
   - Backbone: Pretrained ResNet50 (frozen); Classifier head: 2 output units for independent binary predictions.  
   - Loss: `BCEWithLogitsLoss`; Optimizer: AdamW with cosine learning-rate schedule.  
   - Monitored validation accuracy to save the best model.

3. **Evaluation**:  
   - Metrics: per-label accuracy, precision, recall, F1; multi-label subset accuracy.  
   - Visual: confusion heatmap for combined four-label outcomes.

**Results:**  
- **Final Test Accuracy:** 99.61%  
- **Test Loss (best model):** 0.0248  
- **Per-label Metrics (best model):**  
  - Giraffe: 99.44% accuracy (F1: 0.99)  
  - Zebra:   99.54% accuracy (F1: 0.99)  
