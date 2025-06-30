# Invasive Insect Species Classifier

## Problem Statement  
Invasive insect species cause an estimated \$30 B in annual U.S. damages and threaten ecosystem health by out-competing native species. Early identification from camera-trap or citizen-science imagery is critical for prioritizing interventions. Here, we build an image classifier to distinguish invasive vs. native lookalike insect species.

---

## Approach  

1. **Data**  
   - Two CSVs of image URLs:  
     - `northeast_invasive_insects.csv` (69 K images of target invasive species)  
     - `northeast_native_insects.csv` (34 K images of morphologically similar native species)  
   - Filtered to the northeastern United States.

2. **Model**  
   - **BioCLIP**’s `CustomLabelsClassifier` to embed and classify images by species.  
   - Species list passed directly to the classifier—no additional fine-tuning required.

3. **Pipeline**  
   - Download each URL → preprocess with Pillow → feed into `CustomLabelsClassifier`.  
   - Record predicted vs. true labels, then compute metrics.

---

## Results  

Baseline Performance Metrics (on held-out test set):

| Metric     | Score  |
|-----------:|-------:|
| Accuracy   |  95.14 % |
| Precision  |  95.16 % |
| Recall     |  95.14 % |
| F1 Score   |  95.13 % |

---

## Usage  

1. Download northeast_invasive_insects.csv & northeast_native_insects.csv
2. Run invasive_insects_classifier.ipynb
