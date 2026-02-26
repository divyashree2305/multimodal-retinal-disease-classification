# Multimodal Retinal Disease Classification

## Overview

This repository presents an experimental study on automated retinal disease classification using fundus images.

The objective was to evaluate how different modeling strategies perform under varying image quality conditions and to determine whether integrating image-quality metadata improves diagnostic reliability.

Four approaches were explored:

1. Raw Image Baseline  
2. Segmented Image Baseline  
3. Quality-Aware Preprocessing Model  
4. Multimodal Fusion Model (Best Performing)  

The multimodal model achieved the highest accuracy and demonstrated improved robustness and interpretability.

---

## Dataset

**FIVES: A Fundus Image Dataset for AI-based Vessel Segmentation**

- Classes:
  - A – Artery-related pathology  
  - D – Diabetic Retinopathy  
  - G – Glaucoma  
  - N – Normal  
- 80% Training / 20% Testing split  
- Includes image-quality metadata (Illumination, Blur, Contrast)

Dataset Link:  
https://figshare.com/articles/figure/FIVES_A_Fundus_Image_Dataset_for_AI-based_Vessel_Segmentation/19688169/1


---

## Models and Results

| Model Variant | Description | Accuracy |
|---------------|------------|----------|
| Segmented Image Model | Vessel maps only | 48.5% |
| Raw Image Baseline | Image-only CNN | 54% |
| Quality-Aware Model | Metadata-driven preprocessing | 25% |
| Multimodal Fusion Model | CNN + Metadata (MLP) | 81% |

### Key Observations

- Image-only models struggled under inconsistent lighting and blur conditions.
- Preprocessing alone was insufficient and sometimes degraded performance.
- Integrating metadata significantly improved stability and generalization.
- The multimodal fusion model produced balanced precision, recall, and F1 scores across classes.

---

## Multimodal Architecture

- CNN Backbone (EfficientNet-B0) for visual feature extraction  
- Metadata Branch (MLP) for illumination, blur, and contrast encoding  
- Feature-level concatenation of image and metadata embeddings  
- Fully connected classification head with softmax output  

---

## Evaluation

All experiments were conducted under identical dataset splits, input resolution (224×224), and transfer learning initialization.

Metrics used:

- Accuracy  
- Precision  
- Recall  
- F1-score  
- Macro-AUC  
- Confusion Matrix  
- ROC Curves  

The multimodal model achieved 81% accuracy with strong and balanced performance across all four disease classes.

Grad-CAM was used to visualize model attention. The multimodal model showed improved focus on clinically relevant retinal regions and reduced sensitivity to lighting artifacts.

---

## Tech Stack

- Python  
- PyTorch  
- Torchvision  
- scikit-learn  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Grad-CAM  

---

## Installation

Install dependencies using:

```bash
pip install -r requirements.txt
```
## Important Usage Note

These notebooks will not run directly after cloning.

### To execute the code:

1. Download the FIVES dataset from the link provided above.
2. Place the dataset in a local directory.
3. Update dataset path variables inside the notebooks.
4. Install dependencies from `requirements.txt`.
5. Adjust local model weight paths if required.

This repository represents an experimental research workflow and may require minor environment configuration before execution.

---

## Conclusion

This study demonstrates that incorporating image-quality metadata into CNN-based retinal classification enhances:

- Accuracy  
- Robustness  
- Interpretability  

Multimodal fusion proved to be the most reliable paradigm for retinal disease prediction under varying image quality conditions.
