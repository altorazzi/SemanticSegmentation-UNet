# Semantic Segmentation of Mars Terrain via U-Net & ASPP

## Project Overview
Developed a Deep Learning pipeline to perform **semantic segmentation** on Martian terrain imagery. The goal was to classify every pixel of 64x128 grayscale images into 5 distinct terrain classes (Background, Soil, Bedrock, Sand, Big Rock) to assist autonomous rover navigation.

The final solution employs an **Ensemble of Custom U-Net Architectures**, utilizing **Atrous Spatial Pyramid Pooling (ASPP)** to capture multi-scale context, achieving a mean IoU of **0.73** on the test set.

## Data Engineering & Preprocessing
Real-world data is noisy. A significant portion of the performance gain came from rigorous data cleaning rather than model tuning.

* **Anomaly Detection:** Utilized **PCA and Mahalanobis distance** to identify outliers
* **Manual Label Refinement:** Identified images with incorrect ground truth (specifically for the underrepresented "Big Rock" class). Corrected masks manually using **MATLAB's ROI tools** to ensure high-quality supervision.
* **Addressing Imbalance:** Applied targeted geometric augmentation exclusively to the underrepresented classes to balance the distribution.

## 🧠 Model Architecture

We implemented and ensembled two distinct architectures to balance localization precision with contextual understanding.

### Model 1: U-Net with Resize-Convolution
A custom encoder-decoder structure designed to avoid common segmentation pitfalls.
* **Backbone:** Standard convolutional encoder with He-Normal initialization.
* **Artifact Suppression:** Replaced standard Transpose Convolutions with **Upsampling (Nearest Neighbor) + Convolution**. This architectural choice eliminates the "checkerboard artifacts" often seen in deconvolutional networks.
* **Regularization:** 256-filter bottleneck with L2 regularization to prevent overfitting.

### Model 2: U-Net + ASPP (Atrous Spatial Pyramid Pooling)
Inspired by *DeepLab*, this model integrates an **ASPP block** at the bottleneck.
* **Mechanism:** Uses parallel dilated convolutions (rates: 1, 6, 12, 18) to capture features at multiple resolutions.
* **Benefit:** Allows the network to "see" large features (like large rock formations) without losing resolution, significantly improving the segmentation of class 4 (Big Rock).

### 🏆 Ensemble Strategy
Final predictions are generated via **Soft Voting**:
$$P_{final} = \text{argmax}(P_{Model1} + P_{Model2})$$
This reduced variance and smoothed out decision boundaries, yielding a robust final mask.

## 📉 Training Strategy

* **Loss Function:** **Categorical Focal Crossentropy**. This was critical for handling class imbalance, penalizing hard-to-classify examples more than easy ones.
* **Custom Class Weights:** `[0, 4, 5, 5.5, 50]`. We explicitly zeroed out the "Background" class to force the model to focus purely on terrain features.
* **Optimization:** Adam Optimizer with **ReduceLROnPlateau** (dynamic learning rate decay) and **EarlyStopping**.## 🧠 Model Architecture

We implemented and ensembled two distinct architectures to balance localization precision with contextual understanding.

### Model 1: U-Net with Resize-Convolution
A custom encoder-decoder structure designed to avoid common segmentation pitfalls.
* **Backbone:** Standard convolutional encoder with He-Normal initialization.
* **Artifact Suppression:** Replaced standard Transpose Convolutions with **Upsampling (Nearest Neighbor) + Convolution**. This architectural choice eliminates the "checkerboard artifacts" often seen in deconvolutional networks.
* **Regularization:** 256-filter bottleneck with L2 regularization to prevent overfitting.

### Model 2: U-Net + ASPP (Atrous Spatial Pyramid Pooling)
Inspired by *DeepLab*, this model integrates an **ASPP block** at the bottleneck.
* **Mechanism:** Uses parallel dilated convolutions (rates: 1, 6, 12, 18) to capture features at multiple resolutions.
* **Benefit:** Allows the network to "see" large features without losing resolution, significantly improving the segmentation of class 4 (Big Rock).

### Ensemble Strategy
Final predictions are generated via **Soft Voting**:
$$P_{final} = \text{argmax}(P_{Model1} + P_{Model2})$$
This reduced variance, yielding robust final results

## Training Strategy

* **Loss Function:** **Categorical Focal Crossentropy**. This was critical for handling class imbalance, penalizing hard-to-classify examples more than easy ones.
* **Custom Class Weights:** `[0, 4, 5, 5.5, 50]`. We explicitly zeroed out the "Background" class to force the model to focus purely on terrain features.
* **Optimization:** Adam Optimizer with dynamic learning rate decay and **EarlyStopping**.

---

## Results
- Mean IoU improved from **0.49 → 0.65** after tuning class weights.  
- Final **ensemble model reached 0.73 mean IoU** on the test set.  
- Biggest gains came from:
  - Correcting labels,  
  - Weighting background class = 0,  
  - ASPP-based second model + ensembling.  

---

## Repository Structure
- `Deepmindset_Challenge2_model_1.ipynb` → Baseline U-Net model  
- `Deepmindset_Challenge2_PCA_model_2.ipynb` → ASPP U-Net + Ensembling  
- `Report.pdf` → Full technical report (data prep, modeling details, discussion, references)

---

## How to Run
1. Clone the repo:
   ```bash
   git clone [https://github.com/altorazzi/SemanticSegmentation-UNet.git](https://github.com/altorazzi/SemanticSegmentation-UNet.git)
