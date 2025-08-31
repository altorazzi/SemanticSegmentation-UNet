# Semantic Segmentation of Mars Terrain (DeepMindset Challenge)

## Project Overview
This project was developed as part of the Artificial Neural Networks course (Politecnico di Milano).  
The task was to design and train deep learning models for **semantic segmentation of Mars terrain images**.  
The dataset contained **2,615 grayscale images (64x128 px)** with corresponding masks, classifying each pixel into:

- **0**: Background  
- **1**: Soil  
- **2**: Bedrock  
- **3**: Sand  
- **4**: Big Rock  

Our goal was to produce segmentation masks as close as possible to the ground truth.

---

## Methodology
1. **Data Preprocessing**
   - Outlier detection with PCA & Mahalanobis distance.
   - Manual label refinement using MATLAB ROI tools.
   - Addressed **class imbalance** with augmentation (esp. underrepresented class 4: Big Rock).

2. **Data Augmentation**
   - Normalization to [0, 1].  
   - Random flips for training data.  
   - Careful alignment between images and masks.  

3. **Modeling Approach**
   - **Model 1**: U-Net encoder–decoder with skip connections and He-normal initialization.  
   - **Model 2**: U-Net + **ASPP (Atrous Spatial Pyramid Pooling)** at the bottleneck.  
   - **Ensembling**: Combined probabilities from both models to improve robustness.

4. **Training Strategy**
   - Optimizer: Adam  
   - Loss: **Categorical Focal Crossentropy with custom class weights**  
   - EarlyStopping + ReduceLROnPlateau for stability  

---

## Results
- Mean IoU improved from **0.49 → 0.65** after tuning class weights.  
- Final **ensemble model reached 0.73 mIoU** on the test set.  
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
   git clone https://github.com/altorazzi/SemanticSegmentation-UNet.git
   cd SemanticSegmentation-UNet
