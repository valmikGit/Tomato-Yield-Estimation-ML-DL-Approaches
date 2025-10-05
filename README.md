# Tomato-Yield-Estimation-ML-DL-Approaches

```mermaid
flowchart TD

    A[Start: Input Data] --> B[CSV + Image Folder]
    B --> C[Read CSV: Image Name, Weight, View, Tomato ID]
    C --> D[HSV Conversion + Red Contour Detection]
    D --> E[Extract Bounding Box i.e. Ground Truth]
    E --> F[Save bbox images for visualization]

    F --> G[Dataset Split 80:10:10 → Train/Val/Test]
    G --> H[Resize all images to 224 * 224]
    H --> I[Scale RGB pixels to [0,1]]
    I --> J[Compute subset mean/std]
    J --> K[Normalize each subset with mean/std]
    K --> L[Random Horizontal Flip only for training]

    L --> M[Faster R-CNN ResNet-50-FPN backbone]
    M --> N[Train detector with Adam optimizer]
    N --> O[Use trained detector to predict bboxes on all splits]
    O --> P[Crop detected regions]
    P --> Q[Save detected crops for next stage]

    Q --> R[Feature Extraction per crop]
    R --> R1[MobileNetV3 Backbone → Global pooled feature → α scalar]
    R --> R2[Compute Image Area (H×W)]
    R --> R3[Compute Aspect Ratio (W/H)]
    R --> R4[Compute Avg Pixel Intensity]
    R1 --> S[Concatenate 4 features]
    R2 --> S
    R3 --> S
    R4 --> S

    S --> T[3-layer Fully Connected Network]
    T --> T1[Layer1: 4→64 ReLU activation]
    T1 --> T2[Layer2: 64→32 ReLU activation]
    T2 --> T3[Layer3: 32→1 Regression output]

    T3 --> U[Train with MSE Loss + Adam optimizer]
    U --> V[Bayesian Optimization after each epoch]
    V --> V1[Gaussian Process surrogate model]
    V1 --> V2[Expected Improvement acquisition]
    V2 --> W[5 BO iterations per epoch: 3 epochs total]
    
    W --> X[Final model evaluation on Test Set]
    X --> X1[Compute MSE, MAE, R²]
    X1 --> Y[Save models: Faster R-CNN, MLP, normalization stats]
    Y --> Z[Inference Function: Single image prediction]

    Z --> End[Pipeline Finished]
```
