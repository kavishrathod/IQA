# Image Quality Assessment (IQA) for Exposure Fusion

This repository contains the implementation of a No-Reference Image Quality Assessment (IQA) model specifically designed for exposure fusion images. The project proposes a novel CNN-based framework that predicts image quality scores without any reference image, combining both quantitative analysis and human perception-based learning.

## 📋 Overview

Exposure fusion is a popular method for generating high dynamic range (HDR) images by combining multiple low dynamic range (LDR) images captured at varying exposures. However, assessing the quality of these fused images is challenging due to the lack of reference images and subjective nature of perception.

This project addresses that challenge by introducing a no-reference IQA method that:
- Computes key quantitative metrics such as entropy, contrast, and saturation.
- Collects human perception data via a web-based rating system.
- Trains a Convolutional Neural Network (CNN) to predict image quality based on human visual feedback.

## 🚀 Features
- **No-reference evaluation** — works without ground truth or reference images.
- **Quantitative metrics computation** — entropy, local entropy, contrast, local contrast, saturation.
- **Human perception integration** — collects and uses real user ratings to align machine predictions with visual quality.
- **CNN-based prediction model** — accurately estimates perceptual image quality.
- **Final scoring system** — fuses quantitative and predicted metrics for a holistic quality score.

## 🧩 System Architecture

The proposed IQA pipeline consists of:

### 1. Quantitative Analysis Module
- Extracts entropy, local entropy, contrast, local contrast, and saturation from the fused image.

### 2. Human Perception Data Collection
- Uses a **React + Firebase** web platform to gather quality ratings (contrast, sharpness, saturation) from participants.

### 3. CNN Model
- A 5-output CNN predicts image scores for each quality parameter based on visual features:
  - 3×3 convolution filters with ReLU activation
  - Max pooling (2×2)
  - Dropout regularization (50%)
  - 5 neurons in the output layer (corresponding to 5 image parameters)

### 4. Score Fusion Module
- Combines calculated and predicted scores using weighted similarity:

Final Score = W1 * E + W2 * LE + W3 * C + W4 * LC + W5 * S

Where:
- `E` = Entropy
- `LE` = Local Entropy
- `C` = Contrast
- `LC` = Local Contrast
- `S` = Saturation
- `W1, W2, W3, W4, W5` are the weights.

## 🧠 Technologies Used

| Category             | Tools / Frameworks           |
|----------------------|------------------------------|
| **Programming**       | Python                       |
| **Deep Learning**     | TensorFlow / Keras           |
| **Web Development**   | React, Firebase              |
| **Data Collection**   | Octoparse 8 (Web Scraping)   |
| **Image Processing**  | OpenCV, NumPy                |

## 📊 Dataset

- **Image Source**: Scraped from multiple online repositories using Octoparse 8.
- **Total Images**: 1000+ exposure-fused images.
- **Annotations**: Human perception scores collected via an online survey.
- **Attributes Rated**: Contrast, Saturation, Entropy (clarity and naturalness).

## 📈 Results

The proposed No-Reference Image Quality Assessment (IQA) model has been evaluated on a series of exposure-fused images, and the results demonstrate its ability to effectively align with human perception. The following table presents the performance of the proposed system compared to existing methods:

| Method                                | Image                   | Final Score | Rank |
|---------------------------------------|-------------------------|-------------|------|
| **Mertens et al. (2007)**             | Eiffel Tower Sequence    | 3.19        | 🥇   |
| **Vanmali et al. (2013)**             | Eiffel Tower Sequence    | 3.11        | 🥈   |
| **Kotwal et al. (2011)**              | Eiffel Tower Sequence    | 2.92        | 🥉   |

These results demonstrate that our system is competitive with existing state-of-the-art methods and provides reliable, perceptually accurate quality assessments for exposure-fused images.

## 🔮 Future Work

The current framework serves as a robust starting point for no-reference image quality assessment in exposure fusion. However, there are several opportunities for further improvement and extension of the model:

1. **Detail Preservation Analysis**: 
   - One of the key aspects of exposure fusion is the retention of detail from multiple input images. Future work could include integrating detail preservation metrics to assess how well the fused image retains high-frequency information and structure from the original images.
  
2. **Extension to Other Fusion Techniques**:
   - The current model is focused on exposure fusion, but it could be extended to other image fusion techniques, such as infrared-visible fusion or multi-modal fusion in medical imaging. Adapting the model to these domains would expand its applicability.

3. **Real-Time IQA Scoring**:
   - A public web demo could be developed to enable real-time IQA scoring for exposure-fused images. This would allow users to upload their own images and receive an instant perceptual quality score, along with visual feedback on the key quality parameters.

4. **Integration with Advanced Deep Learning Models**:
   - Future research could explore the use of more advanced deep learning models, such as transformer-based networks or self-supervised learning approaches, to further enhance the accuracy of perceptual quality predictions.

5. **Incorporating User Feedback for Model Improvement**:
   - The model could be made more adaptive by continuously incorporating user feedback into the training process. This would allow the model to improve its predictions over time as more perceptual data is collected.

## 📚 References

The following key references have contributed to the development of this project and the foundation of the methods used:

- **T. Mertens et al.**, *Exposure Fusion*, Pacific Graphics, 2007.  
  This paper introduces the exposure fusion technique, which is a cornerstone for HDR imaging.

- **A. Vanmali et al.**, *Low Complexity Detail Preserving Multi-Exposure Image Fusion*, NCC 2013.  
  The work provides a multi-exposure image fusion method that preserves detail while reducing noise.

- **Y. Zhang and D. M. Chandler**, *No-Reference IQA Based on Log-Derivative Statistics*, SPIE, 2013.  
  This paper introduces a no-reference image quality assessment method that is grounded in statistical image features.

- **K. R. Prabhakar et al.**, *DeepFuse*, ICCV 2017.  
  DeepFuse is a deep learning-based approach to image fusion, which inspired the use of CNNs for perceptual quality prediction in this project.

- **L. Ma et al.**, *FusionGAN: A Generative Adversarial Network for Image Fusion*, IEEE Transactions on Image Processing, 2019.  
  This paper explores the use of generative adversarial networks for image fusion, providing a modern perspective on image fusion tasks.
