# Photo to Pencil Sketch Conversion Using U-Net

## Project Overview
This project presents the design and implementation of a U-Net–based Convolutional Neural Network (CNN) for converting RGB photographs into grayscale, pencil-style sketches. The objective is to learn an effective image-to-image translation model that captures both low-level textures and high-level structural features present in natural images. The proposed approach demonstrates the applicability of deep encoder–decoder architectures in artistic image transformation and computer vision applications.

---

## Model Architecture
The proposed model is based on the U-Net architecture, which follows a symmetric encoder–decoder design with skip connections. The encoder extracts hierarchical feature representations from input photographs through successive convolution and downsampling operations. The decoder reconstructs the sketch images using upsampling layers, while skip connections transfer spatial information from the encoder to the decoder, enabling accurate reconstruction of fine details and edges.

---

## Dataset Description
The dataset consists of paired RGB photographs and their corresponding grayscale sketch images stored in Parquet format. Each image pair represents the same scene in photo and sketch form, enabling supervised learning for image-to-image translation.

### Data Preprocessing
The following preprocessing steps are applied to prepare the dataset for training:
- Decoding image data from Parquet files  
- Resizing images to a consistent resolution  
- Normalizing pixel values  
- Converting images into NumPy arrays  

The dataset is divided into training and testing subsets using an 80:20 split ratio.

---

## Training Configuration
The model is trained using a combination of pixel-level and perceptual loss functions to ensure both numerical accuracy and structural consistency in the generated sketches.

### Training Parameters
- **Optimizer:** Adam  
- **Loss Function:**  
  - Mean Absolute Error (MAE)  
  - Structural Similarity Index Measure (SSIM)  
- **Number of Epochs:** 50  
- **Batch Size:** 16  

The combined MAE–SSIM loss function improves edge preservation and overall visual quality.

---

## Evaluation Methodology
The performance of the trained model is evaluated on the test dataset using multiple quantitative metrics.

### Evaluation Metrics
- Mean Absolute Error (MAE)  
- Mean Squared Error (MSE)  
- Structural Similarity Index Measure (SSIM)  
- Peak Signal-to-Noise Ratio (PSNR)  

In addition to quantitative evaluation, qualitative analysis is performed through visual comparison of input photographs, ground truth sketches, and model-generated outputs. These comparisons demonstrate the model’s effectiveness in producing visually coherent and structurally accurate sketches.

---

## Model Persistence
After training, the model is saved to disk to allow reuse for inference and deployment without the need for retraining. This facilitates efficient experimentation and integration into downstream applications.

---

## Deployment
The trained model is deployed using a Streamlit-based web application.

### Web Application Functionality
The application enables users to upload input photographs and receive pencil-style sketch predictions in real time, demonstrating the practical applicability of the proposed system.

---

## Applications
Potential applications of the proposed approach include:
- Digital art and creative design tools  
- Artistic image translation systems  
- Computer vision research and education  
- Interactive image processing applications  

---

## Technologies Used
- Python  
- TensorFlow and Keras  
- NumPy  
- OpenCV and PIL  
- Streamlit  
- Parquet data processing libraries  

---

## Conclusion
This project demonstrates the effectiveness of U-Net–based encoder–decoder architectures for artistic image translation tasks. By leveraging skip connections and a combined MAE–SSIM loss function, the model successfully generates high-quality pencil-style sketches from RGB photographs. The results highlight the suitability of deep learning methods for practical digital art and computer vision applications.
