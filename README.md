# Image to Sketch Converter 🎨

A deep learning project that transforms RGB photographs into realistic grayscale pencil sketches using a U-Net Convolutional Neural Network (CNN).

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Overview

This project implements an advanced image-to-image translation system that converts regular photographs into artistic pencil-style sketches. Using a U-Net architecture with skip connections, the model learns to capture both fine-grained textures and high-level structural features, producing visually appealing and structurally accurate sketch outputs.

## Features

- **Deep Learning Architecture**: U-Net-based encoder-decoder with skip connections
- **High-Quality Output**: Generates realistic pencil sketches while preserving important details
- **Web Interface**: Interactive Streamlit application for easy image conversion
- **Comprehensive Metrics**: Evaluates model performance using MAE, MSE, SSIM, and PSNR
- **Pre-trained Model**: Ready-to-use trained model for immediate deployment

## Model Architecture

The U-Net architecture consists of:

- **Encoder**: Hierarchical feature extraction through successive convolution and downsampling
- **Decoder**: Upsampling layers to reconstruct sketch images
- **Skip Connections**: Direct pathways from encoder to decoder for preserving spatial information
- **Combined Loss**: MAE + SSIM loss function for improved edge preservation and visual quality

## Dataset

The model is trained on paired RGB photographs and corresponding grayscale sketch images stored in Parquet format. The dataset undergoes the following preprocessing:

- Image decoding from Parquet files
- Resizing to consistent resolution
- Pixel value normalization (0-1 range)
- Train/test split (80:20 ratio)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/satwik-shreshth/Image_To_Sketch.git
cd Image_To_Sketch
```

2. Install required dependencies:
```bash
pip install tensorflow keras numpy opencv-python pillow streamlit pandas pyarrow
```

## Usage

### Training the Model

Open and run the Jupyter notebook:
```bash
jupyter notebook Latest_ImageSketch-Copy1.ipynb
```

The notebook includes:
- Data loading and preprocessing
- Model architecture definition
- Training loop (1000 epochs, batch size 64)
- Evaluation and visualization
- Model saving

### Running the Web Application

Launch the Streamlit interface:
```bash
streamlit run lat_interface.py
```

Then:
1. Open your browser to the provided local URL (typically `http://localhost:8501`)
2. Upload an image using the file uploader
3. Click "Convert to Sketch"
4. Download or view your generated pencil sketch

### Using the Pre-trained Model

```python
from tensorflow import keras
import numpy as np
from PIL import Image

# Load the pre-trained model
model = keras.models.load_model('Latest_image2Sketch_Model.keras')

# Preprocess your image
image = Image.open('your_image.jpg')
image = image.resize((256, 256))  # Adjust size as needed
image_array = np.array(image) / 255.0
image_array = np.expand_dims(image_array, axis=0)

# Generate sketch
sketch = model.predict(image_array)
sketch = (sketch[0] * 255).astype(np.uint8)

# Save or display
sketch_image = Image.fromarray(sketch.squeeze())
sketch_image.save('output_sketch.png')
```

## Training Configuration

| Parameter | Value |
|-----------|-------|
| **Optimizer** | Adam |
| **Loss Function** | MAE + SSIM |
| **Epochs** | 1000 |
| **Batch Size** | 64 |
| **Learning Rate** | Default (0.001) |

## Evaluation Metrics

The model is evaluated using multiple quantitative metrics:

- **MAE** (Mean Absolute Error): Measures average pixel-wise difference
- **MSE** (Mean Squared Error): Emphasizes larger errors
- **SSIM** (Structural Similarity Index): Assesses perceptual quality
- **PSNR** (Peak Signal-to-Noise Ratio): Measures reconstruction quality

## Project Structure

```
Image_To_Sketch/
│
├── Latest_ImageSketch-Copy1.ipynb    # Main training notebook
├── Latest_image2Sketch_Model.keras   # Pre-trained model weights
├── lat_interface.py                  # Streamlit web application
└── README.md                         # Project documentation
```

## Applications

This technology can be applied to:

- **Digital Art**: Create artistic sketches from photographs
- **Design Tools**: Generate sketch templates for artists and designers
- **Education**: Computer vision and deep learning demonstrations
- **Entertainment**: Photo filters and creative effects
- **Preprocessing**: Sketch-based image retrieval systems

## Technical Stack

- **Deep Learning**: TensorFlow, Keras
- **Image Processing**: OpenCV, PIL (Pillow)
- **Data Handling**: NumPy, Pandas, PyArrow
- **Web Framework**: Streamlit
- **Development**: Jupyter Notebook

## Results

The trained model successfully:
- Preserves important structural features and edges
- Generates sketches with natural pencil-stroke appearance
- Maintains good spatial consistency with input images
- Achieves high SSIM scores indicating perceptual quality

## Future Improvements

- [ ] Add support for colored sketch generation
- [ ] Implement style transfer for different sketch styles
- [ ] Optimize model for mobile deployment
- [ ] Add batch processing capabilities
- [ ] Incorporate attention mechanisms for better detail preservation
- [ ] Create REST API for integration with other applications

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is open source and available for personal and educational use.

## Acknowledgments

- U-Net architecture inspiration from the original paper by Ronneberger et al.
- TensorFlow and Keras teams for the excellent deep learning framework
- The open-source community for various tools and libraries

## Contact

**Author**: Satwik Shreshth

**Project Link**: [https://github.com/satwik-shreshth/Image_To_Sketch](https://github.com/satwik-shreshth/Image_To_Sketch)

## Citation

If you use this project in your research or work, please cite:

```bibtex
@software{image_to_sketch,
  author = {Satwik Shreshth},
  title = {Image to Sketch Converter using U-Net},
  year = {2024},
  url = {https://github.com/satwik-shreshth/Image_To_Sketch}
}
```

---

⭐ If you find this project useful, please consider giving it a star!
