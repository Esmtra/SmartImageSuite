# SmartImageSuite

A comprehensive image processing and analysis toolkit developed for ASU Image Processing Course.

## Overview

SmartImageSuite is a powerful Python-based application that provides a wide range of image processing capabilities through an intuitive graphical user interface. The toolkit includes various modules for intensity transformations, spatial and frequency filtering, image restoration, color processing, compression, and wavelet-based operations.

## Features

### Core Modules

1. **Intensity Transform** (`intensity_transform.py`)
   - Linear transformations (brightness/contrast adjustment)
   - Logarithmic and power-law transformations
   - Histogram equalization (global and adaptive)
   - Various thresholding methods (binary, adaptive, Otsu)

2. **Spatial Filters** (`spatial_filters.py`)
   - Mean, Gaussian, and median filtering
   - Bilateral filtering for edge-preserving smoothing
   - Edge detection (Laplacian, Sobel, Canny)
   - Unsharp masking for image sharpening
   - Morphological operations

3. **Frequency Filters** (`frequency_filters.py`)
   - FFT-based frequency domain processing
   - Ideal, Butterworth, and Gaussian filters
   - Low-pass, high-pass, and band-pass filtering
   - Notch filtering for periodic noise removal

4. **Image Restoration** (`restoration.py`)
   - Wiener filtering for deconvolution
   - Lucy-Richardson deconvolution
   - Blind deconvolution
   - Motion blur restoration
   - Noise reduction techniques

5. **Color Processing** (`color_processing.py`)
   - Color space conversions (RGB, HSV, LAB, YUV)
   - Color balance and white balance correction
   - Saturation and hue adjustment
   - Color quantization and segmentation
   - Color transfer between images

6. **Image Compression** (`compression.py`)
   - JPEG and PNG compression
   - Run-length encoding
   - DCT-based compression
   - Wavelet compression
   - Vector quantization
   - Fractal compression (simplified)

7. **Wavelet Tools** (`wavelet_tools.py`)
   - Wavelet decomposition and reconstruction
   - Wavelet-based denoising
   - Image sharpening using wavelets
   - Edge detection with wavelets
   - Image fusion using wavelets
   - Watermarking capabilities

## Installation

### Prerequisites

- Python 3.7 or higher
- Required Python packages (see requirements.txt)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/SmartImageSuite.git
cd SmartImageSuite
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python main.py
```

## Usage

### GUI Interface

The application provides a user-friendly graphical interface with the following features:

- **File Operations**: Open, save, and manage image files
- **Processing Controls**: Select and configure processing methods
- **Real-time Preview**: View original and processed images side by side
- **Parameter Adjustment**: Interactive sliders and controls for fine-tuning
- **Batch Processing**: Apply operations to multiple images

## Project Structure

```
SmartImageSuite/
├── main.py                 # Main application entry point
├── gui/
│   └── interface.py        # GUI implementation
├── modules/
│   ├── intensity_transform.py
│   ├── spatial_filters.py
│   ├── frequency_filters.py
│   ├── restoration.py
│   ├── color_processing.py
│   ├── compression.py
│   └── wavelet_tools.py
├── assets/
│   └── sample_images/      # Sample images for testing
├── README.md
└── report/
    └── final_project_report.pdf
```


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

- **Author**: Eslam Mtrawy
- **Email**: [Eselmtrawy@gmail.com]
