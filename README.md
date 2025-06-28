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

### Command Line Interface

For advanced users, individual modules can be imported and used programmatically:

```python
from modules.intensity_transform import IntensityTransform
from modules.spatial_filters import SpatialFilters

# Load an image
image = cv2.imread('input.jpg')

# Apply intensity transformation
intensity_processor = IntensityTransform()
enhanced = intensity_processor.histogram_equalization(image)

# Apply spatial filtering
spatial_processor = SpatialFilters()
filtered = spatial_processor.gaussian_filter(enhanced, kernel_size=5)
```

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

## Dependencies

### Core Dependencies
- `numpy` - Numerical computing
- `opencv-python` - Computer vision operations
- `tkinter` - GUI framework (included with Python)

### Optional Dependencies
- `pywt` - PyWavelets for wavelet operations
- `matplotlib` - Plotting and visualization
- `scikit-image` - Additional image processing functions

## Examples

### Basic Image Enhancement

```python
from modules.intensity_transform import IntensityTransform
from modules.color_processing import ColorProcessing

# Load image
image = cv2.imread('input.jpg')

# Enhance contrast
intensity_processor = IntensityTransform()
enhanced = intensity_processor.histogram_equalization(image)

# Adjust color balance
color_processor = ColorProcessing()
balanced = color_processor.white_balance(enhanced)
```

### Advanced Filtering

```python
from modules.spatial_filters import SpatialFilters
from modules.frequency_filters import FrequencyFilters

# Apply bilateral filtering
spatial_processor = SpatialFilters()
filtered = spatial_processor.bilateral_filter(image, d=9, sigma_color=75, sigma_space=75)

# Apply frequency domain filtering
freq_processor = FrequencyFilters()
low_pass = freq_processor.butterworth_low_pass_filter(filtered, cutoff_freq=0.3)
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- ASU Image Processing Course for project requirements
- OpenCV community for computer vision algorithms
- PyWavelets developers for wavelet processing capabilities

## Contact

- **Author**: [Your Name]
- **Email**: [your.email@asu.edu]
- **Course**: ASU Image Processing Course
- **Year**: 2024

## Version History

- **v1.0.0** - Initial release with basic functionality
- **v1.1.0** - Added wavelet processing capabilities
- **v1.2.0** - Enhanced GUI and added batch processing
- **v1.3.0** - Improved performance and added documentation

---

**Note**: This project is developed as part of the ASU Image Processing Course curriculum. It serves as a comprehensive demonstration of various image processing techniques and their practical applications. 