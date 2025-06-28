"""
SmartImageSuite Modules Package

This package contains all the image processing modules for the SmartImageSuite application.

Modules:
- intensity_transform: Intensity transformation operations
- spatial_filters: Spatial domain filtering operations
- frequency_filters: Frequency domain filtering operations
- restoration: Image restoration and deconvolution
- color_processing: Color space operations and color processing
- compression: Image compression algorithms
- wavelet_tools: Wavelet-based image processing
"""

from .intensity_transform import IntensityTransform
from .spatial_filters import SpatialFilters
from .frequency_filters import FrequencyFilters
from .restoration import ImageRestoration
from .color_processing import ColorProcessing
from .compression import ImageCompression
from .wavelet_tools import WaveletTools

__all__ = [
    'IntensityTransform',
    'SpatialFilters', 
    'FrequencyFilters',
    'ImageRestoration',
    'ColorProcessing',
    'ImageCompression',
    'WaveletTools'
]

__version__ = '1.0.0'
__author__ = '[Your Name]'
__email__ = '[your.email@asu.edu]' 