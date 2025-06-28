#!/usr/bin/env python3
"""
Intensity Transform Module
Provides various intensity transformation operations for image processing.

Author: [Your Name]
Date: [Current Date]
"""

import numpy as np
import cv2
from typing import Union, Tuple, Optional

class IntensityTransform:
    """
    Class for performing intensity transformations on images.
    """
    
    def __init__(self):
        """Initialize the IntensityTransform class."""
        pass
    
    def linear_transform(self, image: np.ndarray, alpha: float = 1.0, beta: float = 0) -> np.ndarray:
        """
        Apply linear transformation: g(x,y) = α * f(x,y) + β
        
        Args:
            image: Input image
            alpha: Contrast factor (default: 1.0)
            beta: Brightness offset (default: 0)
            
        Returns:
            Transformed image
        """
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    def log_transform(self, image: np.ndarray, c: float = 1.0) -> np.ndarray:
        """
        Apply logarithmic transformation: g(x,y) = c * log(1 + f(x,y))
        
        Args:
            image: Input image
            c: Scaling constant (default: 1.0)
            
        Returns:
            Transformed image
        """
        # Convert to float for log operation
        img_float = image.astype(np.float64)
        log_transformed = c * np.log(1 + img_float)
        
        # Normalize to 0-255 range
        log_transformed = np.clip(log_transformed, 0, 255)
        return log_transformed.astype(np.uint8)
    
    def power_law_transform(self, image: np.ndarray, gamma: float = 1.0, c: float = 1.0) -> np.ndarray:
        """
        Apply power-law transformation: g(x,y) = c * f(x,y)^γ
        
        Args:
            image: Input image
            gamma: Gamma value (default: 1.0)
            c: Scaling constant (default: 1.0)
            
        Returns:
            Transformed image
        """
        # Convert to float for power operation
        img_float = image.astype(np.float64) / 255.0
        power_transformed = c * np.power(img_float, gamma)
        
        # Normalize to 0-255 range
        power_transformed = np.clip(power_transformed * 255, 0, 255)
        return power_transformed.astype(np.uint8)
    
    def histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """
        Apply histogram equalization to enhance image contrast.
        
        Args:
            image: Input image
            
        Returns:
            Equalized image
        """
        if len(image.shape) == 3:
            # Convert to LAB color space for better results
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab[:,:,0] = cv2.equalizeHist(lab[:,:,0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            return cv2.equalizeHist(image)
    
    def adaptive_histogram_equalization(self, image: np.ndarray, 
                                      clip_limit: float = 2.0, 
                                      tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """
        Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
        
        Args:
            image: Input image
            clip_limit: Threshold for contrast limiting (default: 2.0)
            tile_grid_size: Size of grid for histogram equalization (default: (8, 8))
            
        Returns:
            CLAHE processed image
        """
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        
        if len(image.shape) == 3:
            # Apply CLAHE to each channel
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            return clahe.apply(image)
    
    def threshold(self, image: np.ndarray, threshold_value: int = 127, 
                 max_value: int = 255, threshold_type: str = 'binary') -> np.ndarray:
        """
        Apply thresholding to create binary or binary inverted image.
        
        Args:
            image: Input image
            threshold_value: Threshold value (default: 127)
            max_value: Maximum value (default: 255)
            threshold_type: Type of thresholding ('binary', 'binary_inv', 'trunc', 'tozero', 'tozero_inv')
            
        Returns:
            Thresholded image
        """
        threshold_types = {
            'binary': cv2.THRESH_BINARY,
            'binary_inv': cv2.THRESH_BINARY_INV,
            'trunc': cv2.THRESH_TRUNC,
            'tozero': cv2.THRESH_TOZERO,
            'tozero_inv': cv2.THRESH_TOZERO_INV
        }
        
        thresh_type = threshold_types.get(threshold_type, cv2.THRESH_BINARY)
        _, thresholded = cv2.threshold(image, threshold_value, max_value, thresh_type)
        return thresholded
    
    def adaptive_threshold(self, image: np.ndarray, max_value: int = 255,
                          adaptive_method: str = 'gaussian', threshold_type: str = 'binary',
                          block_size: int = 11, c: float = 2) -> np.ndarray:
        """
        Apply adaptive thresholding.
        
        Args:
            image: Input image
            max_value: Maximum value (default: 255)
            adaptive_method: Adaptive method ('mean', 'gaussian')
            threshold_type: Threshold type ('binary', 'binary_inv')
            block_size: Size of pixel neighborhood (default: 11)
            c: Constant subtracted from mean (default: 2)
            
        Returns:
            Adaptively thresholded image
        """
        adaptive_methods = {
            'mean': cv2.ADAPTIVE_THRESH_MEAN_C,
            'gaussian': cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        }
        
        threshold_types = {
            'binary': cv2.THRESH_BINARY,
            'binary_inv': cv2.THRESH_BINARY_INV
        }
        
        adapt_method = adaptive_methods.get(adaptive_method, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
        thresh_type = threshold_types.get(threshold_type, cv2.THRESH_BINARY)
        
        return cv2.adaptiveThreshold(image, max_value, adapt_method, thresh_type, block_size, c)
    
    def otsu_threshold(self, image: np.ndarray, max_value: int = 255) -> Tuple[np.ndarray, float]:
        """
        Apply Otsu's thresholding method.
        
        Args:
            image: Input image
            max_value: Maximum value (default: 255)
            
        Returns:
            Tuple of (thresholded image, optimal threshold value)
        """
        threshold_value, thresholded = cv2.threshold(image, 0, max_value, 
                                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresholded, threshold_value 