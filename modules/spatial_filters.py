#!/usr/bin/env python3
"""
Spatial Filters Module
Provides various spatial filtering operations for image processing.

Author: [Your Name]
Date: [Current Date]
"""

import numpy as np
import cv2
from typing import Union, Tuple, Optional

class SpatialFilters:
    """
    Class for performing spatial filtering operations on images.
    """
    
    def __init__(self):
        """Initialize the SpatialFilters class."""
        pass
    
    def mean_filter(self, image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        Apply mean filtering (averaging filter).
        
        Args:
            image: Input image
            kernel_size: Size of the kernel (default: 3)
            
        Returns:
            Filtered image
        """
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        return cv2.filter2D(image, -1, kernel)
    
    def gaussian_filter(self, image: np.ndarray, kernel_size: int = 3, sigma: float = 1.0) -> np.ndarray:
        """
        Apply Gaussian filtering.
        
        Args:
            image: Input image
            kernel_size: Size of the kernel (default: 3)
            sigma: Standard deviation (default: 1.0)
            
        Returns:
            Filtered image
        """
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    def median_filter(self, image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        Apply median filtering.
        
        Args:
            image: Input image
            kernel_size: Size of the kernel (default: 3)
            
        Returns:
            Filtered image
        """
        return cv2.medianBlur(image, kernel_size)
    
    def bilateral_filter(self, image: np.ndarray, d: int = 9, sigma_color: float = 75, 
                        sigma_space: float = 75) -> np.ndarray:
        """
        Apply bilateral filtering for edge-preserving smoothing.
        
        Args:
            image: Input image
            d: Diameter of each pixel neighborhood (default: 9)
            sigma_color: Filter sigma in the color space (default: 75)
            sigma_space: Filter sigma in the coordinate space (default: 75)
            
        Returns:
            Filtered image
        """
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    
    def laplacian_filter(self, image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        Apply Laplacian filtering for edge detection.
        
        Args:
            image: Input image
            kernel_size: Size of the kernel (default: 3)
            
        Returns:
            Filtered image
        """
        return cv2.Laplacian(image, cv2.CV_64F, ksize=kernel_size)
    
    def sobel_filter(self, image: np.ndarray, dx: int = 1, dy: int = 0, 
                    kernel_size: int = 3) -> np.ndarray:
        """
        Apply Sobel filtering for edge detection.
        
        Args:
            image: Input image
            dx: Order of derivative x (default: 1)
            dy: Order of derivative y (default: 0)
            kernel_size: Size of the kernel (default: 3)
            
        Returns:
            Filtered image
        """
        return cv2.Sobel(image, cv2.CV_64F, dx, dy, ksize=kernel_size)
    
    def canny_edge_detection(self, image: np.ndarray, threshold1: float = 100, 
                            threshold2: float = 200) -> np.ndarray:
        """
        Apply Canny edge detection.
        
        Args:
            image: Input image
            threshold1: First threshold for the hysteresis procedure (default: 100)
            threshold2: Second threshold for the hysteresis procedure (default: 200)
            
        Returns:
            Edge detected image
        """
        return cv2.Canny(image, threshold1, threshold2)
    
    def custom_kernel_filter(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Apply custom kernel filtering.
        
        Args:
            image: Input image
            kernel: Custom kernel matrix
            
        Returns:
            Filtered image
        """
        return cv2.filter2D(image, -1, kernel)
    
    def unsharp_masking(self, image: np.ndarray, kernel_size: int = 3, 
                       sigma: float = 1.0, amount: float = 1.0, threshold: int = 0) -> np.ndarray:
        """
        Apply unsharp masking for image sharpening.
        
        Args:
            image: Input image
            kernel_size: Size of the Gaussian kernel (default: 3)
            sigma: Standard deviation (default: 1.0)
            amount: Sharpening strength (default: 1.0)
            threshold: Threshold for minimum brightness change (default: 0)
            
        Returns:
            Sharpened image
        """
        # Create blurred version
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        
        # Calculate sharpened image
        sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
        
        # Apply threshold
        if threshold > 0:
            mask = np.abs(image.astype(np.int16) - blurred.astype(np.int16)) > threshold
            sharpened = np.where(mask, sharpened, image)
        
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    
    def morphological_operations(self, image: np.ndarray, operation: str = 'erosion', 
                               kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
        """
        Apply morphological operations.
        
        Args:
            image: Input image
            operation: Type of operation ('erosion', 'dilation', 'opening', 'closing')
            kernel_size: Size of the kernel (default: 3)
            iterations: Number of iterations (default: 1)
            
        Returns:
            Processed image
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        
        operations = {
            'erosion': cv2.erode,
            'dilation': cv2.dilate,
            'opening': cv2.morphologyEx,
            'closing': cv2.morphologyEx
        }
        
        if operation in ['opening', 'closing']:
            morph_type = cv2.MORPH_OPEN if operation == 'opening' else cv2.MORPH_CLOSE
            return cv2.morphologyEx(image, morph_type, kernel, iterations=iterations)
        else:
            return operations[operation](image, kernel, iterations=iterations) 