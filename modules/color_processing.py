#!/usr/bin/env python3
"""
Color Processing Module
Provides various color space transformations and color processing operations.

Author: [Your Name]
Date: [Current Date]
"""

import numpy as np
import cv2
from typing import Union, Tuple, Optional

class ColorProcessing:
    """
    Class for performing color processing operations on images.
    """
    
    def __init__(self):
        """Initialize the ColorProcessing class."""
        pass
    
    def rgb_to_hsv(self, image: np.ndarray) -> np.ndarray:
        """
        Convert RGB image to HSV color space.
        
        Args:
            image: Input RGB image
            
        Returns:
            HSV image
        """
        return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    def hsv_to_rgb(self, image: np.ndarray) -> np.ndarray:
        """
        Convert HSV image to RGB color space.
        
        Args:
            image: Input HSV image
            
        Returns:
            RGB image
        """
        return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    
    def rgb_to_lab(self, image: np.ndarray) -> np.ndarray:
        """
        Convert RGB image to LAB color space.
        
        Args:
            image: Input RGB image
            
        Returns:
            LAB image
        """
        return cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    def lab_to_rgb(self, image: np.ndarray) -> np.ndarray:
        """
        Convert LAB image to RGB color space.
        
        Args:
            image: Input LAB image
            
        Returns:
            RGB image
        """
        return cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
    
    def rgb_to_yuv(self, image: np.ndarray) -> np.ndarray:
        """
        Convert RGB image to YUV color space.
        
        Args:
            image: Input RGB image
            
        Returns:
            YUV image
        """
        return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    
    def yuv_to_rgb(self, image: np.ndarray) -> np.ndarray:
        """
        Convert YUV image to RGB color space.
        
        Args:
            image: Input YUV image
            
        Returns:
            RGB image
        """
        return cv2.cvtColor(image, cv2.COLOR_YUV2RGB)
    
    def adjust_brightness(self, image: np.ndarray, factor: float) -> np.ndarray:
        """
        Adjust image brightness.
        
        Args:
            image: Input image
            factor: Brightness factor (>1 for brighter, <1 for darker)
            
        Returns:
            Brightness adjusted image
        """
        return np.clip(image * factor, 0, 255).astype(np.uint8)
    
    def adjust_contrast(self, image: np.ndarray, factor: float) -> np.ndarray:
        """
        Adjust image contrast.
        
        Args:
            image: Input image
            factor: Contrast factor (>1 for higher contrast, <1 for lower contrast)
            
        Returns:
            Contrast adjusted image
        """
        mean = np.mean(image)
        return np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)
    
    def adjust_saturation(self, image: np.ndarray, factor: float) -> np.ndarray:
        """
        Adjust image saturation.
        
        Args:
            image: Input RGB image
            factor: Saturation factor (>1 for higher saturation, <1 for lower saturation)
            
        Returns:
            Saturation adjusted image
        """
        hsv = self.rgb_to_hsv(image)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
        return self.hsv_to_rgb(hsv)
    
    def adjust_hue(self, image: np.ndarray, shift: int) -> np.ndarray:
        """
        Adjust image hue.
        
        Args:
            image: Input RGB image
            shift: Hue shift in degrees (-180 to 180)
            
        Returns:
            Hue adjusted image
        """
        hsv = self.rgb_to_hsv(image)
        hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180
        return self.hsv_to_rgb(hsv)
    
    def color_balance(self, image: np.ndarray, r_factor: float = 1.0, 
                     g_factor: float = 1.0, b_factor: float = 1.0) -> np.ndarray:
        """
        Apply color balance to RGB image.
        
        Args:
            image: Input RGB image
            r_factor: Red channel factor
            g_factor: Green channel factor
            b_factor: Blue channel factor
            
        Returns:
            Color balanced image
        """
        balanced = image.copy().astype(np.float32)
        balanced[:, :, 0] *= b_factor  # Blue
        balanced[:, :, 1] *= g_factor  # Green
        balanced[:, :, 2] *= r_factor  # Red
        return np.clip(balanced, 0, 255).astype(np.uint8)
    
    def white_balance(self, image: np.ndarray, method: str = 'gray_world') -> np.ndarray:
        """
        Apply white balance correction.
        
        Args:
            image: Input RGB image
            method: White balance method ('gray_world', 'max_rgb')
            
        Returns:
            White balanced image
        """
        if method == 'gray_world':
            # Gray world assumption
            mean_r = np.mean(image[:, :, 2])
            mean_g = np.mean(image[:, :, 1])
            mean_b = np.mean(image[:, :, 0])
            
            gray_value = (mean_r + mean_g + mean_b) / 3
            
            r_factor = gray_value / mean_r
            g_factor = gray_value / mean_g
            b_factor = gray_value / mean_b
            
        elif method == 'max_rgb':
            # Max RGB assumption
            max_r = np.max(image[:, :, 2])
            max_g = np.max(image[:, :, 1])
            max_b = np.max(image[:, :, 0])
            
            max_value = max(max_r, max_g, max_b)
            
            r_factor = max_value / max_r
            g_factor = max_value / max_g
            b_factor = max_value / max_b
            
        else:
            raise ValueError(f"Unknown white balance method: {method}")
        
        return self.color_balance(image, r_factor, g_factor, b_factor)
    
    def histogram_equalization_color(self, image: np.ndarray) -> np.ndarray:
        """
        Apply histogram equalization to color image.
        
        Args:
            image: Input RGB image
            
        Returns:
            Histogram equalized image
        """
        # Convert to LAB color space
        lab = self.rgb_to_lab(image)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back to RGB
        return self.lab_to_rgb(lab)
    
    def color_quantization(self, image: np.ndarray, n_colors: int = 8) -> np.ndarray:
        """
        Reduce the number of colors in an image.
        
        Args:
            image: Input RGB image
            n_colors: Number of colors to quantize to
            
        Returns:
            Color quantized image
        """
        # Reshape image for k-means
        data = image.reshape((-1, 3))
        data = np.float32(data)
        
        # Define criteria and apply k-means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(data, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert back to uint8
        centers = np.uint8(centers)
        quantized = centers[labels.flatten()]
        quantized = quantized.reshape(image.shape)
        
        return quantized
    
    def color_segmentation(self, image: np.ndarray, target_color: Tuple[int, int, int], 
                          tolerance: int = 30) -> np.ndarray:
        """
        Segment image based on target color.
        
        Args:
            image: Input RGB image
            target_color: Target color in RGB format
            tolerance: Color tolerance for segmentation
            
        Returns:
            Binary mask of segmented regions
        """
        # Convert target color to numpy array
        target = np.array(target_color, dtype=np.uint8)
        
        # Calculate color distance
        color_diff = np.sqrt(np.sum((image - target) ** 2, axis=2))
        
        # Create binary mask
        mask = color_diff <= tolerance
        
        return mask.astype(np.uint8) * 255
    
    def color_transfer(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Transfer color statistics from source to target image.
        
        Args:
            source: Source image for color statistics
            target: Target image to apply color transfer
            
        Returns:
            Color transferred image
        """
        # Convert to LAB color space
        source_lab = self.rgb_to_lab(source)
        target_lab = self.rgb_to_lab(target)
        
        # Calculate mean and std for each channel
        source_mean = np.mean(source_lab, axis=(0, 1))
        source_std = np.std(source_lab, axis=(0, 1))
        target_mean = np.mean(target_lab, axis=(0, 1))
        target_std = np.std(target_lab, axis=(0, 1))
        
        # Apply color transfer
        result_lab = target_lab.copy().astype(np.float32)
        for i in range(3):
            result_lab[:, :, i] = ((result_lab[:, :, i] - target_mean[i]) * 
                                  (source_std[i] / target_std[i]) + source_mean[i])
        
        # Clip values and convert back to RGB
        result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
        return self.lab_to_rgb(result_lab) 