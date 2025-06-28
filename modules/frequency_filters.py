#!/usr/bin/env python3
"""
Frequency Filters Module
Provides various frequency domain filtering operations for image processing.

Author: [Your Name]
Date: [Current Date]
"""

import numpy as np
import cv2
from typing import Union, Tuple, Optional

class FrequencyFilters:
    """
    Class for performing frequency domain filtering operations on images.
    """
    
    def __init__(self):
        """Initialize the FrequencyFilters class."""
        pass
    
    def fft_transform(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the 2D Fast Fourier Transform of an image.
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (magnitude spectrum, phase spectrum)
        """
        # Convert to float
        img_float = image.astype(np.float32)
        
        # Apply FFT
        f_transform = np.fft.fft2(img_float)
        f_shift = np.fft.fftshift(f_transform)
        
        # Calculate magnitude and phase
        magnitude = np.abs(f_shift)
        phase = np.angle(f_shift)
        
        return magnitude, phase
    
    def inverse_fft(self, magnitude: np.ndarray, phase: np.ndarray) -> np.ndarray:
        """
        Compute the inverse 2D Fast Fourier Transform.
        
        Args:
            magnitude: Magnitude spectrum
            phase: Phase spectrum
            
        Returns:
            Reconstructed image
        """
        # Reconstruct complex FFT
        f_shift = magnitude * np.exp(1j * phase)
        
        # Inverse shift
        f_transform = np.fft.ifftshift(f_shift)
        
        # Inverse FFT
        img_reconstructed = np.fft.ifft2(f_transform)
        
        return np.abs(img_reconstructed).astype(np.uint8)
    
    def ideal_low_pass_filter(self, image: np.ndarray, cutoff_freq: float) -> np.ndarray:
        """
        Apply ideal low-pass filter in frequency domain.
        
        Args:
            image: Input image
            cutoff_freq: Cutoff frequency (0-1)
            
        Returns:
            Filtered image
        """
        # Get image dimensions
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2
        
        # Create mask
        mask = np.zeros((rows, cols), np.uint8)
        y, x = np.ogrid[:rows, :cols]
        mask_area = (x - ccol) ** 2 + (y - crow) ** 2 <= (cutoff_freq * min(rows, cols)) ** 2
        mask[mask_area] = 1
        
        # Apply FFT and filter
        magnitude, phase = self.fft_transform(image)
        filtered_magnitude = magnitude * mask
        
        # Inverse FFT
        return self.inverse_fft(filtered_magnitude, phase)
    
    def ideal_high_pass_filter(self, image: np.ndarray, cutoff_freq: float) -> np.ndarray:
        """
        Apply ideal high-pass filter in frequency domain.
        
        Args:
            image: Input image
            cutoff_freq: Cutoff frequency (0-1)
            
        Returns:
            Filtered image
        """
        # Get image dimensions
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2
        
        # Create mask
        mask = np.ones((rows, cols), np.uint8)
        y, x = np.ogrid[:rows, :cols]
        mask_area = (x - ccol) ** 2 + (y - crow) ** 2 <= (cutoff_freq * min(rows, cols)) ** 2
        mask[mask_area] = 0
        
        # Apply FFT and filter
        magnitude, phase = self.fft_transform(image)
        filtered_magnitude = magnitude * mask
        
        # Inverse FFT
        return self.inverse_fft(filtered_magnitude, phase)
    
    def butterworth_low_pass_filter(self, image: np.ndarray, cutoff_freq: float, 
                                  order: int = 2) -> np.ndarray:
        """
        Apply Butterworth low-pass filter in frequency domain.
        
        Args:
            image: Input image
            cutoff_freq: Cutoff frequency (0-1)
            order: Filter order (default: 2)
            
        Returns:
            Filtered image
        """
        # Get image dimensions
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2
        
        # Create Butterworth filter
        y, x = np.ogrid[:rows, :cols]
        distance = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)
        mask = 1 / (1 + (distance / (cutoff_freq * min(rows, cols))) ** (2 * order))
        
        # Apply FFT and filter
        magnitude, phase = self.fft_transform(image)
        filtered_magnitude = magnitude * mask
        
        # Inverse FFT
        return self.inverse_fft(filtered_magnitude, phase)
    
    def butterworth_high_pass_filter(self, image: np.ndarray, cutoff_freq: float, 
                                   order: int = 2) -> np.ndarray:
        """
        Apply Butterworth high-pass filter in frequency domain.
        
        Args:
            image: Input image
            cutoff_freq: Cutoff frequency (0-1)
            order: Filter order (default: 2)
            
        Returns:
            Filtered image
        """
        # Get image dimensions
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2
        
        # Create Butterworth filter
        y, x = np.ogrid[:rows, :cols]
        distance = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)
        mask = 1 / (1 + (cutoff_freq * min(rows, cols) / (distance + 1e-10)) ** (2 * order))
        
        # Apply FFT and filter
        magnitude, phase = self.fft_transform(image)
        filtered_magnitude = magnitude * mask
        
        # Inverse FFT
        return self.inverse_fft(filtered_magnitude, phase)
    
    def gaussian_low_pass_filter(self, image: np.ndarray, cutoff_freq: float) -> np.ndarray:
        """
        Apply Gaussian low-pass filter in frequency domain.
        
        Args:
            image: Input image
            cutoff_freq: Cutoff frequency (0-1)
            
        Returns:
            Filtered image
        """
        # Get image dimensions
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2
        
        # Create Gaussian filter
        y, x = np.ogrid[:rows, :cols]
        distance = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)
        mask = np.exp(-(distance ** 2) / (2 * (cutoff_freq * min(rows, cols)) ** 2))
        
        # Apply FFT and filter
        magnitude, phase = self.fft_transform(image)
        filtered_magnitude = magnitude * mask
        
        # Inverse FFT
        return self.inverse_fft(filtered_magnitude, phase)
    
    def gaussian_high_pass_filter(self, image: np.ndarray, cutoff_freq: float) -> np.ndarray:
        """
        Apply Gaussian high-pass filter in frequency domain.
        
        Args:
            image: Input image
            cutoff_freq: Cutoff frequency (0-1)
            
        Returns:
            Filtered image
        """
        # Get image dimensions
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2
        
        # Create Gaussian filter
        y, x = np.ogrid[:rows, :cols]
        distance = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)
        mask = 1 - np.exp(-(distance ** 2) / (2 * (cutoff_freq * min(rows, cols)) ** 2))
        
        # Apply FFT and filter
        magnitude, phase = self.fft_transform(image)
        filtered_magnitude = magnitude * mask
        
        # Inverse FFT
        return self.inverse_fft(filtered_magnitude, phase)
    
    def band_pass_filter(self, image: np.ndarray, low_freq: float, high_freq: float) -> np.ndarray:
        """
        Apply band-pass filter in frequency domain.
        
        Args:
            image: Input image
            low_freq: Lower cutoff frequency (0-1)
            high_freq: Upper cutoff frequency (0-1)
            
        Returns:
            Filtered image
        """
        # Apply low-pass filter
        low_filtered = self.butterworth_low_pass_filter(image, high_freq)
        
        # Apply high-pass filter
        return self.butterworth_high_pass_filter(low_filtered, low_freq)
    
    def notch_filter(self, image: np.ndarray, notch_points: list, 
                    notch_width: float = 10) -> np.ndarray:
        """
        Apply notch filter to remove specific frequencies.
        
        Args:
            image: Input image
            notch_points: List of (u, v) coordinates for notch points
            notch_width: Width of the notch (default: 10)
            
        Returns:
            Filtered image
        """
        # Get image dimensions
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2
        
        # Create notch filter mask
        mask = np.ones((rows, cols), np.uint8)
        
        for u, v in notch_points:
            # Create notch at (u, v) and (-u, -v)
            y, x = np.ogrid[:rows, :cols]
            distance1 = np.sqrt((x - (ccol + u)) ** 2 + (y - (crow + v)) ** 2)
            distance2 = np.sqrt((x - (ccol - u)) ** 2 + (y - (crow - v)) ** 2)
            
            mask[distance1 <= notch_width] = 0
            mask[distance2 <= notch_width] = 0
        
        # Apply FFT and filter
        magnitude, phase = self.fft_transform(image)
        filtered_magnitude = magnitude * mask
        
        # Inverse FFT
        return self.inverse_fft(filtered_magnitude, phase) 