#!/usr/bin/env python3
"""
Image Restoration Module
Provides various image restoration and deconvolution operations.

Author: [Your Name]
Date: [Current Date]
"""

import numpy as np
import cv2
from typing import Union, Tuple, Optional

class ImageRestoration:
    """
    Class for performing image restoration operations.
    """
    
    def __init__(self):
        """Initialize the ImageRestoration class."""
        pass
    
    def wiener_filter(self, image: np.ndarray, psf: np.ndarray, noise_power: float = 0.01) -> np.ndarray:
        """
        Apply Wiener filter for image restoration.
        
        Args:
            image: Input degraded image
            psf: Point spread function
            noise_power: Noise-to-signal power ratio (default: 0.01)
            
        Returns:
            Restored image
        """
        # Pad PSF to image size
        psf_padded = np.zeros_like(image, dtype=np.float64)
        psf_padded[:psf.shape[0], :psf.shape[1]] = psf
        
        # Shift PSF to center
        psf_shifted = np.fft.fftshift(psf_padded)
        
        # Compute FFT
        image_fft = np.fft.fft2(image.astype(np.float64))
        psf_fft = np.fft.fft2(psf_shifted)
        
        # Wiener filter
        psf_conj = np.conj(psf_fft)
        psf_mag_sq = np.abs(psf_fft) ** 2
        wiener_filter = psf_conj / (psf_mag_sq + noise_power)
        
        # Apply filter
        restored_fft = image_fft * wiener_filter
        restored = np.fft.ifft2(restored_fft)
        
        return np.abs(restored).astype(np.uint8)
    
    def lucy_richardson_deconvolution(self, image: np.ndarray, psf: np.ndarray, 
                                    iterations: int = 10) -> np.ndarray:
        """
        Apply Lucy-Richardson deconvolution algorithm.
        
        Args:
            image: Input degraded image
            psf: Point spread function
            iterations: Number of iterations (default: 10)
            
        Returns:
            Restored image
        """
        # Pad PSF to image size
        psf_padded = np.zeros_like(image, dtype=np.float64)
        psf_padded[:psf.shape[0], :psf.shape[1]] = psf
        
        # Shift PSF to center
        psf_shifted = np.fft.fftshift(psf_padded)
        
        # Initialize estimate
        estimate = image.astype(np.float64)
        
        # Lucy-Richardson iterations
        for _ in range(iterations):
            # Forward projection
            forward = np.fft.ifft2(np.fft.fft2(estimate) * np.fft.fft2(psf_shifted))
            forward = np.abs(forward)
            
            # Avoid division by zero
            forward = np.where(forward == 0, 1e-10, forward)
            
            # Ratio
            ratio = image.astype(np.float64) / forward
            
            # Backward projection
            backward = np.fft.ifft2(np.fft.fft2(ratio) * np.conj(np.fft.fft2(psf_shifted)))
            backward = np.abs(backward)
            
            # Update estimate
            estimate = estimate * backward
        
        return np.clip(estimate, 0, 255).astype(np.uint8)
    
    def blind_deconvolution(self, image: np.ndarray, kernel_size: Tuple[int, int] = (5, 5),
                          iterations: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply blind deconvolution to estimate both PSF and restored image.
        
        Args:
            image: Input degraded image
            kernel_size: Size of the estimated PSF (default: (5, 5))
            iterations: Number of iterations (default: 10)
            
        Returns:
            Tuple of (restored image, estimated PSF)
        """
        # Initialize PSF estimate
        psf_estimate = np.ones(kernel_size, dtype=np.float64)
        psf_estimate = psf_estimate / np.sum(psf_estimate)
        
        # Initialize image estimate
        image_estimate = image.astype(np.float64)
        
        for _ in range(iterations):
            # Update PSF estimate
            psf_estimate = self._update_psf_estimate(image, image_estimate, psf_estimate)
            
            # Update image estimate using Wiener filter
            image_estimate = self.wiener_filter(image, psf_estimate)
        
        return image_estimate.astype(np.uint8), psf_estimate
    
    def _update_psf_estimate(self, original: np.ndarray, estimate: np.ndarray, 
                           psf: np.ndarray) -> np.ndarray:
        """
        Update PSF estimate in blind deconvolution.
        
        Args:
            original: Original degraded image
            estimate: Current image estimate
            psf: Current PSF estimate
            
        Returns:
            Updated PSF estimate
        """
        # This is a simplified PSF update
        # In practice, more sophisticated methods would be used
        
        # Pad PSF to image size
        psf_padded = np.zeros_like(original, dtype=np.float64)
        psf_padded[:psf.shape[0], :psf.shape[1]] = psf
        
        # Shift PSF to center
        psf_shifted = np.fft.fftshift(psf_padded)
        
        # Compute correlation
        correlation = np.fft.ifft2(np.fft.fft2(original) * np.conj(np.fft.fft2(estimate)))
        correlation = np.abs(correlation)
        
        # Extract PSF estimate
        psf_new = correlation[:psf.shape[0], :psf.shape[1]]
        psf_new = np.clip(psf_new, 0, None)
        psf_new = psf_new / np.sum(psf_new)
        
        return psf_new
    
    def motion_blur_restoration(self, image: np.ndarray, angle: float, length: float) -> np.ndarray:
        """
        Restore motion blur using estimated motion parameters.
        
        Args:
            image: Input motion-blurred image
            angle: Motion angle in degrees
            length: Motion length in pixels
            
        Returns:
            Restored image
        """
        # Create motion blur PSF
        psf = self._create_motion_psf(angle, length, image.shape)
        
        # Apply Wiener filter
        return self.wiener_filter(image, psf)
    
    def _create_motion_psf(self, angle: float, length: float, image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Create motion blur point spread function.
        
        Args:
            angle: Motion angle in degrees
            length: Motion length in pixels
            image_shape: Shape of the image
            
        Returns:
            Motion blur PSF
        """
        # Convert angle to radians
        angle_rad = np.radians(angle)
        
        # Calculate motion vector
        dx = length * np.cos(angle_rad)
        dy = length * np.sin(angle_rad)
        
        # Create PSF
        psf_size = int(np.ceil(max(abs(dx), abs(dy))))
        psf = np.zeros((psf_size, psf_size))
        
        # Draw line in PSF
        center = psf_size // 2
        x1, y1 = center, center
        x2, y2 = int(center + dx), int(center + dy)
        
        # Bresenham's line algorithm
        points = self._bresenham_line(x1, y1, x2, y2)
        for x, y in points:
            if 0 <= x < psf_size and 0 <= y < psf_size:
                psf[y, x] = 1
        
        # Normalize PSF
        psf = psf / np.sum(psf)
        
        return psf
    
    def _bresenham_line(self, x1: int, y1: int, x2: int, y2: int) -> list:
        """
        Bresenham's line algorithm for drawing lines.
        
        Args:
            x1, y1: Start point
            x2, y2: End point
            
        Returns:
            List of points on the line
        """
        points = []
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        
        if x1 < x2:
            sx = 1
        else:
            sx = -1
            
        if y1 < y2:
            sy = 1
        else:
            sy = -1
            
        err = dx - dy
        
        while True:
            points.append((x1, y1))
            
            if x1 == x2 and y1 == y2:
                break
                
            e2 = 2 * err
            if e2 > -dy:
                err = err - dy
                x1 = x1 + sx
            if e2 < dx:
                err = err + dx
                y1 = y1 + sy
                
        return points
    
    def noise_reduction(self, image: np.ndarray, method: str = 'gaussian', 
                       **kwargs) -> np.ndarray:
        """
        Apply noise reduction techniques.
        
        Args:
            image: Input noisy image
            method: Noise reduction method ('gaussian', 'median', 'bilateral')
            **kwargs: Additional parameters for the method
            
        Returns:
            Denoised image
        """
        if method == 'gaussian':
            kernel_size = kwargs.get('kernel_size', 3)
            sigma = kwargs.get('sigma', 1.0)
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
            
        elif method == 'median':
            kernel_size = kwargs.get('kernel_size', 3)
            return cv2.medianBlur(image, kernel_size)
            
        elif method == 'bilateral':
            d = kwargs.get('d', 9)
            sigma_color = kwargs.get('sigma_color', 75)
            sigma_space = kwargs.get('sigma_space', 75)
            return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
            
        else:
            raise ValueError(f"Unknown noise reduction method: {method}") 