#!/usr/bin/env python3
"""
Wavelet Tools Module
Provides wavelet-based image processing operations.

Author: [Your Name]
Date: [Current Date]
"""

import numpy as np
import cv2
from typing import Union, Tuple, Optional, List

class WaveletTools:
    """
    Class for performing wavelet-based image processing operations.
    """
    
    def __init__(self):
        """Initialize the WaveletTools class."""
        pass
    
    def wavelet_decomposition(self, image: np.ndarray, wavelet: str = 'haar', 
                            level: int = 3) -> List[np.ndarray]:
        """
        Perform wavelet decomposition of an image.
        
        Args:
            image: Input image
            wavelet: Wavelet type ('haar', 'db4', 'sym4', etc.)
            level: Decomposition level
            
        Returns:
            List of wavelet coefficients
        """
        try:
            import pywt
        except ImportError:
            raise ImportError("PyWavelets library is required for wavelet operations")
        
        # Perform wavelet decomposition
        coeffs = pywt.wavedec2(image, wavelet, level=level)
        return coeffs
    
    def wavelet_reconstruction(self, coeffs: List[np.ndarray], wavelet: str = 'haar') -> np.ndarray:
        """
        Reconstruct image from wavelet coefficients.
        
        Args:
            coeffs: Wavelet coefficients
            wavelet: Wavelet type
            
        Returns:
            Reconstructed image
        """
        try:
            import pywt
        except ImportError:
            raise ImportError("PyWavelets library is required for wavelet operations")
        
        # Perform wavelet reconstruction
        reconstructed = pywt.waverec2(coeffs, wavelet)
        return reconstructed
    
    def wavelet_denoising(self, image: np.ndarray, wavelet: str = 'db4', 
                         level: int = 3, threshold: float = 0.1) -> np.ndarray:
        """
        Denoise image using wavelet thresholding.
        
        Args:
            image: Input noisy image
            wavelet: Wavelet type
            level: Decomposition level
            threshold: Threshold value for coefficient removal
            
        Returns:
            Denoised image
        """
        # Perform wavelet decomposition
        coeffs = self.wavelet_decomposition(image, wavelet, level)
        
        # Apply thresholding to detail coefficients
        coeffs_thresholded = []
        for i, coeff in enumerate(coeffs):
            if i == 0:
                # Keep approximation coefficients unchanged
                coeffs_thresholded.append(coeff)
            else:
                # Apply thresholding to detail coefficients
                thresholded = self._apply_threshold(coeff, threshold)
                coeffs_thresholded.append(thresholded)
        
        # Reconstruct image
        denoised = self.wavelet_reconstruction(coeffs_thresholded, wavelet)
        return np.clip(denoised, 0, 255).astype(np.uint8)
    
    def _apply_threshold(self, coeff: np.ndarray, threshold: float) -> np.ndarray:
        """
        Apply thresholding to wavelet coefficients.
        
        Args:
            coeff: Wavelet coefficient
            threshold: Threshold value
            
        Returns:
            Thresholded coefficient
        """
        # Soft thresholding
        coeff_thresholded = np.sign(coeff) * np.maximum(np.abs(coeff) - threshold, 0)
        return coeff_thresholded
    
    def wavelet_sharpening(self, image: np.ndarray, wavelet: str = 'db4', 
                          level: int = 3, factor: float = 2.0) -> np.ndarray:
        """
        Sharpen image using wavelet coefficients.
        
        Args:
            image: Input image
            wavelet: Wavelet type
            level: Decomposition level
            factor: Sharpening factor
            
        Returns:
            Sharpened image
        """
        # Perform wavelet decomposition
        coeffs = self.wavelet_decomposition(image, wavelet, level)
        
        # Enhance detail coefficients
        coeffs_enhanced = []
        for i, coeff in enumerate(coeffs):
            if i == 0:
                # Keep approximation coefficients unchanged
                coeffs_enhanced.append(coeff)
            else:
                # Enhance detail coefficients
                enhanced = coeff * factor
                coeffs_enhanced.append(enhanced)
        
        # Reconstruct image
        sharpened = self.wavelet_reconstruction(coeffs_enhanced, wavelet)
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    
    def wavelet_edge_detection(self, image: np.ndarray, wavelet: str = 'haar', 
                              level: int = 2) -> np.ndarray:
        """
        Detect edges using wavelet coefficients.
        
        Args:
            image: Input image
            wavelet: Wavelet type
            level: Decomposition level
            
        Returns:
            Edge detected image
        """
        # Perform wavelet decomposition
        coeffs = self.wavelet_decomposition(image, wavelet, level)
        
        # Extract edge information from detail coefficients
        edge_coeffs = []
        for i, coeff in enumerate(coeffs):
            if i == 0:
                # Zero out approximation coefficients
                edge_coeffs.append(np.zeros_like(coeff))
            else:
                # Keep detail coefficients for edge detection
                edge_coeffs.append(coeff)
        
        # Reconstruct edge image
        edges = self.wavelet_reconstruction(edge_coeffs, wavelet)
        
        # Normalize and threshold
        edges = np.abs(edges)
        edges = (edges - np.min(edges)) / (np.max(edges) - np.min(edges)) * 255
        return edges.astype(np.uint8)
    
    def wavelet_compression(self, image: np.ndarray, wavelet: str = 'db4', 
                           level: int = 3, compression_ratio: float = 0.1) -> Tuple[np.ndarray, float]:
        """
        Compress image using wavelet transform.
        
        Args:
            image: Input image
            wavelet: Wavelet type
            level: Decomposition level
            compression_ratio: Ratio of coefficients to keep
            
        Returns:
            Tuple of (compressed image, compression ratio)
        """
        # Perform wavelet decomposition
        coeffs = self.wavelet_decomposition(image, wavelet, level)
        
        # Calculate number of coefficients to keep
        total_coeffs = sum(coeff.size for coeff in coeffs)
        n_keep = int(total_coeffs * compression_ratio)
        
        # Flatten all coefficients
        all_coeffs = []
        for coeff in coeffs:
            all_coeffs.extend(coeff.flatten())
        
        # Sort by magnitude and keep top coefficients
        coeffs_flat = np.array(all_coeffs)
        sorted_indices = np.argsort(np.abs(coeffs_flat))[::-1]
        
        # Zero out coefficients below threshold
        threshold_idx = sorted_indices[n_keep]
        threshold_value = np.abs(coeffs_flat[threshold_idx])
        
        coeffs_compressed = []
        start_idx = 0
        for coeff in coeffs:
            coeff_size = coeff.size
            coeff_flat = coeffs_flat[start_idx:start_idx + coeff_size]
            
            # Apply threshold
            coeff_thresholded = np.where(np.abs(coeff_flat) >= threshold_value, coeff_flat, 0)
            coeff_compressed = coeff_thresholded.reshape(coeff.shape)
            coeffs_compressed.append(coeff_compressed)
            
            start_idx += coeff_size
        
        # Reconstruct image
        compressed = self.wavelet_reconstruction(coeffs_compressed, wavelet)
        
        # Calculate actual compression ratio
        original_size = image.nbytes
        compressed_size = sum(np.count_nonzero(coeff) for coeff in coeffs_compressed) * 8
        actual_ratio = original_size / compressed_size
        
        return np.clip(compressed, 0, 255).astype(np.uint8), actual_ratio
    
    def wavelet_fusion(self, image1: np.ndarray, image2: np.ndarray, 
                      wavelet: str = 'db4', level: int = 3, 
                      fusion_method: str = 'max') -> np.ndarray:
        """
        Fuse two images using wavelet transform.
        
        Args:
            image1: First input image
            image2: Second input image
            wavelet: Wavelet type
            level: Decomposition level
            fusion_method: Fusion method ('max', 'min', 'average')
            
        Returns:
            Fused image
        """
        # Perform wavelet decomposition on both images
        coeffs1 = self.wavelet_decomposition(image1, wavelet, level)
        coeffs2 = self.wavelet_decomposition(image2, wavelet, level)
        
        # Fuse coefficients
        coeffs_fused = []
        for i in range(len(coeffs1)):
            if fusion_method == 'max':
                fused_coeff = np.maximum(coeffs1[i], coeffs2[i])
            elif fusion_method == 'min':
                fused_coeff = np.minimum(coeffs1[i], coeffs2[i])
            elif fusion_method == 'average':
                fused_coeff = (coeffs1[i] + coeffs2[i]) / 2
            else:
                raise ValueError(f"Unknown fusion method: {fusion_method}")
            
            coeffs_fused.append(fused_coeff)
        
        # Reconstruct fused image
        fused = self.wavelet_reconstruction(coeffs_fused, wavelet)
        return np.clip(fused, 0, 255).astype(np.uint8)
    
    def wavelet_analysis(self, image: np.ndarray, wavelet: str = 'db4', 
                        level: int = 3) -> dict:
        """
        Analyze image using wavelet transform.
        
        Args:
            image: Input image
            wavelet: Wavelet type
            level: Decomposition level
            
        Returns:
            Dictionary containing analysis results
        """
        # Perform wavelet decomposition
        coeffs = self.wavelet_decomposition(image, wavelet, level)
        
        # Analyze coefficients
        analysis = {
            'wavelet': wavelet,
            'level': level,
            'approximation_energy': np.sum(coeffs[0] ** 2),
            'detail_energies': [],
            'total_coefficients': sum(coeff.size for coeff in coeffs),
            'energy_distribution': []
        }
        
        # Calculate energy for each level
        total_energy = analysis['approximation_energy']
        for i, coeff in enumerate(coeffs[1:], 1):
            level_energy = np.sum(coeff ** 2)
            analysis['detail_energies'].append(level_energy)
            total_energy += level_energy
        
        # Calculate energy distribution
        analysis['energy_distribution'] = [analysis['approximation_energy'] / total_energy]
        analysis['energy_distribution'].extend([e / total_energy for e in analysis['detail_energies']])
        
        return analysis
    
    def wavelet_watermarking(self, image: np.ndarray, watermark: np.ndarray, 
                           wavelet: str = 'db4', level: int = 2, 
                           alpha: float = 0.1) -> np.ndarray:
        """
        Embed watermark in image using wavelet transform.
        
        Args:
            image: Host image
            watermark: Watermark image
            wavelet: Wavelet type
            level: Decomposition level
            alpha: Embedding strength
            
        Returns:
            Watermarked image
        """
        # Perform wavelet decomposition on host image
        coeffs = self.wavelet_decomposition(image, wavelet, level)
        
        # Resize watermark to match approximation coefficients
        watermark_resized = cv2.resize(watermark, coeffs[0].shape[::-1])
        
        # Embed watermark in approximation coefficients
        coeffs_watermarked = coeffs.copy()
        coeffs_watermarked[0] = coeffs[0] + alpha * watermark_resized
        
        # Reconstruct watermarked image
        watermarked = self.wavelet_reconstruction(coeffs_watermarked, wavelet)
        return np.clip(watermarked, 0, 255).astype(np.uint8)
    
    def extract_watermark(self, watermarked_image: np.ndarray, original_image: np.ndarray,
                         wavelet: str = 'db4', level: int = 2, 
                         alpha: float = 0.1) -> np.ndarray:
        """
        Extract watermark from watermarked image.
        
        Args:
            watermarked_image: Watermarked image
            original_image: Original host image
            wavelet: Wavelet type
            level: Decomposition level
            alpha: Embedding strength used
            
        Returns:
            Extracted watermark
        """
        # Perform wavelet decomposition on both images
        coeffs_watermarked = self.wavelet_decomposition(watermarked_image, wavelet, level)
        coeffs_original = self.wavelet_decomposition(original_image, wavelet, level)
        
        # Extract watermark from approximation coefficients
        watermark_extracted = (coeffs_watermarked[0] - coeffs_original[0]) / alpha
        
        # Normalize watermark
        watermark_extracted = (watermark_extracted - np.min(watermark_extracted)) / \
                             (np.max(watermark_extracted) - np.min(watermark_extracted)) * 255
        
        return watermark_extracted.astype(np.uint8) 