#!/usr/bin/env python3
"""
Image Compression Module
Provides various image compression algorithms and techniques.

Author: [Your Name]
Date: [Current Date]
"""

import numpy as np
import cv2
from typing import Union, Tuple, Optional
import zlib
import pickle

class ImageCompression:
    """
    Class for performing image compression operations.
    """
    
    def __init__(self):
        """Initialize the ImageCompression class."""
        pass
    
    def jpeg_compression(self, image: np.ndarray, quality: int = 50) -> Tuple[bytes, float]:
        """
        Compress image using JPEG compression.
        
        Args:
            image: Input image
            quality: JPEG quality (0-100, higher is better quality)
            
        Returns:
            Tuple of (compressed data, compression ratio)
        """
        # Encode image to JPEG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded_img = cv2.imencode('.jpg', image, encode_param)
        
        # Calculate compression ratio
        original_size = image.nbytes
        compressed_size = len(encoded_img.tobytes())
        compression_ratio = original_size / compressed_size
        
        return encoded_img.tobytes(), compression_ratio
    
    def png_compression(self, image: np.ndarray, compression_level: int = 6) -> Tuple[bytes, float]:
        """
        Compress image using PNG compression.
        
        Args:
            image: Input image
            compression_level: PNG compression level (0-9, higher is better compression)
            
        Returns:
            Tuple of (compressed data, compression ratio)
        """
        # Encode image to PNG
        encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), compression_level]
        _, encoded_img = cv2.imencode('.png', image, encode_param)
        
        # Calculate compression ratio
        original_size = image.nbytes
        compressed_size = len(encoded_img.tobytes())
        compression_ratio = original_size / compressed_size
        
        return encoded_img.tobytes(), compression_ratio
    
    def run_length_encoding(self, image: np.ndarray) -> Tuple[list, float]:
        """
        Apply Run-Length Encoding (RLE) compression.
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (RLE encoded data, compression ratio)
        """
        # Flatten image
        flat_image = image.flatten()
        
        # Run-length encoding
        encoded = []
        count = 1
        current = flat_image[0]
        
        for pixel in flat_image[1:]:
            if pixel == current:
                count += 1
            else:
                encoded.append((current, count))
                current = pixel
                count = 1
        
        encoded.append((current, count))
        
        # Calculate compression ratio
        original_size = image.nbytes
        compressed_size = len(encoded) * 8  # Approximate size
        compression_ratio = original_size / compressed_size
        
        return encoded, compression_ratio
    
    def dct_compression(self, image: np.ndarray, block_size: int = 8, 
                       compression_ratio: float = 0.1) -> Tuple[np.ndarray, float]:
        """
        Compress image using Discrete Cosine Transform (DCT).
        
        Args:
            image: Input image
            block_size: Size of DCT blocks (default: 8)
            compression_ratio: Ratio of coefficients to keep (default: 0.1)
            
        Returns:
            Tuple of (compressed image, compression ratio)
        """
        # Ensure image dimensions are multiples of block_size
        h, w = image.shape[:2]
        h_pad = ((h + block_size - 1) // block_size) * block_size
        w_pad = ((w + block_size - 1) // block_size) * block_size
        
        padded_image = np.zeros((h_pad, w_pad), dtype=np.float32)
        padded_image[:h, :w] = image.astype(np.float32)
        
        compressed = np.zeros_like(padded_image)
        
        # Process each block
        for i in range(0, h_pad, block_size):
            for j in range(0, w_pad, block_size):
                block = padded_image[i:i+block_size, j:j+block_size]
                
                # Apply DCT
                dct_block = cv2.dct(block)
                
                # Keep only top coefficients
                n_coeffs = int(block_size * block_size * compression_ratio)
                mask = np.zeros_like(dct_block)
                mask.flat[:n_coeffs] = 1
                dct_block = dct_block * mask
                
                # Apply inverse DCT
                idct_block = cv2.idct(dct_block)
                compressed[i:i+block_size, j:j+block_size] = idct_block
        
        # Crop to original size
        result = compressed[:h, :w]
        
        # Calculate compression ratio
        original_size = image.nbytes
        compressed_size = int(original_size * compression_ratio)
        compression_ratio_actual = original_size / compressed_size
        
        return np.clip(result, 0, 255).astype(np.uint8), compression_ratio_actual
    
    def wavelet_compression(self, image: np.ndarray, wavelet: str = 'haar', 
                          level: int = 3, threshold: float = 0.1) -> Tuple[np.ndarray, float]:
        """
        Compress image using wavelet transform.
        
        Args:
            image: Input image
            wavelet: Wavelet type ('haar', 'db4', etc.)
            level: Decomposition level
            threshold: Threshold for coefficient removal
            
        Returns:
            Tuple of (compressed image, compression ratio)
        """
        try:
            import pywt
        except ImportError:
            raise ImportError("PyWavelets library is required for wavelet compression")
        
        # Apply wavelet transform
        coeffs = pywt.wavedec2(image, wavelet, level=level)
        
        # Threshold coefficients
        coeffs_thresholded = []
        for i, coeff in enumerate(coeffs):
            if i == 0:
                # Approximation coefficients
                coeffs_thresholded.append(coeff)
            else:
                # Detail coefficients
                thresholded = pywt.threshold(coeff, threshold, mode='hard')
                coeffs_thresholded.append(thresholded)
        
        # Reconstruct image
        reconstructed = pywt.waverec2(coeffs_thresholded, wavelet)
        
        # Calculate compression ratio
        original_size = image.nbytes
        compressed_size = sum(coeff.nbytes for coeff in coeffs_thresholded)
        compression_ratio = original_size / compressed_size
        
        return np.clip(reconstructed, 0, 255).astype(np.uint8), compression_ratio
    
    def vector_quantization(self, image: np.ndarray, codebook_size: int = 256, 
                          block_size: int = 4) -> Tuple[np.ndarray, list, float]:
        """
        Compress image using Vector Quantization (VQ).
        
        Args:
            image: Input image
            codebook_size: Size of the codebook
            block_size: Size of image blocks
            
        Returns:
            Tuple of (compressed image, codebook, compression ratio)
        """
        # Ensure image dimensions are multiples of block_size
        h, w = image.shape[:2]
        h_pad = ((h + block_size - 1) // block_size) * block_size
        w_pad = ((w + block_size - 1) // block_size) * block_size
        
        padded_image = np.zeros((h_pad, w_pad), dtype=np.uint8)
        padded_image[:h, :w] = image
        
        # Extract blocks
        blocks = []
        for i in range(0, h_pad, block_size):
            for j in range(0, w_pad, block_size):
                block = padded_image[i:i+block_size, j:j+block_size]
                blocks.append(block.flatten())
        
        blocks = np.array(blocks)
        
        # Generate codebook using k-means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
        _, labels, codebook = cv2.kmeans(blocks.astype(np.float32), codebook_size, 
                                       None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Quantize blocks
        compressed_blocks = codebook[labels.flatten()]
        
        # Reconstruct image
        reconstructed = np.zeros((h_pad, w_pad), dtype=np.uint8)
        block_idx = 0
        for i in range(0, h_pad, block_size):
            for j in range(0, w_pad, block_size):
                block = compressed_blocks[block_idx].reshape(block_size, block_size)
                reconstructed[i:i+block_size, j:j+block_size] = block
                block_idx += 1
        
        # Crop to original size
        result = reconstructed[:h, :w]
        
        # Calculate compression ratio
        original_size = image.nbytes
        compressed_size = len(labels) * np.log2(codebook_size) / 8  # Approximate
        compression_ratio = original_size / compressed_size
        
        return result.astype(np.uint8), codebook, compression_ratio
    
    def fractal_compression(self, image: np.ndarray, block_size: int = 8, 
                          search_range: int = 16) -> Tuple[dict, float]:
        """
        Compress image using fractal compression (simplified version).
        
        Args:
            image: Input image
            block_size: Size of range blocks
            search_range: Search range for domain blocks
            
        Returns:
            Tuple of (fractal parameters, compression ratio)
        """
        # This is a simplified fractal compression implementation
        # Real fractal compression is much more complex
        
        h, w = image.shape[:2]
        range_blocks = []
        domain_blocks = []
        
        # Extract range blocks
        for i in range(0, h - block_size + 1, block_size):
            for j in range(0, w - block_size + 1, block_size):
                range_block = image[i:i+block_size, j:j+block_size]
                range_blocks.append((i, j, range_block))
        
        # Extract domain blocks (larger and downsampled)
        domain_size = block_size * 2
        for i in range(0, h - domain_size + 1, domain_size):
            for j in range(0, w - domain_size + 1, domain_size):
                domain_block = image[i:i+domain_size, j:j+domain_size]
                # Downsample domain block
                domain_block = cv2.resize(domain_block, (block_size, block_size))
                domain_blocks.append((i, j, domain_block))
        
        # Find best matches (simplified)
        fractal_params = {}
        for r_i, r_j, range_block in range_blocks:
            best_match = None
            best_error = float('inf')
            
            for d_i, d_j, domain_block in domain_blocks:
                # Calculate error (simplified)
                error = np.mean((range_block - domain_block) ** 2)
                if error < best_error:
                    best_error = error
                    best_match = (d_i, d_j)
            
            fractal_params[(r_i, r_j)] = best_match
        
        # Calculate compression ratio
        original_size = image.nbytes
        compressed_size = len(fractal_params) * 8  # Approximate
        compression_ratio = original_size / compressed_size
        
        return fractal_params, compression_ratio
    
    def calculate_psnr(self, original: np.ndarray, compressed: np.ndarray) -> float:
        """
        Calculate Peak Signal-to-Noise Ratio (PSNR).
        
        Args:
            original: Original image
            compressed: Compressed image
            
        Returns:
            PSNR value in dB
        """
        mse = np.mean((original.astype(np.float64) - compressed.astype(np.float64)) ** 2)
        if mse == 0:
            return float('inf')
        
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr
    
    def calculate_ssim(self, original: np.ndarray, compressed: np.ndarray) -> float:
        """
        Calculate Structural Similarity Index (SSIM).
        
        Args:
            original: Original image
            compressed: Compressed image
            
        Returns:
            SSIM value (0-1, higher is better)
        """
        # Simplified SSIM calculation
        mu_x = np.mean(original)
        mu_y = np.mean(compressed)
        
        sigma_x = np.std(original)
        sigma_y = np.std(compressed)
        sigma_xy = np.mean((original - mu_x) * (compressed - mu_y))
        
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        
        ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / \
               ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x ** 2 + sigma_y ** 2 + c2))
        
        return ssim 