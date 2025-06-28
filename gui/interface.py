#!/usr/bin/env python3
"""
SmartImageSuite GUI Interface
Provides a graphical user interface for the image processing toolkit.

Author: [Your Name]
Date: [Current Date]
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import sys

# Add modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'modules'))

class SmartImageSuiteGUI:
    """
    Main GUI class for SmartImageSuite application.
    """
    
    def __init__(self):
        """Initialize the GUI application."""
        self.root = tk.Tk()
        self.root.title("SmartImageSuite - Image Processing Toolkit")
        self.root.geometry("1200x800")
        
        # Initialize variables
        self.current_image_path = None
        self.processed_image = None
        
        # Setup the interface
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface components."""
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Create menu bar
        self.create_menu()
        
        # Create toolbar
        self.create_toolbar(main_frame)
        
        # Create main content area
        self.create_content_area(main_frame)
        
        # Create status bar
        self.create_status_bar(main_frame)
        
    def create_menu(self):
        """Create the application menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Image", command=self.open_image)
        file_menu.add_command(label="Save Image", command=self.save_image)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Processing menu
        process_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Processing", menu=process_menu)
        process_menu.add_command(label="Intensity Transform", command=self.intensity_transform)
        process_menu.add_command(label="Spatial Filters", command=self.spatial_filters)
        process_menu.add_command(label="Frequency Filters", command=self.frequency_filters)
        process_menu.add_command(label="Restoration", command=self.restoration)
        process_menu.add_command(label="Color Processing", command=self.color_processing)
        process_menu.add_command(label="Compression", command=self.compression)
        process_menu.add_command(label="Wavelet Tools", command=self.wavelet_tools)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        
    def create_toolbar(self, parent):
        """Create the toolbar with common actions."""
        toolbar = ttk.Frame(parent)
        toolbar.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(toolbar, text="Open", command=self.open_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="Save", command=self.save_image).pack(side=tk.LEFT, padx=5)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        ttk.Button(toolbar, text="Reset", command=self.reset_image).pack(side=tk.LEFT, padx=5)
        
    def create_content_area(self, parent):
        """Create the main content area with image display and controls."""
        # Left panel for controls
        control_panel = ttk.LabelFrame(parent, text="Processing Controls", padding="10")
        control_panel.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Right panel for image display
        image_panel = ttk.LabelFrame(parent, text="Image Display", padding="10")
        image_panel.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Add some placeholder controls
        ttk.Label(control_panel, text="Select processing method:").pack(anchor=tk.W)
        self.processing_var = tk.StringVar(value="intensity")
        processing_combo = ttk.Combobox(control_panel, textvariable=self.processing_var, 
                                       values=["Intensity Transform", "Spatial Filters", 
                                              "Frequency Filters", "Restoration", 
                                              "Color Processing", "Compression", "Wavelet Tools"])
        processing_combo.pack(fill=tk.X, pady=5)
        
        ttk.Button(control_panel, text="Apply Processing", command=self.apply_processing).pack(fill=tk.X, pady=10)
        
        # Image display area
        self.image_label = ttk.Label(image_panel, text="No image loaded", 
                                    background="lightgray", anchor=tk.CENTER)
        self.image_label.pack(expand=True, fill=tk.BOTH)
        
    def create_status_bar(self, parent):
        """Create the status bar."""
        status_bar = ttk.Frame(parent)
        status_bar.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.status_label = ttk.Label(status_bar, text="Ready")
        self.status_label.pack(side=tk.LEFT)
        
    def open_image(self):
        """Open an image file."""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.gif")]
        )
        if file_path:
            self.current_image_path = file_path
            self.status_label.config(text=f"Loaded: {os.path.basename(file_path)}")
            # TODO: Load and display image
            
    def save_image(self):
        """Save the processed image."""
        if self.processed_image is None:
            messagebox.showwarning("Warning", "No processed image to save")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Save Image",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        if file_path:
            # TODO: Save image
            self.status_label.config(text=f"Saved: {os.path.basename(file_path)}")
            
    def reset_image(self):
        """Reset to original image."""
        if self.current_image_path:
            # TODO: Reset to original image
            self.status_label.config(text="Image reset to original")
            
    def apply_processing(self):
        """Apply the selected processing method."""
        method = self.processing_var.get()
        if not self.current_image_path:
            messagebox.showwarning("Warning", "Please load an image first")
            return
            
        # TODO: Apply processing based on selected method
        self.status_label.config(text=f"Applied {method}")
        
    def intensity_transform(self):
        """Open intensity transform dialog."""
        messagebox.showinfo("Info", "Intensity Transform module - Coming soon!")
        
    def spatial_filters(self):
        """Open spatial filters dialog."""
        messagebox.showinfo("Info", "Spatial Filters module - Coming soon!")
        
    def frequency_filters(self):
        """Open frequency filters dialog."""
        messagebox.showinfo("Info", "Frequency Filters module - Coming soon!")
        
    def restoration(self):
        """Open restoration dialog."""
        messagebox.showinfo("Info", "Restoration module - Coming soon!")
        
    def color_processing(self):
        """Open color processing dialog."""
        messagebox.showinfo("Info", "Color Processing module - Coming soon!")
        
    def compression(self):
        """Open compression dialog."""
        messagebox.showinfo("Info", "Compression module - Coming soon!")
        
    def wavelet_tools(self):
        """Open wavelet tools dialog."""
        messagebox.showinfo("Info", "Wavelet Tools module - Coming soon!")
        
    def show_about(self):
        """Show about dialog."""
        messagebox.showinfo("About SmartImageSuite", 
                           "SmartImageSuite v1.0\n"
                           "A comprehensive image processing and analysis toolkit\n\n"
                           "Developed for ASU Image Processing Course")
        
    def run(self):
        """Start the GUI application."""
        self.root.mainloop() 