"""
Utility functions for the AI Art Generator.
"""

import os
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from datetime import datetime
import re
import torch
import nltk

# Create required directories
def ensure_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("presets", exist_ok=True)

# Advanced NSFW filtering
def contains_nsfw_content(prompt, threshold=0.7):
    """
    More sophisticated NSFW content filter.
    Returns True if the prompt likely contains NSFW content.
    
    Args:
        prompt: The user's text prompt
        threshold: Confidence threshold for NSFW detection
    """
    # Simple word-based filtering (basic implementation)
    nsfw_terms = [
        "nude", "naked", "sex", "porn", "explicit", "nsfw", "xxx", "adult",
        "pornographic", "genitalia", "erotic", "obscene", "lewd", "sexual"
    ]
    
    # Convert to lowercase for comparison
    prompt_lower = prompt.lower()
    
    # Check for explicit NSFW terms
    for term in nsfw_terms:
        if re.search(r'\b' + re.escape(term) + r'\b', prompt_lower):
            return True
    
    # More advanced context-aware filtering could be implemented here
    # This would typically involve a pre-trained classifier
    
    return False

# Image post-processing functions
def apply_image_effects(image, effects=None):
    """
    Apply various post-processing effects to an image.
    
    Args:
        image: PIL Image to process
        effects: Dictionary of effects to apply
    
    Returns:
        Processed PIL Image
    """
    if effects is None:
        return image
    
    # Make a copy to avoid modifying the original
    processed = image.copy()
    
    # Apply brightness adjustment
    if "brightness" in effects:
        factor = float(effects["brightness"])
        enhancer = ImageEnhance.Brightness(processed)
        processed = enhancer.enhance(factor)
    
    # Apply contrast adjustment
    if "contrast" in effects:
        factor = float(effects["contrast"])
        enhancer = ImageEnhance.Contrast(processed)
        processed = enhancer.enhance(factor)
    
    # Apply saturation adjustment
    if "saturation" in effects:
        factor = float(effects["saturation"])
        enhancer = ImageEnhance.Color(processed)
        processed = enhancer.enhance(factor)
    
    # Apply sharpness adjustment
    if "sharpness" in effects:
        factor = float(effects["sharpness"])
        enhancer = ImageEnhance.Sharpness(processed)
        processed = enhancer.enhance(factor)
    
    # Apply blur
    if "blur" in effects:
        radius = float(effects["blur"])
        processed = processed.filter(ImageFilter.GaussianBlur(radius=radius))
    
    return processed

# Image resizing with proper aspect ratio
def resize_with_aspect_ratio(image, target_width=None, target_height=None):
    """
    Resize an image while maintaining its aspect ratio.
    
    Args:
        image: PIL Image to resize
        target_width: Desired width (optional)
        target_height: Desired height (optional)
    
    Returns:
        Resized PIL Image
    """
    if target_width is None and target_height is None:
        return image
    
    width, height = image.size
    
    # Calculate new dimensions
    if target_width and target_height:
        # Determine which dimension to scale by
        width_ratio = target_width / width
        height_ratio = target_height / height
        
        if width_ratio < height_ratio:
            new_width = target_width
            new_height = int(height * width_ratio)
        else:
            new_height = target_height
            new_width = int(width * height_ratio)
    
    elif target_width:
        new_width = target_width
        new_height = int(height * (target_width / width))
    
    else:  # target_height must be specified
        new_height = target_height
        new_width = int(width * (target_height / height))
    
    return image.resize((new_width, new_height), Image.LANCZOS)

# Save generation metadata
def save_generation_metadata(prompt, style, resolution, filename, model_info, params=None):
    """
    Save detailed metadata about an image generation.
    
    Args:
        prompt: The text prompt used
        style: The art style used
        resolution: The resolution setting
        filename: The output filename
        model_info: Information about the model used
        params: Additional parameters used for generation
    """
    # Load existing metadata or create new DataFrame
    metadata_file = "generation_metadata.csv"
    
    if os.path.exists(metadata_file):
        metadata = pd.read_csv(metadata_file)
    else:
        metadata = pd.DataFrame(columns=[
            "timestamp", "prompt", "style", "resolution", "filename", 
            "model", "steps", "guidance_scale", "seed", "additional_params"
        ])
    
    # Prepare the new metadata entry
    new_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "prompt": prompt,
        "style": style,
        "resolution": resolution,
        "filename": filename,
        "model": model_info.get("model_id", "unknown"),
        "steps": params.get("steps", 30) if params else 30,
        "guidance_scale": params.get("guidance_scale", 7.5) if params else 7.5,
        "seed": params.get("seed", "random") if params else "random",
        "additional_params": str(params) if params else ""
    }
    
    # Add to metadata and save
    metadata = pd.concat([metadata, pd.DataFrame([new_entry])], ignore_index=True)
    metadata.to_csv(metadata_file, index=False)

# Device detection for Torch
def get_torch_device():
    """
    Determine the best available device for torch operations.
    
    Returns:
        torch.device: The best available device (cuda, mps, or cpu)
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # For Apple Silicon (M1/M2) Macs
        return torch.device("mps")
    else:
        return torch.device("cpu")

# Download necessary NLTK data
def download_nltk_data():
    """Download required NLTK data if not already available."""
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

if __name__ == "__main__":
    # Run some basic tests
    ensure_directories()
    download_nltk_data()
    print(f"Using torch device: {get_torch_device()}")
    print(f"NSFW test: {contains_nsfw_content('a beautiful sunset')}")  # Should be False
    print(f"NSFW test: {contains_nsfw_content('nude model')}")  # Should be True 