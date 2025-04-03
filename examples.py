"""
Advanced examples for using Stable Diffusion programmatically.
These examples demonstrate how to extend the AI Art Generator with additional features.
"""

import torch
from diffusers import (
    StableDiffusionPipeline, 
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    DPMSolverMultistepScheduler
)
from PIL import Image
import numpy as np

# Example 1: Basic text-to-image generation
def basic_text_to_image(prompt, output_file="example_basic.png"):
    """
    Basic text-to-image generation with Stable Diffusion.
    """
    model_id = "runwayml/stable-diffusion-v1-5"
    
    # Initialize the pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    
    # Generate the image
    image = pipe(prompt).images[0]
    
    # Save the image
    image.save(output_file)
    print(f"Image saved to {output_file}")
    
    return image

# Example 2: Image-to-image generation
def image_to_image(
    prompt, 
    init_image_path, 
    strength=0.75, 
    output_file="example_img2img.png"
):
    """
    Generate a new image based on an initial image and a prompt.
    
    Args:
        prompt: The text prompt to guide the image generation
        init_image_path: Path to the initial image
        strength: How much to transform the initial image (0-1)
        output_file: Where to save the result
    """
    model_id = "runwayml/stable-diffusion-v1-5"
    
    # Initialize the img2img pipeline
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    
    # Load the initial image
    init_image = Image.open(init_image_path).convert("RGB")
    init_image = init_image.resize((512, 512))
    
    # Generate the new image
    image = pipe(
        prompt=prompt, 
        image=init_image, 
        strength=strength, 
        guidance_scale=7.5
    ).images[0]
    
    # Save the result
    image.save(output_file)
    print(f"Image saved to {output_file}")
    
    return image

# Example 3: Inpainting (modifying specific parts of an image)
def inpainting(
    prompt, 
    init_image_path, 
    mask_image_path, 
    output_file="example_inpaint.png"
):
    """
    Modify specific parts of an image based on a mask.
    
    Args:
        prompt: The text prompt to guide the generation
        init_image_path: Path to the initial image
        mask_image_path: Path to the mask image (white areas will be changed)
        output_file: Where to save the result
    """
    model_id = "runwayml/stable-diffusion-inpainting"
    
    # Initialize the inpainting pipeline
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    
    # Load the initial image and mask
    init_image = Image.open(init_image_path).convert("RGB")
    init_image = init_image.resize((512, 512))
    
    mask_image = Image.open(mask_image_path).convert("RGB")
    mask_image = mask_image.resize((512, 512))
    
    # Generate the new image
    image = pipe(
        prompt=prompt,
        image=init_image,
        mask_image=mask_image,
        guidance_scale=7.5
    ).images[0]
    
    # Save the result
    image.save(output_file)
    print(f"Image saved to {output_file}")
    
    return image

# Example 4: Using different model variants
def use_different_model(
    prompt, 
    model_id="dreamlike-art/dreamlike-diffusion-1.0", 
    output_file="example_model_variant.png"
):
    """
    Generate an image using a different Stable Diffusion model variant.
    
    Args:
        prompt: The text prompt to guide the generation
        model_id: The Hugging Face model ID to use
        output_file: Where to save the result
    """
    # Initialize the pipeline with a different model
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    
    # Generate the image
    image = pipe(prompt).images[0]
    
    # Save the image
    image.save(output_file)
    print(f"Image saved to {output_file}")
    
    return image

# Example 5: Using a seed for reproducible results
def reproducible_generation(
    prompt, 
    seed=42, 
    output_file="example_seeded.png"
):
    """
    Generate an image with a fixed seed for reproducible results.
    
    Args:
        prompt: The text prompt to guide the generation
        seed: The random seed to use
        output_file: Where to save the result
    """
    model_id = "runwayml/stable-diffusion-v1-5"
    
    # Initialize the pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    
    # Set the random seed
    generator = torch.Generator("cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)
    
    # Generate the image
    image = pipe(
        prompt,
        generator=generator,
        num_inference_steps=50
    ).images[0]
    
    # Save the image
    image.save(output_file)
    print(f"Image saved to {output_file}")
    
    return image

if __name__ == "__main__":
    # Example usage
    print("Running examples...")
    
    # Example 1: Basic generation
    basic_text_to_image(
        "A futuristic city with flying cars and neon lights, cyberpunk style",
    )
    
    # Note: To run the other examples, you'll need to provide paths to initial images and masks
    print("To run other examples, uncomment them and provide appropriate image paths")
    
    # Example 2: Image-to-image (uncomment and provide an initial image)
    # image_to_image(
    #     "A painting of a landscape with mountains and a lake, in the style of Bob Ross",
    #     "initial_image.jpg",
    # )
    
    # Example 3: Inpainting (uncomment and provide initial and mask images)
    # inpainting(
    #     "A red sports car",
    #     "initial_image.jpg",
    #     "mask_image.jpg",
    # )
    
    # Example 4: Different model
    # use_different_model(
    #     "A dreamlike landscape with floating islands and waterfalls",
    # )
    
    # Example 5: Reproducible generation
    # reproducible_generation(
    #     "A magical forest with glowing mushrooms and fairies",
    # ) 