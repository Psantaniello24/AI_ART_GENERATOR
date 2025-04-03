import streamlit as st
import torch
from PIL import Image
import pandas as pd
import os
import time
import nltk
from nltk.corpus import stopwords
from datetime import datetime
import io
import re
import requests
from io import BytesIO
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file (local development)
load_dotenv()

# Create directories if they don't exist
os.makedirs("outputs", exist_ok=True)

# Download NLTK data for content filtering
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Define art styles
ART_STYLES = {
    "Photorealistic": "photorealistic, highly detailed, sharp focus, 8k",
    "Anime": "anime style, vibrant, detailed, studio ghibli, hayao miyazaki",
    "Oil Painting": "oil painting, textured, detailed brushstrokes, artistic, canvas",
    "Watercolor": "watercolor painting, soft colors, flowing, artistic",
    "Pixel Art": "pixel art, 8-bit style, retro game art",
    "3D Render": "3d render, octane render, high detail, sharp, CGI",
    "Comic Book": "comic book style, cel shaded, inked, vibrant colors",
    "Sketch": "pencil sketch, hand-drawn, detailed linework, black and white",
    "Abstract": "abstract art, non-representational, geometric shapes, vibrant colors",
    "Digital Art": "digital art, vivid colors, detailed, illustrative style"
}

# DALL-E models
DALLE_MODELS = {
    "DALL-E 3": "dall-e-3",
    "DALL-E 2": "dall-e-2"
}

# NSFW content filter words
def contains_nsfw(prompt):
    """Check if the prompt contains potentially NSFW content"""
    nsfw_terms = ["nude", "naked", "sex", "porn", "explicit", "nsfw", "xxx", "adult"]
    
    # Convert to lowercase for comparison
    prompt_lower = prompt.lower()
    
    # Check for explicit NSFW terms
    for term in nsfw_terms:
        if term in prompt_lower:
            return True
            
    return False

# Load generation history from CSV if exists
def load_history():
    if os.path.exists("generation_history.csv"):
        return pd.read_csv("generation_history.csv")
    else:
        return pd.DataFrame(columns=["timestamp", "prompt", "style", "model", "resolution", "filename"])

# Save generation to history
def save_to_history(prompt, style, model, resolution, filename):
    history = load_history()
    new_row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "prompt": prompt,
        "style": style,
        "model": model,
        "resolution": resolution,
        "filename": filename
    }
    history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)
    history.to_csv("generation_history.csv", index=False)

# Function to generate image using OpenAI's DALL-E API
def generate_image_dalle(client, prompt, style_prompt, model, size, quality="standard"):
    # Combine prompt with style
    full_prompt = f"{prompt}, {style_prompt}"
    
    # Run prediction with DALL-E API
    response = client.images.generate(
        model=model,
        prompt=full_prompt,
        size=size,
        quality=quality,
        n=1
    )
    
    # Get the image URL from the response
    image_url = response.data[0].url
    
    # Download the image
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    
    return image

# Function to get API key from environment or Streamlit secrets
def get_api_key():
    # First try to get from Streamlit secrets (for cloud deployment)
    try:
        return st.secrets["OPENAI_API_KEY"]
    except:
        # If not in Streamlit secrets, try environment variable (local dev)
        return os.getenv("OPENAI_API_KEY")

# Streamlit UI
def main():
    st.set_page_config(page_title="AI Art Generator", layout="wide")
    
    # Get API key from various sources
    api_key = get_api_key()
    
    # Check for OpenAI API token
    if not api_key and 'openai_api' not in st.session_state:
        # API key input
        with st.sidebar:
            st.header("API Setup")
            api_key = st.text_input("Enter OpenAI API Key", 
                                     type="password",
                                     help="Get an API key from platform.openai.com")
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
                st.session_state['openai_api'] = api_key
                st.success("API key set successfully!")
                st.rerun()
            
            st.markdown("Don't have an API key? [Sign up for OpenAI](https://platform.openai.com/signup)")
            return
    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key or st.session_state.get('openai_api', ''))
    
    # Title and description
    st.title("AI Art Generator")
    st.markdown("Create stunning AI-generated art using DALL-E")
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("Image Settings")
        
        # Text prompt
        prompt = st.text_area("Enter your text prompt", 
                              help="Describe what you want to see in the image")
        
        # DALL-E model selection
        model_name = st.selectbox("Select DALL-E model", list(DALLE_MODELS.keys()))
        model = DALLE_MODELS[model_name]
        
        # Art style selection
        style = st.selectbox("Select art style", list(ART_STYLES.keys()))
        
        # Resolution options
        resolution_options = {
            "1024 x 1024 (Square)": "1024x1024",
            "1792 x 1024 (Landscape)": "1792x1024",
            "1024 x 1792 (Portrait)": "1024x1792",
        }
        
        # Show resolution options only for DALL-E 3
        if model == "dall-e-3":
            resolution = st.selectbox("Select resolution", list(resolution_options.keys()))
            size = resolution_options[resolution]
        else:
            # For DALL-E 2, only 1024x1024 is available
            st.text("Resolution: 1024x1024 (fixed for DALL-E 2)")
            resolution = "1024 x 1024 (Square)"
            size = "1024x1024"
        
        # Quality options (only for DALL-E 3)
        if model == "dall-e-3":
            quality = st.selectbox("Image quality", ["standard", "hd"])
        else:
            quality = "standard"
        
        # Generate button
        generate_button = st.button("Generate Image", type="primary")
    
    # Main area for image display
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Display area for generated image
        if "generated_image" in st.session_state:
            st.image(st.session_state.generated_image, caption="Generated Image", use_column_width=True)
            
            # Save the image to a buffer
            img_buffer = io.BytesIO()
            st.session_state.generated_image.save(img_buffer, format="PNG")
            
            # Add download button
            st.download_button(
                label="Download Image",
                data=img_buffer.getvalue(),
                file_name=f"aiart_{int(time.time())}.png",
                mime="image/png"
            )
    
    with col2:
        # Style examples gallery
        st.header("Style Examples")
        for i, (style_name, _) in enumerate(list(ART_STYLES.items())[:5]):  # Show first 5 styles
            st.markdown(f"**{style_name}**: {ART_STYLES[style_name]}")
        
        # History section
        st.header("Generation History")
        history = load_history()
        if not history.empty:
            st.dataframe(
                history[["timestamp", "prompt", "style", "model"]].tail(5),
                use_container_width=True
            )
    
    # Logic for generating images
    if generate_button:
        if not prompt:
            st.error("Please enter a text prompt")
        elif contains_nsfw(prompt):
            st.error("Your prompt appears to contain NSFW content. Please modify your request.")
        else:
            try:
                # Show loading spinner
                with st.spinner("Generating your artwork..."):
                    # Get style prompt
                    style_prompt = ART_STYLES[style]
                    
                    # Generate image using DALL-E
                    image = generate_image_dalle(
                        client,
                        prompt, 
                        style_prompt,
                        model,
                        size,
                        quality
                    )
                    
                    # Save image (if not on Streamlit Cloud where filesystem is read-only)
                    try:
                        timestamp = int(time.time())
                        filename = f"outputs/aiart_{timestamp}.png"
                        image.save(filename)
                        
                        # Save to history
                        save_to_history(prompt, style, model_name, resolution, filename)
                    except:
                        # We might be on Streamlit Cloud where filesystem is read-only
                        pass
                    
                    # Update session state
                    st.session_state.generated_image = image
                    
                    # Force page refresh to show the new image
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Error generating image: {str(e)}")

if __name__ == "__main__":
    main() 