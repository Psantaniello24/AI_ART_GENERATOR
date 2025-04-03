#!/usr/bin/env python
"""
Launch script for the AI Art Generator.
Provides different options for launching the application.
"""

import os
import sys
import argparse
import subprocess
import webbrowser
from utils import ensure_directories, download_nltk_data

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="AI Art Generator launch options")
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=8501, 
        help="Port to run the Streamlit server on (default: 8501)"
    )
    
    parser.add_argument(
        "--browser", 
        action="store_true", 
        help="Automatically open a browser window"
    )
    
    parser.add_argument(
        "--headless", 
        action="store_true", 
        help="Run in headless mode (no browser)"
    )
    
    parser.add_argument(
        "--setup", 
        action="store_true", 
        help="Only run initial setup (download models, etc.)"
    )
    
    parser.add_argument(
        "--examples", 
        action="store_true", 
        help="Run the examples script to generate sample images"
    )
    
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Run with debug logging enabled"
    )
    
    return parser.parse_args()

def setup():
    """Run initial setup tasks."""
    print("Running initial setup...")
    
    # Create necessary directories
    ensure_directories()
    
    # Download NLTK data
    download_nltk_data()
    
    # Optionally download model weights
    # This will depend on the specific implementation
    # For example, using the Hugging Face Hub
    try:
        import torch
        from diffusers import StableDiffusionPipeline
        
        print("Pre-downloading model weights (this may take a while)...")
        model_id = "runwayml/stable-diffusion-v1-5"
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        print("Model weights downloaded successfully!")
    except Exception as e:
        print(f"Error pre-downloading model weights: {str(e)}")
        print("The model will be downloaded when you first run the application.")
    
    print("Setup complete!")

def run_app(args):
    """Run the Streamlit application with the specified arguments."""
    # Set environment variables for Streamlit
    os.environ["STREAMLIT_SERVER_PORT"] = str(args.port)
    
    if args.debug:
        os.environ["STREAMLIT_LOGGER_LEVEL"] = "debug"
    
    # Build the Streamlit command
    streamlit_cmd = [
        "streamlit", "run", "app.py",
        "--server.port", str(args.port),
        "--server.headless", str(args.headless)
    ]
    
    # Launch the browser if requested
    if args.browser and not args.headless:
        # Wait a moment for the server to start
        webbrowser.open(f"http://localhost:{args.port}")
    
    # Run Streamlit
    print(f"Starting AI Art Generator on port {args.port}...")
    subprocess.run(streamlit_cmd)

def run_examples():
    """Run the examples script."""
    print("Running examples script...")
    subprocess.run(["python", "examples.py"])

def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Handle setup only
    if args.setup:
        setup()
        return
    
    # Handle examples only
    if args.examples:
        run_examples()
        return
    
    # Always run setup to ensure directories exist
    ensure_directories()
    
    # Run the app
    run_app(args)

if __name__ == "__main__":
    main() 