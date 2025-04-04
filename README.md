# AI Art Generator

A Python application that generates AI art using OpenAI's DALL-E models with a user-friendly Streamlit interface.

## Example Generation

![Violet Ferrari generated with DALL-E 3](https://github.com/PSantaniello24/ai-art-generator/raw/main/example_images/violet_ferrari.png)

*A stunning violet Ferrari generated using DALL-E 3 with the 3D Render style*


## Try it at : https://ai-art-generator-4cwmsxngnpvzafhbvavvsv.streamlit.app/ 


## Features

- Text-to-image generation with DALL-E 2 and DALL-E 3
- Multiple art style presets
- Customizable resolution options
- Content filtering for NSFW prompts
- Real-time image preview
- Download generated images
- Style examples gallery
- Generation history tracking

## Local Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/ai-art-generator](https://github.com/Psantaniello24/AI_ART_GENERATOR.git
cd AI-ART-GENERATOR
```

2. Create a virtual environment and activate it:
```
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. Install the required dependencies:
```
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory with your OpenAI API key:
```
OPENAI_API_KEY=your_openai_api_key_here
```

5. Run the Streamlit app:
```
streamlit run app.py
```

## System Requirements

- Python 3.8+
- An OpenAI API key with access to DALL-E models

## Configuration

You can customize the available art styles by modifying the `ART_STYLES` dictionary in the `app.py` file.

## Credits

This project uses:
- [OpenAI DALL-E](https://openai.com/dall-e) for image generation
- [Streamlit](https://streamlit.io/) for the web interface

## License

MIT License 
