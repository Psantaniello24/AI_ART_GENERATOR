# AI Art Generator

A Python application that generates AI art using OpenAI's DALL-E models with a user-friendly Streamlit interface.

## Example Generation

![Violet Ferrari generated with DALL-E 3](https://github.com/yourusername/ai-art-generator/raw/main/example_images/violet_ferrari.jpg)

*A stunning violet Ferrari generated using DALL-E 3 with the 3D Render style*

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
git clone https://github.com/yourusername/ai-art-generator.git
cd ai-art-generator
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

## Deployment to Streamlit Cloud

1. Push your code to GitHub:
```
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/yourusername/ai-art-generator.git
git push -u origin main
```

2. Go to [Streamlit Cloud](https://streamlit.io/cloud) and sign in with your GitHub account.

3. Click "New app" and select your GitHub repository.

4. Set the main file path to `app.py`.

5. In the "Advanced settings" section, add your OpenAI API key as a secret:
```
OPENAI_API_KEY = "your_openai_api_key_here"
```

6. Click "Deploy" and wait for your app to be deployed.

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