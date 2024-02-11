import os
import io
import pytesseract
from openai import OpenAI
from PIL import Image
from flask import Flask, request, render_template, jsonify
from dotenv import load_dotenv

app = Flask(__name__)

# Load environment variables from .env file
load_dotenv()

# Get OpenAI API key from environment variable
OpenAI.api_key = os.getenv("OPENAI_API_KEY")

# Make sure pytesseract can find the tesseract binary
pytesseract.pytesseract.tesseract_cmd = r'C:\\Users\\sanju\\tessy\\tesseract.exe'

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # Get the uploaded image
    img = request.files['image']

    # Read the image using Pillow
    image = Image.open(img)

    # Perform the OCR using pytesseract
    text = pytesseract.image_to_string(image)

    # Process the extracted text and generate a response
    response = process_text(text)

    return jsonify(response)

def process_text(text):
    # Generate a buffer message using OpenAI's API
    buffer_message = OpenAI.Completion.create(
        engine="text-davinci-003",
        prompt="Generate a buffer message: " + text,
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.7,
    )

    # Generate a story using OpenAI's API
    story = OpenAI.Completion.create(
        engine="text-davinci-003",
        prompt="Generate a story: " + text,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.7,
    )

    # Generate feedback using OpenAI's API
    feedback = OpenAI.Completion.create(
        engine="text-davinci-003",
        prompt="Generate feedback: " + text,
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.7,
    )

    # Return the generated responses
    return {
        'buffer_message': buffer_message.choices[0].text,
        'story': story.choices[0].text,
        'question': 'Does that make you feel better?',
        'feedback': feedback.choices[0].text,
    }

if __name__ == '__main__':
    app.run(debug=True)