# MonReader

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

MonReader is an innovative document digitization tool that converts images of book pages into audio format using advanced OCR and text-cleaning models. Designed for accessibility, it provides a seamless experience for the blind, researchers, and anyone needing fast, high-quality document scanning in bulk. MonReader detects page flips, captures high-resolution images, dewarps and enhances them, and extracts text with preserved formatting, ultimately generating audio versions of the content.

---

## Key Features

- **Page Flip Detection**: Automatically detects page flips from low-resolution camera previews to capture high-resolution images (currently processes uploaded images).
- **Advanced OCR and Text Processing**: Utilizes Tesseract OCR and Cohere's language models to extract text from images, clean it, and convert it into semantically coherent audio output.
- **Customizable Layout Options**: Supports two-column layouts and dual-page (left and right) scans via user-selectable checkboxes.
- **Accessibility Focus**: Designed to make printed material accessible, with plans for enhanced text-to-speech and multilingual support.

---

## Tech Stack

| Technology | Purpose |
|------------|---------|
| **Python** | Core programming language for processing and logic |
| **Flask**  | Web framework for the user interface |
| **AWS**    | Deployed on EC2 for scalable processing |
| **Cohere** | Language models for text cleaning and semantic processing |
| **Tesseract** | Open-source OCR engine for text extraction |

---

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/clturner/monreader.git
   cd monreader


Install Docker:Ensure Docker is installed on your system. Refer to Docker's official installation guide for detailed instructions.

Build and Run the Docker Container:
docker build -t monreader .
docker run -p 5000:5000 monreader


Access the Application:Open your browser to http://localhost:5000 to access the MonReader web interface.



Note: For AWS EC2 deployment, ensure your instance has Docker installed and appropriate IAM roles for AWS services. Refer to AWS documentation for EC2 setup.


Usage Instructions

Upload an Image:Navigate to the web interface at http://localhost:5000 and upload an image of a book page.

Select Layout Options (optional):

Check "My pages are 2 column layout" for two-column formats.
Check "This image contains two pages (left and right)" for dual-page scans.


Submit:Click Submit to process the image. MonReader will extract text and generate an audio file.

Output:Download or play the processed text and audio directly via the interface.



Example Input & Output
Example images are located in the pages/ directory, showcasing:

Single-page scans
Two-column layouts
Dual-page (left and right) scans

Upload these images to test MonReader’s OCR and audio generation capabilities. Outputs include extracted text and audio files with read-aloud versions.

Status & Roadmap
MonReader is a prototype demonstrating the potential for a mobile application. It currently processes images via a Flask web interface, with


