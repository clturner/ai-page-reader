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

## Setup Instructions 🚀

> Get *MonReader* up and running in just a few steps!

[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![AWS](https://img.shields.io/badge/AWS-EC2-orange?style=flat&logo=amazon-aws&logoColor=white)](https://aws.amazon.com/ec2/)

1. **Install Docker** 🐳:
   Ensure *Docker* is installed on your system. Follow the [official Docker installation guide](https://docs.docker.com/get-docker/) for step-by-step instructions.

2. **Build and Run the Docker Container**:
   Use the following commands to build and launch *MonReader*:
   ```bash
   docker build -t monreader .
   docker run -p 5000:5000 monreader

   ## Access the Application 🌐

[![Flask](https://img.shields.io/badge/Flask-2.0+-green?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Accessibility](https://img.shields.io/badge/Accessibility-Focused-blue?style=flat&logo=accessible-icon)](https://www.w3.org/WAI/)

Open your browser and navigate to `http://localhost:5000` to explore the *MonReader* web interface.

> **Note** ☁️: For AWS EC2 deployment, ensure your instance has *Docker* installed and the necessary IAM roles configured. Check the [AWS EC2 documentation](https://docs.aws.amazon.com/ec2/) for setup details.

---

## Usage Instructions 📖

Follow these steps to digitize your book pages with *MonReader*:

1. **Upload an Image** 📸:
   Visit `http://localhost:5000` and *upload* an image of a book page via the web interface.

2. **Select Layout Options** (optional):
   Customize processing with these checkboxes:

   | Option | Description |
   |--------|-------------|
   | **My pages are 2 column layout** | Select for pages with two-column text formats. |
   | **This image contains two pages (left and right)** | Select for scans of open books with left and right pages. |

3. **Submit** 🚀:
   Click *Submit* to process the image. *MonReader* will extract text and generate an audio file.

4. **Output** 🎧:
   *Download* or *play* the extracted text and audio directly from the interface.

> **Pro Tip** 🧪: Experiment with different page layouts to see how *MonReader* handles complex formats!

---

## Example Input & Output 📚

Example images are located in the `pages/` directory, demonstrating:
- **Single-page scans**: Standard one-page layouts.
- **Two-column layouts**: Pages with dual-column text.
- **Dual-page (left and right) scans**: Open book images with both pages visible.

*Upload these images* to test *MonReader*’s OCR and audio generation capabilities. The output includes:
- **Extracted Text**: Cleaned and formatted text from the page.
- **Audio Files**: Read-aloud versions for accessibility.

---

## Status & Roadmap 🌟

*MonReader* is currently a **prototype**, showcasing the potential for a mobile application. It processes images via a *Flask-based web interface*, with core features like:
- Page flip detection
- OCR text extraction
- Audio generation

> **Vision** 🌍: Transform *MonReader* into a seamless mobile app for real-time document digitization and audio conversion, empowering accessibility for all!
