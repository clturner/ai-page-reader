from flask import Flask, request, render_template, send_file
import boto3
import uuid
import os
from dotenv import load_dotenv
from PIL import Image, ImageOps, ImageDraw
import io
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
import cohere
import logging
from werkzeug.exceptions import HTTPException
import traceback

# Setup
load_dotenv()
cohere_api_key = os.getenv("CO_API_KEY")
co = cohere.Client(cohere_api_key)
app = Flask(__name__)

os.makedirs("static/audio", exist_ok=True)
os.makedirs("static/images", exist_ok=True)

aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_REGION")
bucket_name = os.getenv("S3_BUCKET")

s3 = boto3.client('s3', aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key, region_name=aws_region)
textract = boto3.client('textract', aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key, region_name=aws_region)
polly = boto3.client('polly', aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key, region_name=aws_region)
comprehend = boto3.client('comprehend', aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key, region_name=aws_region)

# Set up basic logging to console
logging.basicConfig(level=logging.WARNING)


def try_wrapper(func):
    def wrapped(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.exception(f"Error in {func.__name__}")
            return None
    return wrapped

@app.errorhandler(Exception)
def handle_exception(e):
    # Special case: suppress favicon.ico 404 errors (common in browsers)
    if request.path == '/favicon.ico':
        return '', 204  # No Content, don't log

    # Let Flask handle expected HTTP errors like 404, 403, etc.
    if isinstance(e, HTTPException) and e.code < 500:
        return e

    # Log unexpected errors (500+)
    logging.exception("An internal server error occurred:")
    return "Internal Server Error", 500


# Function to upload an image to S3 and return the key
@try_wrapper
def upload_to_s3(image_obj, label):
    if image_obj.mode != "RGB":
        image_obj = image_obj.convert("RGB")  # ✅ Ensure JPEG-compatible format

    buffer = io.BytesIO()
    image_obj.save(buffer, format="JPEG")
    buffer.seek(0)
    key = f"{uuid.uuid4()}_{label}.jpg"
    s3.upload_fileobj(buffer, bucket_name, key)
    return key

# Function to save an image locally as JPEG
@try_wrapper
def save_image(image_obj, label):
    if image_obj.mode != "RGB":
        image_obj = image_obj.convert("RGB")  # ✅ Convert before saving as JPEG

    filename = f"images/{label}-{uuid.uuid4()}.jpg"
    image_obj.save(os.path.join("static", filename))
    return filename

from PIL import ImageDraw

# Function to normalize lighting in a PIL image using OpenCV
@try_wrapper
def normalize_lighting(pil_img):
    # Convert PIL image to OpenCV BGR format
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to L-channel (lightness)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # Merge the channels and convert back to BGR
    merged = cv2.merge((cl, a, b))
    bgr_normalized = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    # Convert back to PIL image
    normalized_pil = Image.fromarray(cv2.cvtColor(bgr_normalized, cv2.COLOR_BGR2RGB))
    return normalized_pil

# Function to save density overlay image
@try_wrapper
def save_density_overlay(thresh_img, density, filename):
    norm_density = np.clip(density / np.max(density) * 255, 0, 255).astype(np.uint8)
    overlay = cv2.cvtColor(thresh_img, cv2.COLOR_GRAY2BGR)
    height = thresh_img.shape[0]
    
    for x in range(len(norm_density)):
        line_height = int(norm_density[x])
        y_top = max(0, height - line_height)
        y_bottom = min(height - 1, height)
        cv2.line(overlay, (x, height), (x, y_top), (0, 255, 0), 1)
    
    cv2.imwrite(filename, overlay)

# Function to apply CLAHE if needed based on average brightness and contrast
@try_wrapper
def apply_clahe_if_needed(img_gray):
    avg_brightness = np.mean(img_gray)
    contrast = np.std(img_gray)

    print(f"[Split] Avg brightness: {avg_brightness:.2f}, Contrast: {contrast:.2f}")
    if avg_brightness < 160 or contrast < 70:
        print("[Split] Applying CLAHE")
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img_gray)
    else:
        print("[Split] Skipping CLAHE")
        return img_gray


# Function to detect vertical split in an image and draw the split line if found
@try_wrapper
def detect_vertical_split_column(image, density_threshold=2, min_gap_width_ratio=0.01, debug_filename=None):
    print(f"[Split] Using density threshold: {density_threshold}, min_gap_width_ratio: {min_gap_width_ratio}")

    # === Convert to grayscale and apply CLAHE ===
    img_array_uint8 = np.array(image.convert("L")).astype(np.uint8)
    norm_img = apply_clahe_if_needed(img_array_uint8)

    if debug_filename:
        cv2.imwrite("static/images/debug_normalized.jpg", norm_img)
        print("[Split] Image lighting normalized and saved as debug_normalized.jpg")

    # === Adaptive thresholding ===
    thresh = cv2.adaptiveThreshold(
        norm_img, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=25,
        C=15
    )

    # === Morphological opening to enhance whitespace ===
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    binary = thresh > 0
    height, width = binary.shape
    print(f"[Split] Image size: {width}x{height}")

    # === Check overall text coverage ===
    text_coverage_ratio = np.mean(np.sum(binary, axis=0) > 0)
    print(f"[Split] Text coverage ratio: {text_coverage_ratio:.3f}")
    #if text_coverage_ratio > 0.99:
    #    print("[Split] Skipping column detection – likely single column")
    #    return None
    # === Compute average text density in vertical bands ===
    band_height = 50
    band_densities = []
    for y in range(0, height, band_height):
        band = binary[y:y + band_height, :]
        density = np.sum(band, axis=0)
        band_densities.append(density)
    avg_density = np.mean(band_densities, axis=0)
    # Smooth density with moving average to highlight consistent low-density regions
    window_size = 15
    smoothed_density = np.convolve(avg_density, np.ones(window_size)/window_size, mode='same')

    # === Plot density debug ===
    if debug_filename:
        bin_debug_path = debug_filename.replace(".jpg", "_binary.jpg")
        cv2.imwrite(os.path.join("static", bin_debug_path), thresh)
        plot_path = debug_filename.replace(".jpg", "_density_plot.jpg")
        plt.figure(figsize=(12, 4))
        plt.plot(avg_density)
        plt.title("Average Text Density Across Columns")
        plt.xlabel("Column Index (X position)")
        plt.ylabel("Text Density")
        plt.savefig(plot_path)
        plt.close()

    # === Detect low-density gaps ===
    low_density_cols = np.where(smoothed_density < density_threshold)[0]
    if len(low_density_cols) == 0:
        return None
    print(f"[Split] Low density column count: {len(low_density_cols)}")

    gaps = np.split(low_density_cols, np.where(np.diff(low_density_cols) > 1)[0] + 1)
    gap_centers = [int(np.mean(gap)) for gap in gaps if len(gap) > 0]
    # Ignore gaps that are too close to the left/right edge (e.g., within 50 px)
    edge_margin = int(0.05 * width)  # 5% of width
    filtered_gap_centers = [x for x in gap_centers if edge_margin < x < (width - edge_margin)]

    if not filtered_gap_centers:
        print("[Split] No valid gap found away from edges.")
        return None

    center_x = width // 2
    closest_gap_center = min(filtered_gap_centers, key=lambda x: abs(x - center_x))
    selected_gap = [g for g in gaps if int(np.mean(g)) == closest_gap_center][0]


    print(f"[Split] Widest gap width: {len(selected_gap)} pixels")
    print(f"[Split] Required min gap width: {int(width * min_gap_width_ratio)} pixels")
    print(f"[Split] Gap position: x = {int(np.mean(selected_gap))}")

    # === Check if gap is wide enough ===
    # Loosen min width for very wide images (e.g., > 3000px)
    if width > 3000:
        min_gap_width_ratio = 0.002  # allow ~8px gaps
    required_min_gap = max(int(width * min_gap_width_ratio), 10)

    if len(selected_gap) < required_min_gap:
        print(f"[Split] Gap too narrow — still drawing overlay for debug.")
        if debug_filename:
            save_density_overlay(thresh, avg_density, os.path.join("static", "images", "debug_density_overlay.jpg"))
            print(f"[Debug] Density overlay saved at static/images/debug_density_overlay.jpg")
            debug_img = image.convert("RGB")
            draw = ImageDraw.Draw(debug_img)
            draw.line((int(np.mean(selected_gap)), 0, int(np.mean(selected_gap)), height), fill="orange", width=2)
            debug_img.save(debug_filename)
        return None

    # === Final split line with bias adjustment ===
    split_x = int(np.mean(selected_gap))
    bias = -10  # Shift a bit left to avoid clipping right column
    split_x = max(0, split_x + bias)
    print(f"[Split] Adjusted split_x (with bias): {split_x}")

    # === Save debug image with split line ===
    if debug_filename:
        print(f"[Split] Saving debug image with split line at x={split_x}")
        save_density_overlay(thresh, avg_density, os.path.join("static", "images", "debug_density_overlay.jpg"))
        print(f"[Debug] Density overlay saved at static/images/debug_density_overlay.jpg")
        debug_img = image.convert("RGB")
        draw = ImageDraw.Draw(debug_img)
        draw.line((split_x, 0, split_x, height), fill="red", width=2)
        debug_img.save(debug_filename)

    print(f"[Split] Final split_x returned: {split_x}")
    return split_x

# Function to deskew an image using OpenCV
#    tweak angle_threshold=1.0 up to 1.5 or 2.0 if it's still over-rotating
@try_wrapper
def deskew_image(pil_img, angle_threshold=1.0):
    img = np.array(pil_img.convert('L'))
    img = cv2.bitwise_not(img)

    coords = np.column_stack(np.where(img > 0))
    if len(coords) == 0:
        return pil_img, False

    angle = cv2.minAreaRect(coords)[-1]
    print(f"[Deskew] Detected angle: {angle:.2f} degrees")

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) == 90.0:
        print("[Deskew] Ignoring 90-degree angle (likely false positive)")
        return pil_img, False

    if abs(angle) < angle_threshold:
        return pil_img, False

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(np.array(pil_img), M, (w, h),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(rotated), True


# Function to detect the dominant language of the text
@try_wrapper
def detect_language(text):
    response = comprehend.detect_dominant_language(Text=text)
    languages = response['Languages']
    if languages:
        return languages[0]['LanguageCode']  # e.g., 'en', 'es', 'pt'
    return 'en'  # default fallback

# Language to voice mapping for Polly
LANGUAGE_VOICE_MAP = {
    'en': 'Joanna',      # English (US)
    'es': 'Lucia',       # Spanish (Mexico)
    'pt': 'Camila',      # Portuguese (Brazil)
    'fr': 'Celine',      # French (France)
    'de': 'Vicki',       # German
    'it': 'Carla',       # Italian
    'ja': 'Mizuki',      # Japanese
    'hi': 'Aditi',       # Hindi
    'zh': 'Zhiyu',       # Chinese (Mandarin)
    # Add more as needed
}

# Split long text into chunks for Polly
@try_wrapper
def split_text(text, max_length=1400):
    import re
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_length:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# Extract text from an S3 object using Textract
@try_wrapper
def extract_text_from_image(s3_key):
    response = textract.detect_document_text(
        Document={"S3Object": {"Bucket": bucket_name, "Name": s3_key}}
    )
    lines = [block["Text"] for block in response["Blocks"] if block["BlockType"] == "LINE"]
    return "\n".join(lines)

# Function to clean OCR text for narration
@try_wrapper
def light_clean_ocr_text(text: str, wrap_speak: bool = True, max_length: int = None) -> str:
    # Replace curly apostrophes and quotes
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    
    # Escape ampersands & basic XML characters safely
    text = text.replace("&", "&amp;")
    
    # Temporarily protect SSML tags
    text = text.replace("<emphasis", "|||EMPH_OPEN|||")  
    text = text.replace("</emphasis>", "|||EMPH_CLOSE|||")
    text = text.replace("<break", "|||BREAK_OPEN|||")
    text = text.replace("/>", "|||BREAK_CLOSE|||")
    text = text.replace("<speak>", "").replace("</speak>", "")  # remove nested speak tags
    
    # Escape any other < > characters
    text = text.replace("<", "&lt;").replace(">", "&gt;")

    # Restore allowed SSML tags
    text = text.replace("|||EMPH_OPEN|||", "<emphasis")
    text = text.replace("|||EMPH_CLOSE|||", "</emphasis>")
    text = text.replace("|||BREAK_OPEN|||", "<break")
    text = text.replace("|||BREAK_CLOSE|||", "/>")

    # Remove stray OCR artifacts like isolated letters or page numbers
    text = re.sub(r'\b[nfse]\b', '', text)  # lone letters that commonly show up
    text = re.sub(r'\b\d+\.\b', '', text)   # stray page numbers or numbered bullets

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Optionally trim text to max_length for AWS Polly
    if max_length and len(text) > max_length:
        text = text[:max_length].rsplit(' ', 1)[0] + "..."

    # Remove non-printable/control characters
    text = ''.join(c for c in text if c.isprintable())
    return text if not wrap_speak else f"<speak>{text}</speak>"

# Function to clean OCR text using Cohere's 'command-r-plus' model
@try_wrapper
def clean_text_with_cohere(ocr_text: str, co) -> str:
    """
    Cleans OCR text for narration using Cohere's 'command-r-plus' model.

    This function:
    - Removes marginal numbers or stray text.
    - Fixes obvious OCR errors (e.g., gibberish or misspelled words).
    - Preserves the original structure and content (no summarizing or paraphrasing).

    Parameters:
        ocr_text (str): The raw OCR text to clean.
        co: Cohere client instance.

    Returns:
        str: The cleaned text.
    """
    prompt = (
            "You are a text-cleaning assistant.\n"
            "Your task is to clean up OCR-scanned text.\n"
            "Only do the following:\n"
            "- Fix obviously misspelled words (correct spelling only if the intended word is clear).\n"
            "- Remove stray characters or margin artifacts (e.g., page numbers, isolated digits or letters at the start or end).\n"
            "- Do NOT summarize, paraphrase, reword, or restructure anything.\n"
            "- Do NOT remove valid sentences or content.\n"
            "- Keep all punctuation and sentence boundaries intact unless corrupted.\n"
            "- Respond ONLY with the cleaned text. Do NOT explain your changes or say if you made any.\n\n"
            f"OCR TEXT:\n{ocr_text}\n\n"
            "Cleaned Text:"
    )

    try:
        response = co.generate(
            model='command-r-plus',
            prompt=prompt,
            max_tokens=1200,     # extend if your text is longer
            temperature=0.2,
            truncate=None,
        )
        return response.generations[0].text.strip()
    except Exception as e:
        print("Error from Cohere:", e)
        return ""

# Main route
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files or request.files["image"].filename == "":
            return render_template("index.html", text=None, audio=None, error="No image was uploaded.")

        try:
            file = request.files["image"]
            try:
                image = Image.open(file.stream)
                image = ImageOps.exif_transpose(image)  # Fix EXIF rotation
                print(f"[Debug] Image size after EXIF transpose: {image.size}")
                width, height = image.size
            except Exception as e:
                raise RuntimeError(f"Error loading image: {e}")

            split_pages = "split_pages" in request.form

            # === ✅ Initialize all optional vars ===
            split_left_x = None
            split_right_x = None
            left_filename = None
            right_filename = None
            debug_left = None
            debug_right = None
            left_text = ""
            right_text = ""
            full_text = ""

            if split_pages:
                try:
                    print("[Split] Two-page mode enabled – forcing vertical split")
                    mid_x = width // 2
                    raw_left = image.crop((0, 0, mid_x, height))
                    raw_right = image.crop((mid_x, 0, width, height))

                    deskewed_left, rotated_left = deskew_image(raw_left)
                    deskewed_right, rotated_right = deskew_image(raw_right)

                    split_left_x = detect_vertical_split_column(deskewed_left, density_threshold=5, min_gap_width_ratio=0.005)
                    if split_left_x:
                        debug_left = "images/debug-split_left.jpg"
                        detect_vertical_split_column(deskewed_left, debug_filename=os.path.join("static", debug_left))

                    split_right_x = detect_vertical_split_column(deskewed_right, density_threshold=5, min_gap_width_ratio=0.005)
                    if split_right_x:
                        debug_right = "images/debug-split_right.jpg"
                        detect_vertical_split_column(deskewed_right, debug_filename=os.path.join("static", debug_right))

                    height_left = deskewed_left.height
                    height_right = deskewed_right.height

                    if split_left_x:
                        left_col1 = deskewed_left.crop((0, 0, split_left_x, height_left))
                        left_col2 = deskewed_left.crop((split_left_x, 0, deskewed_left.width, height_left))
                        left_text = extract_text_from_image(upload_to_s3(left_col1, "left-col1"))
                        left_text += "\n" + extract_text_from_image(upload_to_s3(left_col2, "left-col2"))

                        left_filename_col1 = save_image(left_col1, "left-col1")
                        left_filename_col2 = save_image(left_col2, "left-col2")

                        left_filename = left_filename_col1
                    else:
                        left_text = extract_text_from_image(upload_to_s3(deskewed_left, "left"))
                        left_filename = save_image(deskewed_left, "left")

                    if split_right_x:
                        right_col1 = deskewed_right.crop((0, 0, split_right_x, height_right))
                        right_col2 = deskewed_right.crop((split_right_x, 0, deskewed_right.width, height_right))
                        right_text = extract_text_from_image(upload_to_s3(right_col1, "right-col1"))
                        right_text += "\n" + extract_text_from_image(upload_to_s3(right_col2, "right-col2"))

                        right_filename_col1 = save_image(right_col1, "right-col1")
                        right_filename_col2 = save_image(right_col2, "right-col2")

                        right_filename = right_filename_col1
                    else:
                        right_text = extract_text_from_image(upload_to_s3(deskewed_right, "right"))
                        right_filename = save_image(deskewed_right, "right")

                    full_text = f"{left_text.strip()}\n{right_text.strip()}"
                except Exception as e:
                    raise RuntimeError(f"Error processing split pages: {e}")

            else:
                try:
                    deskewed_img, rotated_img = deskew_image(image)
                    split_x = detect_vertical_split_column(deskewed_img)
                    debug_img = None

                    if rotated_img:
                        debug_img = "images/debug-single.jpg"
                        deskewed_img.save(os.path.join("static", debug_img))

                    if split_x:
                        debug_left = "images/debug-split_single.jpg"
                        detect_vertical_split_column(deskewed_img, debug_filename=os.path.join("static", debug_left))

                        height_single = deskewed_img.height
                        width_single = deskewed_img.width
                        col1 = deskewed_img.crop((0, 0, split_x, height_single))
                        col2 = deskewed_img.crop((split_x, 0, width_single, height_single))
                        full_text = extract_text_from_image(upload_to_s3(col1, "col1"))
                        full_text += "\n" + extract_text_from_image(upload_to_s3(col2, "col2"))
                        left_filename = save_image(col1, "single-col1")
                        right_filename = save_image(col2, "single-col2")
                    else:
                        single_key = upload_to_s3(deskewed_img, "single")
                        full_text = extract_text_from_image(single_key)
                        left_filename = save_image(deskewed_img, "single")

                    if not split_x and rotated_img:
                        debug_left = debug_img
                except Exception as e:
                    raise RuntimeError(f"Error processing single page: {e}")

            if not full_text.strip():
                return render_template("index.html", text=None, audio=None, error="No readable text detected in the image.")

            try:
                raw_text = light_clean_ocr_text(full_text, wrap_speak=False)
                raw_text = clean_text_with_cohere(raw_text, co)
            except Exception as e:
                raise RuntimeError(f"Text cleaning failed: {e}")

            try:
                chunks = split_text(raw_text)
                audio_filename = f"audio/speech-{uuid.uuid4()}.mp3"
                audio_path = os.path.join("static", audio_filename)

                language_code = detect_language(raw_text)
                voice_id = LANGUAGE_VOICE_MAP.get(language_code, 'Joanna')

                with open(audio_path, "wb") as out_file:
                    for chunk in chunks:
                        response = polly.synthesize_speech(
                            Text=chunk,
                            TextType="text",
                            OutputFormat="mp3",
                            VoiceId=voice_id
                        )
                        out_file.write(response["AudioStream"].read())
            except Exception as e:
                raise RuntimeError(f"Text-to-speech failed: {e}")

            if debug_left and os.path.exists(os.path.join("static", debug_left)):
                debug_left_render = debug_left
            else:
                debug_left_render = None

            if debug_right and os.path.exists(os.path.join("static", debug_right)):
                debug_right_render = debug_right
            else:
                debug_right_render = None

            return render_template(
                "index.html",
                text=raw_text,
                audio=audio_filename,
                left_image=left_filename_col1 if split_left_x else left_filename,
                right_image=right_filename_col1 if split_right_x else right_filename,
                left_image_2=left_filename_col2 if split_left_x else None,
                right_image_2=right_filename_col2 if split_right_x else None,
                debug_left=None,
                debug_right=None,

                error=None
            )

        except Exception as e:
            logging.exception("Error during image processing")
            return render_template(
                "index.html",
                text=None,
                audio=None,
                error=str(e),
                left_image=None,
                right_image=None,
                left_image_2=None,
                right_image_2=None,
                debug_left=None,
                debug_right=None
            )

    return render_template("index.html", text=None, audio=None, error=None)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
