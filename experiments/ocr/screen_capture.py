"""EXPERIMENTAL: unfinished OCR/Donut prototype; not used by main.py."""

import ctypes
from PIL import ImageGrab
import numpy as np
import cv2
import pytesseract
try:
    import easyocr
    easyocr_available = True
except ImportError:
    easyocr_available = False
import time
import os
import psutil
import datetime
import re

# Set path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Create a log file
with open("ocr_log.txt", "a") as log:
    log.write(f"\n\n--- OCR Session Started: {datetime.datetime.now()} ---\n")

# Set process priority to below normal to avoid detection
try:
    process = psutil.Process(os.getpid())
    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    with open("ocr_log.txt", "a") as log:
        log.write("Process priority set to below normal\n")
except Exception as e:
    with open("ocr_log.txt", "a") as log:
        log.write(f"Failed to set process priority: {str(e)}\n")

# Change process name in Windows task manager
try:
    ctypes.windll.kernel32.SetConsoleTitleW("System Service")
    with open("ocr_log.txt", "a") as log:
        log.write("Console title changed to 'System Service'\n")
except Exception as e:
    with open("ocr_log.txt", "a") as log:
        log.write(f"Failed to change console title: {str(e)}\n")

# Minimal preprocessing function for OCR
def advanced_preprocess(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize to make text larger for OCR
    scale_percent = 150
    width = int(gray.shape[1] * scale_percent / 100)
    height = int(gray.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(gray, dim, interpolation=cv2.INTER_LINEAR)
    # Contrast enhancement
    eq = cv2.equalizeHist(resized)
    # Denoise (median filter is good for text)
    denoised = cv2.medianBlur(eq, 3)
    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 15, 8)
    # Morphological opening to remove small noise
    kernel = np.ones((2,2), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    return opened

def clean_ocr_text(text):
    """
    Clean OCR text to improve readability and remove artifacts.
    
    Args:
        text (str): Raw OCR text from pytesseract
        
    Returns:
        str: Cleaned and formatted text
    """
    # Step 1: Remove non-printable and problematic characters
    # Keep only ASCII characters and common punctuation
    cleaned = ''.join(char for char in text if ord(char) < 128)
    
    # Step 2: Normalize whitespace (but preserve paragraph breaks)
    cleaned = re.sub(r' +', ' ', cleaned)  # Multiple spaces to single space
    cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned)  # Preserve paragraph breaks
    
    # Step 3: Fix common OCR errors and normalize punctuation
    replacements = {
        # Common OCR errors
        'l1': 'n',
        'rn': 'm',
        '0': 'o',
        '1': 'l',
        # Programming symbols
        '[ ]': '[]',
        '( )': '()',
        '{ }': '{}',
        # Normalize quotes
        '"': '"',
        '"': '"',
        ''': "'",
        ''': "'",
    }
    
    for error, correction in replacements.items():
        cleaned = cleaned.replace(error, correction)
    
    # Step 4: Fix spacing around punctuation
    cleaned = re.sub(r'\s+([.,;:!?)])', r'\1', cleaned)
    cleaned = re.sub(r'([({[<])\s+', r'\1', cleaned)
    
    # Step 5: Remove common UI artifacts
    ui_patterns = [
        r'Search\s+results',
        r'All\s+Bookmarks',
        r'Submit',
        r'Run',
        r'Test\s+Result',
        r'Case\s+\d+',
        r'\d+\s+online',
        r'Saved',
        r'Ln\s+\d+,\s+Col\s+\d+',
    ]
    
    for pattern in ui_patterns:
        cleaned = re.sub(pattern, '', cleaned)
    
    # Step 6: Fix line breaks in sentences
    cleaned = re.sub(r'([a-z,;:])\n([a-z])', r'\1 \2', cleaned)
    
    # Step 7: Detect and format code blocks
    if re.search(r'class\s+\w+[:(]|def\s+\w+\s*\(|function\s+\w+\s*\(', cleaned):
        # Improve code indentation
        lines = cleaned.split('\n')
        in_code_block = False
        for i, line in enumerate(lines):
            if re.search(r'class\s+\w+[:(]|def\s+\w+\s*\(|function\s+\w+\s*\(', line):
                in_code_block = True
            if in_code_block and line.strip() and i > 0:
                # Add proper indentation to code lines
                if 'def ' in line or 'class ' in line:
                    lines[i] = '    ' + line.strip()
                else:
                    lines[i] = '        ' + line.strip()
        cleaned = '\n'.join(lines)
    
    # Step 8: Format problem statements
    if "Example" in cleaned:
        cleaned = re.sub(r'(Example\s+\d+:)', r'\n\n\1', cleaned)
    
    if "Input:" in cleaned:
        cleaned = re.sub(r'(Input:)', r'\n\1', cleaned)
    
    if "Output:" in cleaned:
        cleaned = re.sub(r'(Output:)', r'\n\1', cleaned)
    
    return cleaned.strip()

def extract_main_context(text):
    """
    Extract main context (questions/statements) from OCR text.
    Returns only lines likely to be questions or main statements.
    """
    lines = text.split('\n')
    candidates = [line for line in lines if '?' in line or re.match(r'^(who|what|when|where|why|how)\b', line.lower())]
    # Remove short/noisy lines
    candidates = [line for line in candidates if len(line.strip()) > 10 and sum(c.isalnum() for c in line) > 5]
    if candidates:
        return '\n'.join(candidates)
    # Fallback: longest line with enough alphanumeric content
    valid_lines = [line for line in lines if len(line.strip()) > 10 and sum(c.isalnum() for c in line) > 5]
    if valid_lines:
        return max(valid_lines, key=len)
    return ''



# Save debug images
def save_debug_image(original, processed, iteration):
    debug_dir = "ocr_debug"
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
    
    timestamp = int(time.time())
    cv2.imwrite(f"{debug_dir}/original_{timestamp}_{iteration}.jpg", original)
    cv2.imwrite(f"{debug_dir}/processed_{timestamp}_{iteration}.jpg", processed)
    
    with open("ocr_log.txt", "a") as log:
        log.write(f"Debug images saved for iteration {iteration}\n")

def capture_and_ocr(x1=0, y1=0, x2=1920, y2=1080, interval=2):
    """
    Captures screen region and performs OCR
    
    Args:
        x1, y1: Top-left coordinates of capture region
        x2, y2: Bottom-right coordinates of capture region
        interval: Time between captures in seconds
    """
    with open("ocr_log.txt", "a") as log:
        log.write(f"Starting OCR capture for full screen: ({x1},{y1}) to ({x2},{y2})\n")
    
    iteration = 0
    
    try:
        # Keep console visible for debugging
        # ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)
        
        while True:
            iteration += 1
            start_time = time.time()
            
            with open("ocr_log.txt", "a") as log:
                log.write(f"\nIteration {iteration} started at {datetime.datetime.now()}\n")
            
            print(f"Capturing screen - iteration {iteration}")
            
            # Capture full screen
            screenshot = ImageGrab.grab(bbox=(x1, y1, x2, y2))
            
            # Convert to numpy array for OpenCV processing
            img_np = np.array(screenshot)
            

            # Apply advanced processing
            processed_img = advanced_preprocess(img_np)

            # Save debug images every 5 iterations
            if iteration % 5 == 0:
                save_debug_image(img_np, processed_img, iteration)

            # OCR processing with custom configuration (single block mode)
            custom_config = r'--oem 3 --psm 6'
            raw_text = pytesseract.image_to_string(processed_img, config=custom_config)
            cleaned_text = clean_ocr_text(raw_text)
            main_context = extract_main_context(cleaned_text)

            # Write both raw and filtered results to file
            with open("ocr_results.txt", "w") as f:
                f.write("--- RAW OCR ---\n")
                f.write(cleaned_text)
                f.write("\n\n--- MAIN CONTEXT ---\n")
                f.write(main_context)

            # Print a preview of the main context
            print(f"Main Context Preview: {main_context[:100].replace(chr(10), ' ')}...")

            # Log the results
            with open("ocr_log.txt", "a") as log:
                log.write(f"OCR Text Length: {len(cleaned_text)} characters\n")
                log.write(f"Main Context Length: {len(main_context)} characters\n")
                log.write(f"First 100 chars: {main_context[:100].replace(chr(10), ' ')}\n")
                log.write(f"Processing time: {time.time() - start_time:.2f} seconds\n")
            
            # Optional: Wait for interval
            time.sleep(interval)
            
    except KeyboardInterrupt:
        with open("ocr_log.txt", "a") as log:
            log.write(f"OCR capture stopped after {iteration} iterations\n")
        print(f"OCR capture stopped after {iteration} iterations")



# Static image Donut processing
def process_static_image_with_donut(image_path):
    from PIL import Image
    from transformers import DonutProcessor, VisionEncoderDecoderModel
    import torch
    try:
        with open("ocr_log.txt", "a") as log:
            log.write(f"Processing static image with Donut: {image_path}\n")
        print(f"Processing static image with Donut: {image_path}")
        # Load Donut model and processor (DocVQA-finetuned)
        processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
        model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        # Load and resize image
        image = Image.open(image_path).convert("RGB")
        image = image.resize((1280, 960))
        # Use the correct DocVQA prompt
        task_prompt = "<s_docvqa><s_question>What is the main question in this image?</s_question><s_answer>"
        inputs = processor(image, task_prompt, return_tensors="pt").to(device)
        # Generate output
        outputs = model.generate(**inputs, max_length=512, num_beams=1)
        print(f"Raw output tensor: {outputs}")
        # Decode with and without skipping special tokens
        decoded_with_special = processor.batch_decode(outputs, skip_special_tokens=False)[0]
        decoded_no_special = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        print(f"Decoded (with special tokens): {decoded_with_special}")
        print(f"Decoded (skip special tokens): {decoded_no_special}")
        # Save both results to ocr_results.txt
        with open("ocr_results.txt", "w") as f:
            f.write("--- DONUT OUTPUT (with special tokens) ---\n")
            f.write(decoded_with_special)
            f.write("\n\n--- DONUT OUTPUT (skip special tokens) ---\n")
            f.write(decoded_no_special)
        print(f"Donut Output Preview: {decoded_no_special[:200].replace(chr(10), ' ')}...")
        # Log everything to ocr_log.txt
        with open("ocr_log.txt", "a") as log:
            log.write(f"Raw output tensor: {outputs}\n")
            log.write(f"Decoded (with special tokens): {decoded_with_special}\n")
            log.write(f"Decoded (skip special tokens): {decoded_no_special}\n")
            log.write(f"Donut Output Length: {len(decoded_no_special)} characters\n")
            log.write(f"First 200 chars: {decoded_no_special[:200].replace(chr(10), ' ')}\n")
    except Exception as e:
        print(f"Donut extraction error: {e}")
        with open("ocr_results.txt", "w") as f:
            f.write("--- DONUT OUTPUT ---\n")
            f.write(f"ERROR: {e}\n")
        with open("ocr_log.txt", "a") as log:
            log.write(f"Donut extraction error: {e}\n")

# Find the latest original image in ocr_debug
import glob
# Comment out previous OCR pipeline
# image_files = glob.glob("ocr_debug/original_*.jpg")
# if image_files:
#     latest_image = max(image_files, key=os.path.getctime)
#     process_static_image(latest_image)
# else:
#     print("No original images found in ocr_debug folder.")

# Use Donut for static image OCR
sample_dir = os.path.join(os.path.dirname(__file__), "samples")
image_files = glob.glob(os.path.join(sample_dir, "original_*.jpg"))
if image_files:
    latest_image = max(image_files, key=os.path.getctime)
    process_static_image_with_donut(latest_image)
else:
    print("No original images found in ocr_debug folder.")
