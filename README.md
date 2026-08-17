# 📄 Document Scanner

A browser-based document scanner built with **OpenCV** and **Streamlit**. Upload a photo (or snap one with your camera), and the app automatically detects the document's edges, corrects the perspective, and cleans it up into a crisp, print-ready scan — no physical scanner required.

![Demo placeholder](docs/demo.gif)
*Replace this with a screenshot or GIF of the app in action.*

---

## ✨ Features

- **Automatic document detection** — finds the document in a photo even against cluttered backgrounds, using GrabCut foreground segmentation and Canny edge detection
- **Perspective correction** — warps the detected corners into a flat, top-down view via a 4-point homography transform
- **Image enhancement** — adaptive thresholding and morphological cleanup to produce a sharp, high-contrast, print-style output
- **Debug/visualization mode** — toggle to see every step of the pipeline (background removal, edge map, candidate contours, detected corners, and each enhancement stage)
- **Export to PNG or PDF** — download the final scan in either format
- **Upload or capture** — works with an uploaded image or a live camera shot, right in the browser

---

## 🧠 How it works

The pipeline runs in two stages:

**1. Detection & Scanning** (`scan/scanner_module.py`)
1. Resize the input image for consistent processing speed
2. Apply morphological closing to reduce noise
3. Remove the background using OpenCV's **GrabCut** algorithm
4. Convert to grayscale, blur, and run **Canny edge detection**
5. Find contours and identify the 4-point quadrilateral most likely to be the document (falling back to a convex hull or minimum-area rectangle if a clean quad isn't found)
6. Order the four corners and compute a **perspective transform** to warp the document into a flat rectangle

**2. Enhancement** (`scan/improvement_module.py`)
1. Convert the scanned result to grayscale
2. Apply Gaussian blur to reduce noise
3. Apply **adaptive thresholding** to produce a clean, high-contrast black-and-white output
4. Apply a morphological opening to remove small artifacts

Enable **debug mode** in the app to see each of these steps visualized side by side.

---

## 🖼️ Debug Mode Preview

![Debug mode placeholder](docs/debug-mode.png)
*Replace this with a screenshot of the debug pipeline expander.*

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+

### Installation

```bash
git clone https://github.com/AminSorena32/Document-scanner.git
cd Document-scanner
pip install -r requirements.txt
```

### Run the app

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints in your terminal (usually `http://localhost:8501`).

---

## 🌐 Live Demo

*Add your Streamlit Community Cloud link here once deployed:*
👉 [Try it live](#)

---

## 🛠️ Tech Stack

- **Python**
- **OpenCV** — image processing, edge detection, perspective transform
- **Streamlit** — web app interface
- **NumPy** — array/image manipulation
- **Pillow** — PDF export

---

## ⚠️ Known Limitations

- Works best against a reasonably contrasting background; very cluttered or low-contrast scenes can affect corner detection
- The enhancement step binarizes the output (black & white), which is ideal for text documents but will lose color information from stamps, photos, or colored diagrams
- Very large or very small input images may need threshold tuning for optimal results

---

## 📌 Roadmap / Ideas

- [ ] Multi-page PDF support (combine multiple scans into one document)
- [ ] Manual corner adjustment if auto-detection misses
- [ ] OCR integration to extract text from scans

---

## 📄 License

*Add a license (e.g. MIT) here — see [choosealicense.com](https://choosealicense.com/) if you're not sure which one to pick.*
