# camera-mask-python
# 🎭 Camera Mask Engine

An offline, lightweight Python application for real-time face tracking and privacy masking. Built with **OpenCV** and **MediaPipe**, it offers smooth expression mapping, and mode toggling.
---

## ✨ Key Features

- **Interactive Console Loader:** Clean startup screen featuring an animated green progress bar with real-time percentage indicators.
- **Dual Mask Modes:**
  - **Blur Mask (Default):** Applies a strong Gaussian face blur (frosted glass effect) with dynamic eye and mouth animations overlay.
  - **Bodycam Pixelation:** Toggleable pixelation effect for tactical/bodycam privacy.
- **OBS Studio Friendly:** Dedicated window visibility toggle to prevent stream capture glitches.
---

## ⌨️ Controls

| **`B`** | Toggle between **Blur Mask** and **Bodycam Pixelation** |
| **`H`** | Hide/Show the camera window (useful for background OBS capture) |
| **`ESC`** | Exit the application |

---

## 🚀 Quick Start

### 1. Prerequisites
Make sure you have Python 3.9+ installed on your system.

### 2. Installation
Clone the repository and install the dependencies:

```bash
git clone [https://github.com/YOUR_USERNAME/camera-mask-python.git](https://github.com/YOUR_USERNAME/camera-mask-python.git)
cd camera-mask-python
pip install -r requirements.txt

3. Model Asset

The face_landmarker.task file is included in the repository. If missing, the app will automatically attempt to download it on the first launch.

4. Running the App
Bash
python main.py

 Packaging to .exe

To compile the project into a standalone Windows executable:

Activate your virtual environment (if using one).

Run PyInstaller:

Bash
pyinstaller --onedir --noconfirm --collect-all mediapipe --collect-all cv2 --collect-all opencv-python main.py
Important: Copy face_landmarker.task into dist/main/ right alongside main.exe.
