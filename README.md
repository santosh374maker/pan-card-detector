<div align="center">

# 🪪 PAN Card Detector

### AI-powered Indian PAN card detection using YOLOv8

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-FF6B35?style=for-the-badge&logo=yolo&logoColor=white)](https://ultralytics.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

<br/>

> Upload any ID card photo — the model detects if it's an Indian PAN card and draws a bounding box around it. All other card types are rejected with a "Not a PAN Card" overlay.

<br/>

<!-- Replace the line below with your actual GIF once you have it -->
![Demo GIF](assets/demo.gif)

<br/>

</div>

---

## 📺 Demo Video

<!-- Replace the URL below with your actual YouTube video link -->
[![Watch Demo on YouTube](https://img.shields.io/badge/Watch%20Demo-YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://youtube.com/your-video-link)

---

## 📸 Screenshots

<!-- Replace with your actual screenshots -->
| Input | Output — PAN Card | Output — Not PAN Card |
|---|---|---|
| ![input](assets/screenshot_input.jpg) | ![pan](assets/screenshot_pan.jpg) | ![notpan](assets/screenshot_notpan.jpg) |

---

## ✨ Features

- **PAN card detection** — detects Indian PAN cards with 99.5% accuracy (mAP50)
- **Real-time rejection** — Aadhaar, driving licence, credit cards → "Not a PAN Card"
- **Single bounding box** — draws one tight yellow box around the detected card
- **Confidence score** — displays model certainty as a percentage
- **Download result** — save the annotated output image directly from the app
- **Clean dark UI** — built with Streamlit, professional dark theme

---

## 🧠 How It Works

```
Input Image
    │
    ▼
┌─────────────────────────┐
│   YOLOv8n Inference     │  ← Custom trained on PAN card dataset
│   conf threshold = 0.20 │
└────────────┬────────────┘
             │
     ┌───────┴────────┐
     ▼                ▼
PAN Card ✓        Not PAN Card ✗
Yellow bbox       Red overlay +
+ confidence %    "Not a PAN Card"
```

---

## 🗂️ Project Structure

```
project_yolo/
├── dataset/
│   ├── images/
│   │   ├── all/          ← all augmented images
│   │   ├── train/        ← training split (auto)
│   │   └── test/         ← test split (auto)
│   └── labels/
│       ├── all/          ← all YOLO label .txt files
│       ├── train/        ← train labels (auto)
│       └── test/         ← test labels (auto)
├── runs/
│   └── detect/
│       └── pancard_detector6/
│           └── weights/
│               └── best.pt   ← trained model weights
├── yolo_weights/
│   └── yolov8n.pt            ← pretrained base weights
├── raw_images/
│   ├── pan_cards/            ← original PAN card photos
│   ├── aadhaar/              ← Aadhaar card photos (negatives)
│   └── other_ids/            ← other ID photos (negatives)
├── output/                   ← detect.py saves results here
├── app.py                    ← Streamlit web app
├── augment.py                ← image augmentation script
├── setup_folders.py          ← folder setup + train/test split
├── fix_all.py                ← image corruption fix script
├── fix_labels.py             ← label class ID fix script
├── train.py                  ← YOLO training script
├── detect.py                 ← CLI inference script
├── data.yaml                 ← YOLO dataset config
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/pan-card-detector.git
cd pan-card-detector
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the web app
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

### 4. Or use the CLI
```bash
python detect.py \
  --weights "runs/detect/pancard_detector6/weights/best.pt" \
  --input "your_card_image.jpg" \
  --output "output" \
  --conf 0.20
```

---

## 🏋️ Train Your Own Model

### Step 1 — Collect images
```
raw_images/pan_cards/     ← PAN card photos  (min 10)
raw_images/aadhaar/       ← Aadhaar photos   (negatives)
raw_images/other_ids/     ← Other ID photos  (negatives)
```

### Step 2 — Augmentation
```bash
python augment.py --input raw_images/pan_cards/ --output dataset/images/all/ --count 80
```

### Step 3 — Label with LabelImg
```bash
pip install labelImg
labelImg
```
- Open Dir → `dataset/images/all/`
- Change Save Dir → `dataset/labels/all/`
- Format → **YOLO**
- Draw box on PAN cards → label as `pancard`
- Skip Aadhaar / other IDs (press `D`)

### Step 4 — Setup folders and split
```bash
python setup_folders.py
```

### Step 5 — Download base weights
```bash
curl -L https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt -o yolo_weights/yolov8n.pt
```

### Step 6 — Train
```bash
python train.py --epochs 100 --batch 8 --imgsz 640
```

### Step 7 — Test
```bash
python detect.py --weights "runs/detect/pancard_detector6/weights/best.pt" --input "test.jpg"
```

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| Model | YOLOv8n |
| mAP50 | **0.995** |
| mAP50-95 | 0.662 |
| Training epochs | 150 |
| Image size | 640×640 |
| Training dataset | 80 augmented PAN + 20 negatives |
| Inference speed | ~100ms on CPU |

---

## 🌐 Deployment

### Streamlit Cloud (Recommended — Free)
1. Push this repo to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect your GitHub repo
4. Set main file → `app.py`
5. Click **Deploy**

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| [YOLOv8](https://ultralytics.com) | Object detection model |
| [OpenCV](https://opencv.org) | Image processing + drawing |
| [Streamlit](https://streamlit.io) | Web application framework |
| [NumPy](https://numpy.org) | Array operations |
| [Pillow](https://pillow.readthedocs.io) | Image format handling |
| [LabelImg](https://github.com/HumanSignal/labelImg) | Manual image annotation |

---

## ⭐ Show Your Support

If this project helped you, please consider giving it a **star** on GitHub!

[![Star on GitHub](https://img.shields.io/github/stars/YOUR_USERNAME/pan-card-detector?style=social)](https://github.com/YOUR_USERNAME/pan-card-detector)

---

## 👨‍💻 Author

<div align="center">

**S Santosh Achary**

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/YOUR_USERNAME)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/YOUR_PROFILE)
[![YouTube](https://img.shields.io/badge/YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://youtube.com/YOUR_CHANNEL)

*Built with ❤️ using YOLOv8 + Streamlit*

</div>

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <sub>Made by <a href="https://github.com/YOUR_USERNAME">S Santosh Achary</a></sub>
</div>