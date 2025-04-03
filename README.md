# 🧠 Document Understanding with LayoutLMv3 + Detectron2 + Tesseract OCR

This project is a complete end-to-end **Document Understanding** pipeline that combines **layout detection**, **OCR**, and **transformer-based document intelligence**. It leverages:

- **Detectron2** for layout (region) detection
- **Tesseract OCR** for extracting text from detected regions
- **LayoutLMv3** for document understanding using both visual and textual features

---

## 🔍 Project Overview

The goal is to extract structured and semantically rich information from unstructured documents like scanned forms, invoices, and reports.

The pipeline performs:

1. **Region detection** using a fine-tuned Mask R-CNN (PubLayNet) via Detectron2
2. **OCR extraction** on detected regions using Tesseract
3. **Token classification** using [LayoutLMv3](https://huggingface.co/microsoft/layoutlmv3-base) to understand relationships between layout and text

---

## 📁 Dataset

We use the **TextOCR** dataset available on Kaggle:

📦 [TextOCR: Text Extraction from Images](https://www.kaggle.com/datasets/robikscube/textocr-text-extraction-from-images-dataset/data)

> **Note:** Due to size, the dataset files are excluded from this repo. Please download them separately and place them in the appropriate directories.

---

## 🤖 Trained Model

The fine-tuned LayoutLMv3 model is uploaded on Hugging Face:

🧠 [varad-suryavanshi12/OCR-model](https://huggingface.co/varad-suryavanshi12/OCR-model)

---

## 🛠️ Installation

Install all dependencies using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

### ⚠️ Detectron2 Installation

Detectron2 requires OS-specific installation. Please follow official instructions:

🔗 [Detectron2 Install Guide](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md)

---

## 🏋️‍♂️ Training the Model

```bash
python document_understanding.py \
  --mode train \
  --annot_file annot.csv \
  --img_file img.csv \
  --root_dir train_val_images/train_images \
  --output_dir ./output \
  --batch_size 2 \
  --epochs 3
```

You can adjust `batch_size`, `epochs`, and paths as needed.

---

## 🔎 Running Inference

Use the full pipeline (Detectron2 + OCR + LayoutLMv3) on a document image:

```bash
python document_understanding.py \
  --mode infer \
  --image_path path/to/sample_document.jpg
```

The output includes:
- Detected layout regions
- OCR results per region
- Token-level predictions from LayoutLMv3

---

## 📂 Project Structure

```
document_understanding_ai/
├── document_understanding.py   # Main pipeline script
├── requirements.txt            # Dependencies
├── annot.csv / img.csv         # Dataset metadata
├── train_val_images/           # (excluded) Image folder from Kaggle
├── output/                     # Model checkpoints (ignored in Git)
└── README.md                   # You are here
```

---

## 🧹 .gitignore

Dataset and large files are excluded from Git. See the included `.gitignore` file for details.

---

## 🙌 Acknowledgements

- [Detectron2](https://github.com/facebookresearch/detectron2)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [LayoutLMv3](https://huggingface.co/microsoft/layoutlmv3-base)
- [TextOCR Dataset](https://www.kaggle.com/datasets/robikscube/textocr-text-extraction-from-images-dataset/data)

---

## 📬 Contact

Feel free to connect or reach out for collaboration!

**Author**: Varad Suryavanshi  
🔗 [Hugging Face Profile](https://huggingface.co/varad-suryavanshi12)
