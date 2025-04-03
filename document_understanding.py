import os
import cv2
import ast
import json
import pytesseract
import pandas as pd
import numpy as np
import torch
from PIL import Image
import argparse

# Set up MPS fallback for Apple Silicon
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Define the device: use MPS if available, otherwise CPU.
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# Hugging Face Transformers
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification, TrainingArguments, Trainer

# Detectron2 imports
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

###############################################
# Detectron2 Layout Detection Functions
###############################################

def setup_detectron2():
    """Set up Detectron2 with a pre-trained PubLayNet model for layout detection."""
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("Misc/publaynet/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("Misc/publaynet/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # adjust threshold as needed
    predictor = DefaultPredictor(cfg)
    return predictor

def detect_regions(image, predictor):
    """Detect layout regions in the image using Detectron2."""
    outputs = predictor(image)
    instances = outputs["instances"].to("cpu")
    boxes = instances.pred_boxes if instances.has("pred_boxes") else None
    if boxes is not None:
        detected_boxes = boxes.tensor.numpy().tolist()
    else:
        detected_boxes = []
    return detected_boxes

###############################################
# OCR Functions using Tesseract
###############################################

def ocr_regions(image, boxes):
    """
    For each detected region, crop the region and run Tesseract OCR.
    Returns a list of dictionaries: { "box": [x1, y1, x2, y2], "text": ... }
    """
    ocr_results = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cropped_region = image[y1:y2, x1:x2]
        # Optional: preprocess cropped_region (e.g., thresholding) for better OCR accuracy
        text = pytesseract.image_to_string(cropped_region)
        ocr_results.append({"box": [x1, y1, x2, y2], "text": text})
    return ocr_results

###############################################
# LayoutLMv3 Inference Function
###############################################

def run_layoutlmv3_inference(pil_image, words, normalized_boxes):
    """
    Runs LayoutLMv3 on the provided image, words, and bounding boxes.
    """
    # Disable built-in OCR because we're providing our own tokens/boxes.
    processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base")
    model.to(device)
    
    encoding = processor(pil_image, words, boxes=normalized_boxes, return_tensors="pt")
    # Move all encoding tensors to the device.
    encoding = {k: v.to(device) for k, v in encoding.items()}
    outputs = model(**encoding)
    return outputs

###############################################
# Dataset Class for CSV-based Data
###############################################

class DocumentDatasetCSV(torch.utils.data.Dataset):
    """
    Custom Dataset that merges annot.csv and img.csv.
    annot.csv must have: id (annotation id), image_id, bbox, utf8_string, ...
    img.csv must have: id (image id), width, height, set, file_name.
    
    This class renames:
      - annot.csv's "image_id" to "img_id"
      - img.csv's "id" to "img_id"
    and then merges on "img_id". It also assigns a dummy label (0) for each token.
    """
    def __init__(self, annot_csv, img_csv, root_dir="."):
        # Read CSV files
        df_annot = pd.read_csv(annot_csv)
        df_img = pd.read_csv(img_csv)
        
        # Rename columns so they don't conflict:
        df_annot = df_annot.rename(columns={"image_id": "img_id"})
        df_img = df_img.rename(columns={"id": "img_id"})
        
        # Merge on "img_id"
        self.df_merged = df_annot.merge(df_img, on="img_id", how="left")
        
        # Group by the image id
        self.grouped = self.df_merged.groupby("img_id")
        self.img_ids = list(self.grouped.groups.keys())
        self.root_dir = root_dir

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        group = self.grouped.get_group(img_id)
        file_name = group["file_name"].unique()[0]
        image_path = os.path.join(self.root_dir, file_name)
        image = Image.open(image_path).convert("RGB")
        
        words = []
        boxes = []
        labels = []  # Dummy labels
        for _, row in group.iterrows():
            bbox_str = row["bbox"]  # Expected format: "[x, y, w, h]"
            bbox_list = ast.literal_eval(bbox_str)
            x, y, w, h = bbox_list
            x1, y1 = x, y
            x2, y2 = x + w, y + h
            # Ensure text is a string
            words.append(str(row["utf8_string"]))
            boxes.append([x1, y1, x2, y2])
            labels.append(0)
        
        sample = {"img_id": img_id, "image": image, "words": words, "boxes": boxes, "labels": labels}
        return sample

###############################################
# Data Collator for LayoutLMv3 Training
###############################################

def data_collator(samples, processor):
    images = [s["image"] for s in samples]
    words = [s["words"] for s in samples]
    boxes = [s["boxes"] for s in samples]
    labels = [s["labels"] for s in samples]  # Dummy labels
    
    # Normalize boxes for each image (scale boxes to [0, 1000])
    norm_boxes = []
    for s in samples:
        w, h = s["image"].size
        curr_boxes = []
        for box in s["boxes"]:
            x1, y1, x2, y2 = box
            curr_boxes.append([
                int(1000 * x1 / w),
                int(1000 * y1 / h),
                int(1000 * x2 / w),
                int(1000 * y2 / h)
            ])
        norm_boxes.append(curr_boxes)
    
    encoding = processor(
        images,
        words,
        boxes=norm_boxes,
        word_labels=labels,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    return encoding

###############################################
# Training Function for LayoutLMv3
###############################################

def train_model(args):
    # Disable built-in OCR
    processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base")
    model.to(device)
    
    dataset = DocumentDatasetCSV(annot_csv=args.annot_file, img_csv=args.img_file, root_dir=args.root_dir)
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_steps=50,
        remove_unused_columns=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=lambda samples: data_collator(samples, processor),
    )
    
    trainer.train()
    trainer.save_model(args.output_dir)
    print("Training completed. Model saved to", args.output_dir)

###############################################
# Inference Pipeline (OCR + LayoutLMv3 Fusion)
###############################################

def inference_pipeline(image_path):
    image_cv = cv2.imread(image_path)
    if image_cv is None:
        print("Error: Unable to load image", image_path)
        return
    
    predictor = setup_detectron2()
    detected_boxes = detect_regions(image_cv, predictor)
    print("Detected Regions:", detected_boxes)
    
    ocr_results = ocr_regions(image_cv, detected_boxes)
    for res in ocr_results:
        print("OCR result:", res)
    
    words = []
    boxes = []
    for res in ocr_results:
        tokens = res["text"].split()  # Basic token split; adjust as needed
        for token in tokens:
            words.append(token)
            boxes.append(res["box"])
    
    height, width, _ = image_cv.shape
    norm_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box
        norm_boxes.append([
            int(1000 * x1 / width),
            int(1000 * y1 / height),
            int(1000 * x2 / width),
            int(1000 * y2 / height)
        ])
    
    pil_image = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
    processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
    outputs = run_layoutlmv3_inference(pil_image, words, norm_boxes)
    print("LayoutLMv3 Outputs:", outputs)

###############################################
# Main Function and Argument Parsing
###############################################

def main():
    parser = argparse.ArgumentParser(description="Document Understanding Pipeline")
    parser.add_argument("--mode", type=str, choices=["train", "infer"], required=True,
                        help="Choose 'train' to fine-tune LayoutLMv3 or 'infer' to run OCR+Vision Transformer fusion on a sample image.")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save trained model (for training mode)")
    
    # Training arguments
    parser.add_argument("--annot_file", type=str, default="annot.csv", help="Path to annotation CSV")
    parser.add_argument("--img_file", type=str, default="img.csv", help="Path to image metadata CSV")
    parser.add_argument("--root_dir", type=str, default="train_val_images/train_images", help="Root directory for image files")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    
    # Inference arguments
    parser.add_argument("--image_path", type=str, default="sample_document.jpg", help="Path to a sample image for inference")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train_model(args)
    elif args.mode == "infer":
        inference_pipeline(args.image_path)
    else:
        print("Invalid mode selected.")

if __name__ == "__main__":
    main()
