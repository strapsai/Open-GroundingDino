import argparse
import os
import json
import numpy as np
import torch
from PIL import Image
from inference_on_a_image import load_image, load_model, get_grounding_output, plot_boxes_to_image

def run_inference_on_directory(image_dir, config_file, checkpoint_path, text_prompt, output_dir, box_threshold, text_threshold, token_spans, cpu_only):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model = load_model(config_file, checkpoint_path, cpu_only=cpu_only)
    
    # JSONL output file
    jsonl_path = os.path.join(output_dir, "predictions.jsonl")
    
    with open(jsonl_path, "w") as jsonl_file:
        # Write prompt at the top level
        jsonl_file.write(json.dumps({"prompt": text_prompt}) + "\n")
        
        for image_name in os.listdir(image_dir):
            image_path = os.path.join(image_dir, image_name)
            
            # Check if it's an image file
            if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            
            # Load image
            image_pil, image = load_image(image_path)
            
            # Run inference
            boxes_filt, pred_phrases = get_grounding_output(
                model, image, text_prompt, box_threshold, text_threshold, cpu_only=cpu_only, token_spans=token_spans
            )
            
            # Save prediction image
            pred_dict = {
                "boxes": boxes_filt,
                "size": [image_pil.size[1], image_pil.size[0]],  # H, W
                "labels": pred_phrases,
            }
            
            image_with_box = plot_boxes_to_image(image_pil, pred_dict)[0]
            save_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_pred.jpg")
            image_with_box.save(save_path)
            
            # Prepare JSONL entry in ODVG format
            jsonl_entry = {
                "image": image_name,
                "boxes": boxes_filt.tolist(),
                "labels": pred_phrases,
            }
            
            jsonl_file.write(json.dumps(jsonl_entry) + "\n")
            
            print(f"Processed: {image_name} -> {save_path}")
    
    print(f"All predictions saved in {jsonl_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grounding DINO - Inference on a Directory", add_help=True)
    parser.add_argument("--config_file", "-c", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint_path", "-p", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--image_dir", "-d", type=str, required=True, help="Directory containing images")
    parser.add_argument("--text_prompt", "-t", type=str, required=True, help="Text prompt")
    parser.add_argument("--output_dir", "-o", type=str, required=True, help="Output directory for predictions")
    parser.add_argument("--box_threshold", type=float, default=0.3, help="Box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="Text threshold")
    parser.add_argument("--token_spans", type=str, default=None, help="Token spans for phrase detection")
    parser.add_argument("--cpu-only", action="store_true", help="Run on CPU only (default: False)")
    
    args = parser.parse_args()
    
    run_inference_on_directory(
        args.image_dir, args.config_file, args.checkpoint_path, args.text_prompt,
        args.output_dir, args.box_threshold, args.text_threshold, args.token_spans, args.cpu_only
    )
