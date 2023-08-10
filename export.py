from transformers import CLIPModel, CLIPProcessor
import torch
from openvino.runtime import serialize
from openvino.tools import mo
import numpy as np
from PIL import Image
from pathlib import Path
import requests

query = "Who developed the Theory of General Relativity?"
image_url = "https://www.storypick.com/wp-content/uploads/2016/01/AE-2.jpg"
image = Image.open(requests.get(image_url, stream=True).raw)
im_tensor = np.array(image)

model_checkpoint = "openai/clip-vit-base-patch16"

model = CLIPModel.from_pretrained(model_checkpoint).eval()
processor = CLIPProcessor.from_pretrained(model_checkpoint)

model_name = model_checkpoint.split('/')[-1]

onnx_model_path = Path("onnx") / f"{model_name}.onnx"
onnx_model_path.parent.mkdir(exist_ok=True)

inputs = processor(text=[query], images=[im_tensor], return_tensors="pt")

torch.onnx.export(
    model,  # model is being run
    dict(inputs),
    onnx_model_path,  # where to save the model
    opset_version=14,  # the ONNX version to export the model to
    input_names=["input_ids", "pixel_values",
                 "attention_mask"],  # the model's input names
    output_names=[
        "logits_per_image", "logits_per_text", "text_embeds", "image_embeds"
    ],  # the model's output names
    dynamic_axes={  # variable length axes
        "input_ids": {
            0: "batch",
            1: "sequence"
        },
        "pixel_values": {
            0: "batch",
            1: "num_channels",
            2: "height",
            3: "width"
        },
        "attention_mask": {
            0: "batch",
            1: "sequence"
        },
        "logits_per_image": {
            0: "batch"
        },
        "logits_per_text": {
            0: "batch"
        },
        "text_embeds": {
            0: "batch"
        },
        "image_embeds": {
            0: "batch"
        }
    })

text_ov_model = mo.convert_model(
    onnx_model_path,
    compress_to_fp16=True,
    input="input_ids,attention_mask",
    output="text_embeds",
)

# get image size after preprocessing from the processor
crops_info = processor.image_processor.crop_size.values() if hasattr(
    processor,
    "image_processor") else processor.feature_extractor.crop_size.values()
processed_image_height_width = ",".join(map(str, crops_info))
image_ov_model = mo.convert_model(
    onnx_model_path,
    compress_to_fp16=True,
    input="pixel_values",
    input_shape=f"[1,3,{processed_image_height_width}]",
    output="image_embeds",
)

ov_dir = Path("ir")
ov_dir.mkdir(exist_ok=True)
text_model_path = ov_dir / f"{model_name}_text.xml"
image_model_path = ov_dir / f"{model_name}_image.xml"

# write resulting models on disk
serialize(text_ov_model, str(text_model_path))
serialize(image_ov_model, str(image_model_path))