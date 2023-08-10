from openvino.runtime import Core
from transformers import CLIPProcessor
import os
import gradio as gr
import numpy as np
import faiss
from PIL import Image
from zipfile import ZipFile

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

index = faiss.IndexFlatL2(512)
image_list = []

core = Core()

text_model = core.read_model('./ir/clip-vit-base-patch16_text.xml')
image_model = core.read_model('./ir/clip-vit-base-patch16_image.xml')

text_model = core.compile_model(model=text_model, device_name='CPU')
image_model = core.compile_model(model=image_model, device_name='CPU')


def search(text, num_output):
    text_inputs = dict(processor(text=[text], return_tensors="np"))
    text_embed_query = text_model(text_inputs)[text_model.output()]
    D, I = index.search(text_embed_query, num_output)
    results = [image_list[i] for i in list(I[0])]
    return results


def zip_to_index(file_obj):
    global index
    global image_list
    root = os.path.basename(file_obj.name).split('/')[-1].split('.')[0]
    zf = ZipFile(file_obj.name)
    zf.extractall('./')
    for item in os.listdir(root):
        image_file = os.path.join(root, item)
        image = np.array(Image.open(image_file))
        image_embed = dict(processor(images=[image], return_tensors="np"))
        image_embed_index = image_model(image_embed)[image_model.output()]
        index.add(image_embed_index)
        image_list.append(root + '/' + item)
    return 'Vector of embedded images is created'


embedding = gr.Interface(zip_to_index, "file", "text")
searching = gr.Interface(fn=search,
                         inputs=["text", gr.Slider(1, 10, step=1)],
                         outputs=gr.Gallery().style(columns=[4],
                                                    object_fit="contain",
                                                    height="auto"))
demo = gr.TabbedInterface([embedding, searching], [
    "Step 1: Upload the images fold in .zip format", "Step 2: Search with text"
], "Use text to search image")

if __name__ == "__main__":
    demo.launch()