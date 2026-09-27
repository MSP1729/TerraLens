# Local 4-bit VQA server for SatQuery (RTX 3050 4GB)
import torch, gradio as gr
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

MODEL = 'AdaptLLM/remote-sensing-Qwen2.5-VL-3B-Instruct'
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                         bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL, quantization_config=bnb, device_map='cuda:0')
# small image size keeps memory under 4GB
processor = AutoProcessor.from_pretrained(MODEL, min_pixels=128*28*28, max_pixels=512*28*28)

def ask(image, question):
    if image is None or not question.strip():
        return 'Upload an image and ask a question.'
    messages = [{'role': 'user', 'content': [{'type': 'image', 'image': image.convert('RGB')},
                                             {'type': 'text', 'text': question}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    inputs = processor(text=[text], images=images, videos=videos, padding=True, return_tensors='pt').to('cuda:0')
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=256)
    torch.cuda.empty_cache()
    return processor.batch_decode([out[0][len(inputs.input_ids[0]):]], skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)[0]

gr.Interface(fn=ask, inputs=[gr.Image(type='pil', label='Satellite image'), gr.Textbox(label='Question')],
             outputs=gr.Textbox(label='Answer'), title='SatQuery VQA (local)').queue().launch(server_port=7870)