from fastapi import FastAPI
from pydantic import BaseModel
import torch
import soundfile as sf
import numpy as np
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import uuid
import os

app = FastAPI()
device = "cuda" if torch.cuda.is_available() else "cpu"

model = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts").to(device)
prompt_tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
desc_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)

class TTSRequest(BaseModel):
    text: str
    description: str = "A female speaker delivers expressive speech in high quality."

@app.post("/tts")
def generate_audio(data: TTSRequest):
    prompt_input = prompt_tokenizer(data.text, return_tensors="pt").to(device)
    desc_input = desc_tokenizer(data.description, return_tensors="pt").to(device)

    output = model.generate(
        input_ids=desc_input.input_ids,
        attention_mask=desc_input.attention_mask,
        prompt_input_ids=prompt_input.input_ids,
        prompt_attention_mask=prompt_input.attention_mask
    )
    audio = output.cpu().numpy().squeeze()

    filename = f"{uuid.uuid4().hex}.wav"
    sf.write(filename, audio, model.config.sampling_rate)

    return {"message": "Audio generated", "file": filename}
