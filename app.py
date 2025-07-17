from fastapi import FastAPI
from pydantic import BaseModel
import torch
import soundfile as sf
import numpy as np
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import uuid
import os
from utils import main

app = FastAPI()
device = "cuda" if torch.cuda.is_available() else "cpu"

model = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts").to(device)
prompt_tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
desc_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)

class TTSRequest(BaseModel):
    text: str
    description: str = "A female speaker delivers expressive speech in high quality."

def split_text(text):
    sentences = re.split(r'(?<=[।.!?])\s+', text)
    chunks, current = [], ""
    for sent in sentences:
        test = current + " " + sent
        if len(prompt_tokenizer.encode(test)) < max_tokens:
            current = test
        else:
            chunks.append(current.strip())
            current = sent
    if current:
        chunks.append(current.strip())
    return chunks

@app.post("/tts")
async def generate_audio(data: TTSRequest):
max_tokens = 300  # Safe token limit
    text = data.text.strip()
    chunks = split_text(text)
    desc_input = desc_tokenizer(data.description, return_tensors="pt").to(device)

    audios = []
    for chunk in chunks:
        prompt_input = prompt_tokenizer(chunk, return_tensors="pt").to(device)

        output = model.generate(
            input_ids=desc_input.input_ids,
            attention_mask=desc_input.attention_mask,
            prompt_input_ids=prompt_input.input_ids,
            prompt_attention_mask=prompt_input.attention_mask
        )
        audio = output.cpu().numpy().squeeze()
        audios.append(audio)

    full_audio = np.concatenate(audios)

    filename = f"{uuid.uuid4().hex}.wav"
    sf.write(filename, full_audio, model.config.sampling_rate)

    try:
        read_url = await main(filename)
    finally:
        if os.path.exists(filename):
            os.remove(filename)

    return {
        "message": "TTS audio generated and uploaded successfully.",
        "url": read_url
    }
