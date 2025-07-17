from fastapi import FastAPI
from pydantic import BaseModel
import torch
import soundfile as sf
import numpy as np
import re
import uuid
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
from utils import main

app = FastAPI()
device = "cuda" if torch.cuda.is_available() else "cpu"

model = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts").to(device)
prompt_tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
desc_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)

class TTSRequest(BaseModel):
    text: str
    description: str = "A female speaker delivers expressive speech in high quality."

import re
import unicodedata

def split_sentences(text):
    text = unicodedata.normalize("NFC", text)
    return re.split(r'(?<=[।.!?॥])\s+', text.strip())


def generate_by_sentence(description, full_prompt):
    desc_ids = desc_tokenizer(description, return_tensors="pt").to(device)
    sentences = split_sentences(full_prompt)

    final_audio = []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        prompt_ids = prompt_tokenizer(sentence, return_tensors="pt").to(device)

        out = model.generate(
            input_ids=desc_ids.input_ids,
            attention_mask=desc_ids.attention_mask,
            prompt_input_ids=prompt_ids.input_ids,
            prompt_attention_mask=prompt_ids.attention_mask,
            max_new_tokens=1000
        )
        audio = out.cpu().numpy().squeeze()
        final_audio.append(audio)

        silence = np.zeros(int(0.05 * model.config.sampling_rate))
        final_audio.append(silence)

    return np.concatenate(final_audio)


