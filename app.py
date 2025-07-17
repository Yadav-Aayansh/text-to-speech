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

executor = ThreadPoolExecutor(max_workers=12)  # A100 can handle this easily

class TTSRequest(BaseModel):
    text: str
    description: str = "A female speaker delivers expressive speech in high quality."

def improved_split_text(text, max_tokens=250):
    """
    Enhanced text splitting with proper Hindi tokenization
    """
    # Better Hindi sentence splitting pattern
    sentence_pattern = r'(?<=[।.!?।])\s+'
    sentences = re.split(sentence_pattern, text.strip())
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # Use actual tokenizer for accurate token counting
        test_chunk = current_chunk + " " + sentence if current_chunk else sentence
        token_count = len(prompt_tokenizer.encode(test_chunk))
        
        if token_count <= max_tokens:
            current_chunk = test_chunk
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # Handle oversized sentences
            if len(prompt_tokenizer.encode(sentence)) > max_tokens:
                words = sentence.split()
                temp_chunk = ""
                for word in words:
                    test_word_chunk = temp_chunk + " " + word if temp_chunk else word
                    if len(prompt_tokenizer.encode(test_word_chunk)) <= max_tokens:
                        temp_chunk = test_word_chunk
                    else:
                        if temp_chunk:
                            chunks.append(temp_chunk.strip())
                        temp_chunk = word
                if temp_chunk:
                    current_chunk = temp_chunk
            else:
                current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return [chunk for chunk in chunks if chunk.strip()]


def improved_tts_chunk(chunk, description, temperature=0.7, min_length=10, max_length=1000):
    """
    Enhanced TTS generation with proper parameters and error handling
    """
    if not chunk.strip():
        return np.array([])
    
    try:
        with torch.inference_mode():  # Better performance than torch.no_grad
            prompt_input = prompt_tokenizer(chunk, return_tensors="pt").to(device)
            desc_input = desc_tokenizer(description, return_tensors="pt").to(device)
            
            output = model.generate(
                input_ids=desc_input.input_ids,
                attention_mask=desc_input.attention_mask,
                prompt_input_ids=prompt_input.input_ids,
                prompt_attention_mask=prompt_input.attention_mask,
                temperature=temperature,
                do_sample=True,
                min_length=min_length,
                max_length=max_length,
                pad_token_id=desc_tokenizer.eos_token_id,
                eos_token_id=desc_tokenizer.eos_token_id,
                repetition_penalty=1.1,
                length_penalty=1.0
            )
            
            audio = output.cpu().numpy().squeeze()
            
            # Handle empty audio generation
            if audio.size == 0:
                silence_duration = 0.1
                sample_rate = model.config.sampling_rate
                silence_samples = int(silence_duration * sample_rate)
                audio = np.zeros(silence_samples, dtype=np.float32)
            
            return audio
            
    except Exception as e:
        print(f"Error generating audio for chunk: {e}")
        # Return silence on error
        silence_duration = 0.1
        sample_rate = model.config.sampling_rate
        silence_samples = int(silence_duration * sample_rate)
        return np.zeros(silence_samples, dtype=np.float32)

def add_crossfade(audio1, audio2, crossfade_duration=0.05):
    """
    Add smooth crossfade between audio segments to prevent clicks
    """
    if len(audio1) == 0:
        return audio2
    if len(audio2) == 0:
        return audio1
    
    sample_rate = model.config.sampling_rate
    crossfade_samples = int(crossfade_duration * sample_rate)
    
    # Skip crossfade if audio is too short
    if len(audio1) < crossfade_samples or len(audio2) < crossfade_samples:
        return np.concatenate([audio1, audio2])
    
    # Extract fade regions
    fade_out_region = audio1[-crossfade_samples:]
    fade_in_region = audio2[:crossfade_samples]
    
    # Create fade curves
    fade_out_curve = np.linspace(1.0, 0.0, crossfade_samples)
    fade_in_curve = np.linspace(0.0, 1.0, crossfade_samples)
    
    # Apply crossfade
    faded_out = fade_out_region * fade_out_curve
    faded_in = fade_in_region * fade_in_curve
    crossfaded = faded_out + faded_in
    
    # Combine audio
    before_crossfade = audio1[:-crossfade_samples]
    after_crossfade = audio2[crossfade_samples:]
    
    return np.concatenate([before_crossfade, crossfaded, after_crossfade])

def add_silence_padding(audio, padding_duration=0.1):
    """
    Add silence padding to prevent abrupt starts/ends
    """
    if len(audio) == 0:
        return audio
    
    sample_rate = model.config.sampling_rate
    padding_samples = int(padding_duration * sample_rate)
    silence = np.zeros(padding_samples, dtype=audio.dtype)
    
    return np.concatenate([silence, audio, silence])



@app.post("/tts")
async def improved_generate_audio(data: TTSRequest):
    text = data.text.strip()
    
    if not text:
        return {"error": "Empty text provided"}
    
    # Improved text splitting
    chunks = improved_split_text(text, max_tokens=250)
    
    if not chunks:
        return {"error": "No valid text chunks found"}
    
    print(f"Processing {len(chunks)} chunks")
    
    # Generate audio for each chunk
    audio_segments = []
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}: {chunk[:50]}...")
        
        try:
            audio = await asyncio.to_thread(
                improved_tts_chunk, 
                chunk, 
                data.description,
                temperature=0.7,
                min_length=10,
                max_length=1000
            )
            
            if len(audio) > 0:
                padded_audio = add_silence_padding(audio, padding_duration=0.05)
                audio_segments.append(padded_audio)
                print(f"Generated audio for chunk {i+1}: {len(audio)} samples")
            else:
                print(f"Warning: No audio generated for chunk {i+1}")
                
        except Exception as e:
            print(f"Error processing chunk {i+1}: {e}")
            continue
    
    if not audio_segments:
        return {"error": "No audio segments generated"}
    
    # Concatenate with crossfading
    print("Concatenating audio segments with crossfading...")
    full_audio = audio_segments[0]
    
    for i in range(1, len(audio_segments)):
        full_audio = add_crossfade(full_audio, audio_segments[i], crossfade_duration=0.03)
    
    # Add final padding and normalize
    full_audio = add_silence_padding(full_audio, padding_duration=0.1)
    
    # Normalize audio to prevent clipping
    if len(full_audio) > 0:
        max_val = np.max(np.abs(full_audio))
        if max_val > 0:
            full_audio = full_audio / max_val * 0.95
    
    print(f"Final audio length: {len(full_audio)} samples ({len(full_audio) / model.config.sampling_rate:.2f} seconds)")
    
    # Save the audio
    filename = f"{uuid.uuid4().hex}.wav"
    sf.write(filename, full_audio, model.config.sampling_rate)
    
    try:
        read_url = await main(filename)
        return {
            "message": "TTS audio generated and uploaded successfully.",
            "url": read_url,
            "chunks_processed": len(chunks),
            "audio_duration": len(full_audio) / model.config.sampling_rate
        }
    finally:
        if os.path.exists(filename):
            os.remove(filename)
