from transformers import MusicgenForConditionalGeneration, AutoProcessor
import scipy.io.wavfile as wavfile, numpy as np
from datetime import datetime
from translation_text import traslation_text

def generate_music(user_prompt, duration=10):
    prompt = traslation_text(user_prompt)
    
    MODEL_PATH = 'facebook/musicgen-small'
    TOKENS_COUNT=512
    SAMPLING_RATE = 32_000
    
    model = MusicgenForConditionalGeneration.from_pretrained(MODEL_PATH)
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    
    inputs = processor(text=[prompt], padding=True, return_tensors='pt')
    
    max_tokens = int(TOKENS_COUNT * (duration / 10))
    
    audio_values = model.generate(**inputs, max_new_tokens=max_tokens)
    
    audio_array = audio_values[0].cpu().numpy()
    
    if len(audio_array.shape) == 1: pass
    elif audio_array.shape[0] == 1: audio_array = audio_array[0]
    else: raise ValueError(f"Неподдержтиваемая форма аудио")
    
    audio_array = np.clip(audio_array, -1.0, 1.0)

    filename = f"music_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    wavfile.write(filename, SAMPLING_RATE, (audio_array * 32767).astype(np.int16))