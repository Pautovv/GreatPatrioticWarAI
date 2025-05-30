import torch, os
from music_generation.translation_text import traslation_text
from transformers import pipeline
from diffusers import StableDiffusionPipeline
from datetime import datetime
from extract_scene import extract_scene_from_text


def analyze_emotions(text: str):
    emotion_analyzer = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=3
    )
    result = emotion_analyzer(text)
    emotions = [e["label"].lower() for e in result[0] if e["score"] > 0.2]
    return emotions

def build_prompt(emotions, scenes):
    style = "1940s wartime painting, low saturation"
    emotion_part = f"evoking {', '.join(emotions)}" if emotions else "emotional tone"
    scene_part = ", ".join(scenes)
    return f"{scene_part}, {style}, {emotion_part}"

def generate_image(prompt: str, negative_prompt: str, save_path: str, pipe):
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=512,
        width=768,
        guidance_scale=7.5
    ).images[0]
    image.save(save_path)

def main():
    print("=== Генератор эмоциональных картин из военных дневников ===\n")

    text = input()
    text = traslation_text(text)

    emotions = analyze_emotions(text)

    scenes = extract_scene_from_text(text)

    final_prompt = build_prompt(emotions, scenes)

    negative_prompt = (
        "people, hands, faces, text, watermark, signature, bright colors, "
        "cartoon, modern buildings, vehicles, animals, low quality, "
        "oversaturated colors, low contrast, blurry details, extra limbs"
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    image_path = os.path.join(output_dir, f"image_{timestamp}.png")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )

    generate_image(final_prompt, negative_prompt, image_path, pipe)
