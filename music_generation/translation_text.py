from transformers import MarianModel, MarianTokenizer

def traslation_text(text):
    MODEL_PATH = 'Helsinki-NLP/opus-mt-ru-en'
    
    tokenizer = MarianTokenizer.from_pretrained(MODEL_PATH)
    model = MarianModel.from_pretrained(MODEL_PATH)
    
    inputs = tokenizer(text, return_tensors='pt', padding=True)
    translation = model.generate(**inputs, max_length=512)
    
    text = model.batch_decode(translation, skip_special_tokens=True)[0]
    
    return text