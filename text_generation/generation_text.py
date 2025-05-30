import torch, re, language_tool_python
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2LMHeadModel
from detoxify import Detoxify
from replaced_words import anachronisms, toxic_words


MODEL_PATH = "ptvnck/ww2_finetuning_gpt2"

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
model.eval()

tools = language_tool_python.LanguageTool('ru-RU')
toxic = Detoxify('original')


def preprocessing_output(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    paragraphs = [' '.join(sentences[i:i+3]) for i in range(0, len(sentences), 3)]
    text = '\n\n'.join(paragraphs)
    
    mistakes = tools.check(text)
    text = language_tool_python.utils.correct(text, mistakes)
    
    for word, synonym in anachronisms.items():
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        text = pattern.sub(synonym, text)
    
    def change_toxic_words(match):
        word = match.group(0).lower()
        return toxic_words.get(word, word)
    
    toxic_pattern = re.compile(r'\b(' + '|'.join(map(re.escape, toxic_words.keys())) + r')\b', flags=re.IGNORECASE)
    text = toxic_pattern.sub(change_toxic_words, text)
    
    return text

def generate_story(user_prompt):
    full_prompt = (
        "Ты — писатель, задача которого — создать короткий художественный рассказ, строго основанный на содержании и атмосфере предоставленного отрывка из военного дневника. "
        "Твоя цель — не написать общую историю о войне, а глубоко раскрыть именно те события, чувства и образы, которые присутствуют в данном конкретном отрывке. "
        "Не добавляй персонажей, имена (вроде Жукова, Карпова и т.д.), локации или события, которых нет в тексте дневника. Не пиши о героизме или наградах, если этого нет в отрывке.\n\n"
        "ОТРЫВОК ДНЕВНИКА:\n"
        "-------------------------------------\n"
        f"{user_prompt}\n"
        "-------------------------------------\n\n"
        "ЗАДАНИЕ:\n"
        "1. Внимательно прочти отрывок выше.\n"
        "2. Напиши рассказ от третьего лица, который передает исключительно переживания автора дневника: его жизнь на 'дрожащей земле' под 'воющим небом', постоянное напряжение слуха, ожидание звуков сирен, бомб или самолетов, страх ('в меня или не в меня?'), мимолетное облегчение после отбоя и сложное отношение к дому как к защите и одновременно угрозе ('дома душили своих хозяев').\n"
        "3. Твой рассказ должен быть сфокусирован ТОЛЬКО на этих элементах. Используй образы и язык, близкие к оригинальному отрывку.\n"
        "4. Передай атмосферу войны, холод, страх, голос земли, как это чувствуется в дневнике.\n"
        "5. Не используй современные слова. Пиши эмоционально, живо, образно.\n\n"
        "РАССКАЗ:\n"
    )
    
    input_ids = tokenizer.encode(full_prompt, return_tensors="pt")
    if input_ids.shape[1] > 2048:
        input_ids = input_ids[:, -2048:]
    
    with torch.no_grad():
        output_sequences = model.generate(
            input_ids,
            max_new_tokens=400,
            temperature=0.4,
            top_k=40,
            top_p=0.90,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
        )
    
    prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    full_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    story = full_text[len(prompt_text):].strip()

    final_text = preprocessing_output(story)
    toxicity_results = toxic.predict(final_text)

    return final_text, toxicity_results

