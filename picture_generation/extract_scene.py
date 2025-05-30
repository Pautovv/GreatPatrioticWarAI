def extract_scene_from_text(text: str, top_k=4) -> list:
    text = text.lower()
    scene_keywords = {
    # Смерть и потери
    "смерть": "desolate field under dark clouds",
    "могила": "war graveyard with wooden crosses",
    "потеря": "abandoned helmet in muddy field",

    # Боевые действия
    "город": "ruined city street with rubble and smoke",
    "дом": "house with broken windows",
    "лес": "dark foggy forest with fallen trees",
    "река": "calm river under cloudy sky",
    "окоп": "muddy battlefield with trenches and barbed wire",
    "станция": "abandoned train station with broken signs",
    "взрыв": "explosion in a warzone, smoke and fire",
    "снаряд": "scattered shells on a muddy road",
    "битва": "chaotic battlefield, smoke and soldiers",
    "танк": "damaged tank on muddy field",
    "бомба": "air raid destruction site",
    "флаг": "tattered flag on battlefield",
    "солдат": "lonely soldier sitting on crate",
    "оружие": "discarded rifles lying in the mud",
    "пулемёт": "machine gun nest in snowy trench",
    "штурм": "soldiers charging through ruins under fire",

    # Зимняя тематика
    "зима": "snow-covered trench and barbed wire",
    "снег": "snow-covered ruins with smoke rising",
    "метель": "blizzard over deserted battlefield",
    "замёрзший": "frozen river with broken bridge",
    "мороз": "frosty trench with breath visible in the air",
    "ледяной": "icy road with abandoned military trucks",
    "шуба": "soldier in heavy coat walking through snow",
    "валики": "snow-covered sandbags in front line position",
    "зимний лес": "snowy forest with distant echoes of war",
    "снежинки": "snowflakes falling over quiet battlefield",
    "тёплая одежда": "stack of military winter uniforms near fire",

    # Быт и тыл
    "деревня": "lonely village road with empty houses",
    "кухня": "makeshift kitchen with old pots and wood stove",
    "письмо": "open letter on a soldier's desk",
    "бумага": "diary page with ink and candlelight",
    "поезд": "steam train leaving war-torn station",
    "ночь": "night sky with searchlights and distant gunfire",
    "дождь": "rainy street, puddles and gloom",
    "туман": "fog-covered battlefield with silhouettes",
    "окно": "window with broken glass and curtain blowing",
    "радио": "old radio on a wooden crate",
    "свеча": "single candle lighting a dark bunker",

    # Еда военного времени
    "еда": "simple meal on a tin plate",
    "хлеб": "loaf of bread beside army knife",
    "каша": "pot of barley porridge over small fire",
    "суп": "tin bowl with watery soup and spoon",
    "консервы": "open tin cans beside mess kit",
    "паёк": "military ration pack on wooden crate",
    "кухня полевая": "field kitchen with smoke and soldiers waiting",
    "чай": "cup of black tea steaming on cold morning",
    "варенье": "glass jar of jam next to old spoon",
    "картошка": "boiled potatoes in dented metal bowl",
    "сахар": "sugar cubes wrapped in paper on soldier's desk",

    # Эмоции и символика
    "грусть": "desolate landscape under gray sky",
    "страх": "empty village at twilight, eerie silence",
    "прощание": "two figures hugging at train station",
    "ожидание": "woman waiting at window with candle",
    "молитва": "small chapel with candlelight",
    "тоска": "dark empty room with open letter",
    "надежда": "sunlight breaking through clouds over ruins",

    # Природа и окружение
    "небо": "cloudy sky over vast field",
    "поле": "endless field with wild grass",
    "камни": "rocky hillside with sparse vegetation",
    "тропа": "narrow path through forest",
}


    found_scenes = []
    for keyword, scene in scene_keywords.items():
        if keyword in text and scene not in found_scenes:
            found_scenes.append(scene)
            if len(found_scenes) >= top_k:
                break

    return found_scenes if found_scenes else ["emotional wartime painting"]