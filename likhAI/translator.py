from transformers import MarianMTModel, MarianTokenizer

language_map = {
    'hindi': 'en-hi',
    'french': 'en-fr',
    'german': 'en-de',
    'spanish': 'en-es',
    'italian': 'en-it'
}

def load_model(language):
    # Find the language pair in the map
    language_pair = language_map.get(language.lower())
    if not language_pair:
        raise ValueError(f"Translation model for language '{language}' not found.")

    model_name = f'Helsinki-NLP/opus-mt-{language_pair}'
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)

    return model, tokenizer

def translate_text(text, language):
    model, tokenizer = load_model(language)

    # Tokenize the input text
    tokens = tokenizer.encode(text, return_tensors="pt")

    # Generate the translated text
    translated_tokens = model.generate(
        tokens,
        num_beams=4,
        max_length=min(len(tokens[0]) * 2, 512),  # Dynamically cap max_length for shorter inputs
        early_stopping=True
    )

    # Decode the generated tokens to text
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

    return translated_text

# input_text = '''I am good and happy'''

# tt1 = translate_text(input_text, 'hindi')
# tt2 = translate_text(input_text, 'german')
# tt3 = translate_text(input_text, 'spanish')
# tt4 = translate_text(input_text, 'italian')
# tt5 = translate_text(input_text, 'french')

# print(tt1)
# print(tt2)
# print(tt3)
# print(tt4)
# print(tt5)