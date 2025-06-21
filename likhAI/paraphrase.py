from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt_tab')

tokenizer = AutoTokenizer.from_pretrained('tuner007/pegasus_paraphrase')

model = AutoModelForSeq2SeqLM.from_pretrained('tuner007/pegasus_paraphrase')

paraphraser = pipeline('text2text-generation', model = model,tokenizer = tokenizer, truncation = True)

def paraphrase_text(input_text):
    sentences = nltk.sent_tokenize(input_text)

    arr1 = []
    arr2 = []

    for i in range(len(sentences)):
        arr1.append(paraphraser(sentences[i]))

    for i in range(len(arr1)):
        arr2.append(arr1[i][0]['generated_text'])

    para = ''.join(arr2)

    return para