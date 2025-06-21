from transformers import PegasusForConditionalGeneration
from transformers import PegasusTokenizer
from transformers import pipeline
# Pick model
model_name = "google/pegasus-xsum"

# Load pretrained tokenizer
pegasus_tokenizer = PegasusTokenizer.from_pretrained(model_name)

summarizer = pipeline(
    "summarization", 
    model=model_name, 
    tokenizer=pegasus_tokenizer, 
    framework="pt"
)

def summarize_text(input_text):
    
    input_length = len(pegasus_tokenizer.tokenize(input_text))
    max_length = max(10, input_length // 2)  # Ensure max_length is at least 10
    min_length = max(5, input_length // 4)   # Ensure min_length is at least 5

    # Generate summary
    summary = summarizer(
        input_text,
        max_length = max_length,
        min_length = min_length,
        length_penalty = 2.0,
        num_beams = 4,
        early_stopping=True
    )

    return summary[0]['summary_text']