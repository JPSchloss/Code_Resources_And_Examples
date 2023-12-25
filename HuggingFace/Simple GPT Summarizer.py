from transformers import GPT2LMHeadModel, GPT2Tokenizer

def load_model(model_name="gpt2"):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    return tokenizer, model

def summarize_text(text, tokenizer, model, max_length=100):
    prompt = "Summarize this text:\n" + text + "\n\nSummary:"
    inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=max_length, length_penalty=5.0, num_beams=2, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Load the model
tokenizer, model = load_model()

# Example text to summarize
text_to_summarize = """
[Insert a long piece of text here that you want to summarize]
"""


# Generate summary
summary = summarize_text(text_to_summarize, tokenizer, model)
print("Summary:", summary)
