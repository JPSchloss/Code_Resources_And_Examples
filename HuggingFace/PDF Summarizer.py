import PyPDF2
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

def load_model(model_name="gpt2"):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    return tokenizer, model

def summarize_text(text, tokenizer, model, max_length=100):
    # Truncate the text to a maximum number of tokens (512 for GPT-2)
    max_tokens = 512 - max_length
    tokens = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=max_tokens)
    
    # Generate the prompt for summarization
    prompt = "Summarize this text:\n"
    prompt_tokens = tokenizer.encode(prompt, return_tensors='pt')
    
    # Concatenate prompt and text tokens, ensuring not to exceed the model's maximum input length
    inputs = torch.cat([prompt_tokens, tokens], dim=-1)
    inputs = inputs[:, :512]

    # Set attention mask
    attention_mask = torch.ones(inputs.shape)
    attention_mask[:, len(prompt_tokens[0]):len(prompt_tokens[0]) + len(tokens[0])] = 0

    # Generate the summary
    summary_ids = model.generate(inputs, max_length=max_length, length_penalty=5.0, num_beams=2, early_stopping=True, attention_mask=attention_mask)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def extract_text_from_pdf(pdf_file):
    with open(pdf_file, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
    return text

# Load the model
tokenizer, model = load_model()

# Load and extract text from PDF
pdf_file = '/Users/jonathanschlosser/Desktop/2111.02187.pdf' #'path/to/your/pdf/file.pdf'  # Replace with your PDF file path
extracted_text = extract_text_from_pdf(pdf_file)

# Generate summary
summary = summarize_text(extracted_text, tokenizer, model)
print("Summary:", summary)
