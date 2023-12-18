# Import necessary libraries
import torch
from transformers import BartTokenizer, BartForConditionalGeneration

# Define a class to handle review summarization tasks
class ReviewSummarizer:
    def __init__(self, model_name='facebook/bart-large-cnn'):
        # Initialize the tokenizer and model with the given model name (or the default BART model)
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        
        # Check if a GPU is available and set it as the default device; otherwise, use CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move the model to the appropriate device
        self.model.to(self.device)

    def predict(self, text):
        # Set the model to evaluation mode for inference
        self.model.eval()
        
        # Tokenize the given text, ensuring it doesn't exceed the max length and is prepared for the model
        tokens = self.tokenizer([text], max_length=1024, return_tensors='pt', truncation=True).to(self.device)
        
        # Generate a summary using the model; set the beam size, min and max summary length, and enable early stopping
        summary_ids = self.model.generate(tokens['input_ids'], num_beams=4, min_length=30, max_length=100, early_stopping=True)
        
        # Decode the generated tokens to produce the summary text
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        return summary

if __name__ == "__main__":
    # Step 1: Initialize the summarizer
    print('Step 1: Initializing the summarizer...')
    summarizer = ReviewSummarizer()

    # Step 2: Make a prediction (summarize a review)
    print('Step 2: Building summary...')
    # Provide a review text as input to the summarizer
    summary = summarizer.predict('''
                                Food came up pretty quickly but all orders whether said in dining or not are packed in to go bags.Seating ok. 
                                DoorDash and Uber eats drivers a plenty in here they seem to be frustrated because the orders may take longer 
                                than the order time quota is off. I liked the ice cream they had it was nice and kinda the fries but the 
                                bathroom was horrendous never will I use the bathroom there otherwise it's nice in there
                                ''')
    # Print the generated summary
    print(f"Summary: {summary}")

