from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

class SentimentAnalyzer:
    def __init__(self):
        self.model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.label_mapping = {
            0: "very negative",
            1: "negative",
            2: "neutral",
            3: "positive",
            4: "very positive"
        }

    def analyze(self, text):
        # Tokenize and encode the text
        inputs = self.tokenizer(
            text, padding=True, truncation=True, return_tensors="pt", max_length=512
        )

        # Perform inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Compute softmax values to get probabilities
        probabilities = F.softmax(logits, dim=1)

        # Get the label with highest probability
        predicted_label_idx = torch.argmax(probabilities, dim=1).item()
        predicted_label_str = self.label_mapping[predicted_label_idx]

        return predicted_label_str, probabilities

def main():
    # Initialize the SentimentAnalyzer
    analyzer = SentimentAnalyzer()

    # Unlabeled complex real-world text dataset
    dataset = [
        "The international conference on machine learning was a pivotal moment for the field, showcasing groundbreaking research.",
        "Despite its groundbreaking achievements, the project failed to attract sufficient funding for the next research cycle.",
        "The product launch exceeded all expectations, generating unprecedented user engagement metrics.",
        "While the graphical interface is stunning, the software suffers from multiple severe bugs rendering it almost unusable."
    ]

    for text in dataset:
        predicted_label, probabilities = analyzer.analyze(text)
        print(f"Text: {text}")
        print(f"Predicted Label: {predicted_label}")
        print(f"Predicted Probabilities: {probabilities}")
        print("---")

if __name__ == "__main__":
    main()
