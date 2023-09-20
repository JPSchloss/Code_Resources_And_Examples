from tqdm import tqdm
import csv
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Clear CUDA cache to free up memory
torch.cuda.empty_cache()

# Check if a GPU is available and set it as the default device; otherwise, use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SentimentClassifier:
    def __init__(self, model_name='distilbert-base-uncased', num_labels=2):
        # Initialize the tokenizer and model
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        
        # Set the device and move the model to the device
        self.device = device
        self.model.to(self.device)

    # Load and tokenize data from a CSV file, then create DataLoaders.
    def prepare_data(self, file_path, batch_size=2000):
        reviews, labels = [], []
        input_ids_list, attention_mask_list = [], []

        # Load reviews and labels from CSV
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in tqdm(reader, desc="Reading Data", unit="lines"):
                review, sentiment = row['text'], 1 if int(float(row['stars'])) > 3 else 0
                reviews.append(review)
                labels.append(sentiment)

        # Tokenize reviews in batches
        for i in tqdm(range(0, len(reviews), batch_size), desc="Tokenizing", unit="batch"):
            batch = reviews[i:i+batch_size]
            encodings = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            input_ids_list.append(encodings['input_ids'])
            attention_mask_list.append(encodings['attention_mask'])

        # Create datasets and DataLoaders
        input_ids = torch.cat(input_ids_list, dim=0)
        attention_mask = torch.cat(attention_mask_list, dim=0)
        labels = torch.tensor(labels)
        
        dataset = TensorDataset(input_ids, attention_mask, labels)
        train_size, val_size = int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        return DataLoader(train_dataset, batch_size=16, 
                          shuffle=True, num_workers=4), DataLoader(val_dataset, batch_size=16, num_workers=4)




    # Train the model on the given data.
    def train(self, train_dataloader, val_dataloader, epochs=2, lr=2e-5, accum_steps=4):
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        best_val_loss = float("inf")
        epochs_without_improvement = 0

        # Training loop
        for epoch in range(epochs):
            print(f"Epoch: {epoch+1}")
            self.model.train()

            for step, batch in enumerate(tqdm(train_dataloader, desc="Training", unit="batch")):
                input_ids, attention_mask, labels = [item.to(self.device) for item in batch]

                # Forward pass and loss computation
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / accum_steps
                loss.backward()

                # Backward pass and optimization
                if (step + 1) % accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            # Validation loop
            self.model.eval()
            val_loss = 0

            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc="Validation", unit="batch"):
                    input_ids, attention_mask, labels = [item.to(self.device) for item in batch]
                    outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                    val_loss += outputs.loss.item()

            avg_val_loss = val_loss / len(val_dataloader)
            print(f"Validation Loss: {avg_val_loss}")

            # Early stopping logic
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save_model("sentiment_classifier_model.pth")
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= 3:
                    print("Early stopping")
                    break

    # Save the model to a file.
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    # Load a model from a file.
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    # Predict the sentiment of a given text.
    def predict(self, text):
        self.model.eval()
        tokens = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(self.device)

        with torch.no_grad():
            logits = self.model(**tokens).logits
            prediction = torch.argmax(logits, dim=-1).item()

        return "Positive" if prediction else "Negative"

if __name__ == "__main__":
    print('Step 1: Initializing the classifier...')
    classifier = SentimentClassifier()

    print('Step 2: Preparing the data...')
    train_dataloader, val_dataloader = classifier.prepare_data("sampled_yelp_reviews.csv")

    print('Step 3: Training the classifier...')
    classifier.train(train_dataloader, val_dataloader, epochs=2)

    print('Step 4: Saving the trained model...')
    classifier.save_model("sentiment_classifier_model.pth")

    print('Step 5: (Optional) Loading the saved model...')
    classifier.load_model("sentiment_classifier_model.pth")

    print('Step 6: Making a prediction...')
    sentiment = classifier.predict("The waiter didnt greet us when we first sat down!")
    print(f"Sentiment: {sentiment}")
