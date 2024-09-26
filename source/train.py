import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import hamming_loss, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForSequenceClassification
import numpy as np
import json
from tqdm import tqdm, trange

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')


# Dataset class
class AAPDDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

# Load AAPD dataset from text files
def load_data(train_text_path, train_labels_path, dev_text_path, dev_labels_path, test_text_path, test_labels_path):
    def read_file(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f]

    train_texts = read_file(train_text_path)
    train_labels = [label.split() for label in read_file(train_labels_path)]
    dev_texts = read_file(dev_text_path)
    dev_labels = [label.split() for label in read_file(dev_labels_path)]
    test_texts = read_file(test_text_path)
    test_labels = [label.split() for label in read_file(test_labels_path)]

    # Use MultiLabelBinarizer for label encoding
    mlb = MultiLabelBinarizer()
    # Fit on all labels to ensure consistent encoding across all splits
    all_labels = train_labels + dev_labels + test_labels
    mlb.fit(all_labels)

    # Transform labels for each split
    train_binary_labels = mlb.transform(train_labels)
    dev_binary_labels = mlb.transform(dev_labels)
    test_binary_labels = mlb.transform(test_labels)

    return (train_texts, train_binary_labels), (dev_texts, dev_binary_labels), (test_texts, test_binary_labels), mlb

# Model definition
class SentenceEmbeddingClassifier(nn.Module):
    def __init__(self, pretrained_model, num_labels):
        super(SentenceEmbeddingClassifier, self).__init__()
        self.sentence_encoder = pretrained_model
        self.classifier = nn.Linear(self.sentence_encoder.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.sentence_encoder(input_ids=input_ids, attention_mask=attention_mask)
        sentence_embedding = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token embedding
        logits = self.classifier(sentence_embedding)
        return logits

# Training function with progress bar
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc="Training")
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
    return total_loss / len(train_loader)


# Evaluation function with progress bar
def evaluate(model, data_loader, criterion, device, mlb):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_texts = []
    
    progress_bar = tqdm(data_loader, desc="Evaluating")
    with torch.no_grad():
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_texts.extend(data_loader.dataset.texts[i] for i in range(len(labels)))

            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

    avg_loss = total_loss / len(data_loader)
    hamming_loss_score = hamming_loss(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='micro')
    recall = recall_score(all_labels, all_preds, average='micro')
    f1 = f1_score(all_labels, all_preds, average='micro')

    # Convert binary predictions back to string labels
    pred_labels = mlb.inverse_transform(np.array(all_preds))
    true_labels = mlb.inverse_transform(np.array(all_labels))

    results = {
        'metrics': {
            'hamming_loss': hamming_loss_score,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        },
        'predictions': [
            {'text': text, 'true_labels': true, 'predicted_labels': pred}
            for text, true, pred in zip(all_texts, true_labels, pred_labels)
        ]
    }

    return avg_loss, results


# Save results to JSON file
def save_results(results, filename):
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)


# Main training loop with progress tracking
def main():
    # Hyperparameters
    model_path = "nomic-ai/nomic-bert-2048"  # Replace with your fine-tuned model
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    pretrained_model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config, trust_remote_code=True)
    batch_size = 16
    num_epochs = 10
    learning_rate = 2e-5

    # Load data
    print("Loading data...")
    (train_texts, train_labels), (dev_texts, dev_labels), (test_texts, test_labels), mlb = load_data(
        'datasets/AAPD/text_train', 'datasets/AAPD/label_train',
        'datasets/AAPD/text_val', 'datasets/AAPD/label_val',
        'datasets/AAPD/text_test', 'datasets/AAPD/label_test'
    )
    num_labels = len(mlb.classes_)

    # Initialize tokenizer and datasets
    print("Initializing tokenizer and datasets...")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = AAPDDataset(train_texts, train_labels, tokenizer)
    dev_dataset = AAPDDataset(dev_texts, dev_labels, tokenizer)
    test_dataset = AAPDDataset(test_texts, test_labels, tokenizer)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize model, loss function, and optimizer
    print("Initializing model...")
    model = SentenceEmbeddingClassifier(pretrained_model, num_labels).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    best_f1_score = 0
    print("Starting training...")
    for epoch in trange(num_epochs, desc="Epochs"):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        dev_loss, dev_results = evaluate(model, dev_loader, criterion, device, mlb)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Dev Loss: {dev_loss:.4f}")
        print(f"Dev Metrics:")
        print(f"Hamming Loss: {dev_results['metrics']['hamming_loss']:.4f}")
        print(f"Precision: {dev_results['metrics']['precision']:.4f}")
        print(f"Recall: {dev_results['metrics']['recall']:.4f}")
        print(f"F1-score: {dev_results['metrics']['f1_score']:.4f}")
        
        # Save the best model based on F1-score
        if dev_results['metrics']['f1_score'] > best_f1_score:
            best_f1_score = dev_results['metrics']['f1_score']
            torch.save(model.state_dict(), 'best_model.pth')
            save_results(dev_results, 'best_model_dev_results.json')
            print("Best model saved!")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load('best_model.pth'))
    test_loss, test_results = evaluate(model, test_loader, criterion, device, mlb)
    print("\nTest Set Evaluation:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Hamming Loss: {test_results['metrics']['hamming_loss']:.4f}")
    print(f"Precision: {test_results['metrics']['precision']:.4f}")
    print(f"Recall: {test_results['metrics']['recall']:.4f}")
    print(f"F1-score: {test_results['metrics']['f1_score']:.4f}")
    save_results(test_results, 'test_results.json')

    print("Training and evaluation completed!")

if __name__ == "__main__":
    main()