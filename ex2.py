

###################################################
# Exercise 2 - Natural Language Processing 67658  #
###################################################
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.datasets import fetch_20newsgroups
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
print("im here")
NUM_OF_EPOCHS = 20 
SINGLE = "single"
MULTY = "multy"


# subset of categories that we will use
category_dict = {'comp.graphics': 'computer graphics',
                'rec.sport.baseball': 'baseball',
                'sci.electronics': 'science, electronics',
                'talk.politics.guns': 'politics, guns'
                }


class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# class Mlp_Classifier:
#     def __init__()




class Perceptron(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim = 500,model_type = "single", lr = 1e-3):
        super(Perceptron, self).__init__()
        self.model_type = model_type
        if  self.model_type == SINGLE:
            self.model = nn.Linear(input_dim, num_classes)
        elif self.model_type == MULTY:
            self.hidden_layer = nn.Linear(input_dim, hidden_dim)
            self.output_layer = nn.Linear(hidden_dim, num_classes)
            self.relu = nn.ReLU()
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr) 

    def forward(self, x):
        if self.model_type  == SINGLE:
            return self.model(x)
        elif self.model_type == MULTY:
            x = self.relu(self.hidden_layer(x))  # Pass through hidden layer with ReLU activation
            return self.output_layer(x)
    
    
    def train_step(self, batch_X, batch_y):
        """
        Perform a single training step.
        :param batch_X: Input features for the batch.
        :param batch_y: True labels for the batch.
        :return: Loss for the batch.
        """
        self.optimizer.zero_grad()  # Zero gradients
        outputs = self.forward(batch_X)  # Forward pass
        loss = self.criterion(outputs, batch_y)  # Compute loss
        loss.backward()  # Backpropagation
        self.optimizer.step()  # Update parameters
        return loss.item()



def get_data(categories=None, portion=1.):
    """
    Get data for given categories and portion
    :param portion: portion of the data to use
    :return:
    """
    # get data
    from sklearn.datasets import fetch_20newsgroups
    data_train = fetch_20newsgroups(categories=categories, subset='train', remove=('headers', 'footers', 'quotes'),
                                    random_state=21)
    data_test = fetch_20newsgroups(categories=categories, subset='test', remove=('headers', 'footers', 'quotes'),
                                random_state=21)

    # train
    train_len = int(portion*len(data_train.data))
    x_train = np.array(data_train.data[:train_len])
    y_train = data_train.target[:train_len]
    # remove empty entries
    non_empty = x_train != ""
    x_train, y_train = x_train[non_empty].tolist(), y_train[non_empty].tolist()

    # test
    x_test = np.array(data_test.data)
    y_test = data_test.target
    non_empty = np.array(x_test) != ""
    x_test, y_test = x_test[non_empty].tolist(), y_test[non_empty].tolist()
    return x_train, y_train, x_test, y_test




# Q1,2

def MLP_classification(portion=1., model=None):
    x_train, y_train, x_test, y_test = get_data(categories=category_dict.keys(), portion=portion)

    vectorizer = TfidfVectorizer(max_features=2000)
    x_train_tfidf = vectorizer.fit_transform(x_train).toarray()
    x_test_tfidf = vectorizer.transform(x_test).toarray()
    x_train_tensor = torch.tensor(x_train_tfidf, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    x_test_tensor = torch.tensor(x_test_tfidf, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    num_features = x_train_tfidf.shape[1]  # TF-IDF feature size
    num_classes = len(np.unique(y_train))  # Number of classes
    model = Perceptron(input_dim=num_features, num_classes=num_classes,model_type = model)

    train_dataset = TextDataset(x_train_tensor, y_train_tensor)
    
    # Split the training data into training and validation sets (80% training, 20% validation)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    train_losses = []
    val_accuracies = []

    for epoch in range(NUM_OF_EPOCHS):
        model.train()
        epoch_loss = 0.0
        correct_preds = 0
        total_preds = 0

        for batch_X, batch_y in train_loader:
            batch_loss = model.train_step(batch_X, batch_y)  # Train step
            epoch_loss += batch_loss

            # For calculating accuracy
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == batch_y).sum().item()
            total_preds += batch_y.size(0)

        # Calculate average loss per epoch
        train_losses.append(epoch_loss / len(train_loader))
        
        # Calculate accuracy on validation set after each epoch
        model.eval()
        val_correct_preds = 0
        val_total_preds = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                _, predicted = torch.max(outputs, 1)
                val_correct_preds += (predicted == batch_y).sum().item()
                val_total_preds += batch_y.size(0)

        val_accuracy = val_correct_preds / val_total_preds
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch + 1}/{NUM_OF_EPOCHS}, Loss: {train_losses[-1]:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    model.eval()
    with torch.no_grad():
        y_pred = model(x_test_tensor).argmax(dim=1).numpy()
        accuracy = accuracy_score(y_test, y_pred)
    print(f"Final Test Accuracy: {accuracy:.4f}")
    
    return train_losses, val_accuracies


# def MLP_classification(portion=1., model=None):
#     """
#     Perform linear classification
#     :param portion: portion of the data to use
#     :return: classification accuracy
#     """
#     print("start get data")
#     x_train, y_train, x_test, y_test = get_data(categories=category_dict.keys(), portion=portion)
#     print("finish get data")


#     vectorizer = TfidfVectorizer(max_features=2000)
#     x_train_tfidf = vectorizer.fit_transform(x_train).toarray()
#     x_test_tfidf = vectorizer.transform(x_test).toarray()
#     x_train_tensor = torch.tensor(x_train_tfidf, dtype=torch.float32)
#     y_train_tensor = torch.tensor(y_train, dtype=torch.long)
#     x_test_tensor = torch.tensor(x_test_tfidf, dtype=torch.float32)
#     y_test_tensor = torch.tensor(y_test, dtype=torch.long)
#     num_features = x_train_tfidf.shape[1]  # TF-IDF feature size
#     num_classes = len(np.unique(y_train))  # Number of classes
#     model = model(input_dim=num_features, num_classes=num_classes)

#     train_dataset = TextDataset(x_train_tensor, y_train_tensor)
#     train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)


#     print("finish data load ")
#         # Initialize model if not provided
#     num_features = x_train_tfidf.shape[1]
#     num_classes = len(np.unique(y_train))
#     train_losses = []
#     val_accuracies = []
#     for epoch in range(NUM_OF_EPOCHS):
#         print(f"starting epoch {epoch}")
#         model.train()
#         epoch_loss = 0.0
#         for batch_X, batch_y in train_loader:
#             batch_loss = model.train_step(batch_X, batch_y)  # Use the model's train_step
#             epoch_loss += batch_loss

#         train_losses.append(epoch_loss / len(train_loader))  # Average loss for this epoch
#         print(f"Epoch {epoch + 1}, Loss: {train_losses[-1]:.4f}")

#         model.eval()
#         with torch.no_grad():
#             y_pred = model(x_test_tensor).argmax(dim=1).numpy()
#             accuracy = accuracy_score(y_test, y_pred)
#             val_accuracies.append(accuracy)  # Append accuracy for this epoch

#         print(f"Epoch {epoch + 1}, Loss: {train_losses[-1]:.4f}, Accuracy: {val_accuracies[-1]:.4f}")
#     return train_losses, val_accuracies



def plot_results(portions, all_train_losses, all_val_accuracies,title = ""):
    """
    Plot training losses and validation accuracies for each portion.
    :param portions: List of data portions.
    :param all_train_losses: List of train losses for each portion.
    :param all_val_accuracies: List of validation accuracies for each portion.
    """
    for i, portion in enumerate(portions):
        train_losses = all_train_losses[i]
        val_accuracies = all_val_accuracies[i]

        # Plot Training Loss
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, NUM_OF_EPOCHS + 1), train_losses, label="Train Loss", color="blue")
    
        plt.title(f"Train Loss (Portion={portion}) for {title}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        # Plot Validation Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(range(1, NUM_OF_EPOCHS + 1), val_accuracies, label="Validation Accuracy", color="green")
        plt.title(f"Validation Accuracy (Portion={portion}) for {title}")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.tight_layout()
        plot_filename = f"./{title}_portion_{portion}.png"
        plt.savefig(plot_filename)
        print(f"Plot saved as {plot_filename}")
        plt.show()
        


# Q3
def transformer_classification(portion=1. ,dev = "dev" ):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from torch.utils.data import DataLoader
    import evaluate
    from tqdm import tqdm
    from torch.optim import AdamW

    class Dataset(torch.utils.data.Dataset):
        """
        Dataset for loading data
        """
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

        def __len__(self):
            return len(self.labels)

    def train_epoch(model, data_loader, optimizer, dev='cpu'):
        """
        Perform an epoch of training of the model with the optimizer
        :param model:
        :param data_loader:
        :param optimizer:
        :param dev:
        :return: Average loss over the epoch
        """
        model.train()
        total_loss = 0.
        # iterate over batches
        for batch in tqdm(data_loader):
            input_ids = batch['input_ids'].to(dev)
            attention_mask = batch['attention_mask'].to(dev)
            labels = batch['labels'].to(dev)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            # Backward pass
            optimizer.zero_grad()  # Clear gradients from the previous step
            loss.backward()        # Compute gradients
            optimizer.step()       # Update the model's parameters

            # Accumulate loss
            total_loss += loss.item()
            # Calculate average loss
        avg_loss = total_loss / len(data_loader)
        return avg_loss

    def evaluate_model(model, data_loader, dev='cpu', metric=None):
        """
        model - 
        data_loader - validation set 
        """
        model.eval()
        all_predictions = []  # List to store predictions
        all_labels = []  # List to store true labels

        with torch.no_grad():
            for batch in tqdm(data_loader):
                input_ids = batch['input_ids'].to(dev)
                attention_mask = batch['attention_mask'].to(dev)
                labels = batch['labels'].to(dev)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits  # Get raw predictions (logits)
                predictions = torch.argmax(logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy()) # numpy required working on CPU
                all_labels.extend(labels.cpu().numpy()) # # numpy required working on CPU
            # Use the provided metric to compute evaluation score (e.g., accuracy)
        result = metric.compute(predictions=all_predictions, references=all_labels)
        return result

    x_train, y_train, x_test, y_test = get_data(categories=category_dict.keys(), portion=portion)

    # Parameters
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_labels = len(category_dict)
    epochs = 3
    batch_size = 16
    learning_rate = 5e-5


    # Model, tokenizer, and metric
    model = AutoModelForSequenceClassification.from_pretrained('distilroberta-base', num_labels=num_labels).to(dev)
    optimizer = AdamW(model.parameters(), lr= learning_rate)
    tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
    metric = evaluate.load("accuracy") # 

    # Datasets and DataLoaders
    train_dataset = Dataset(tokenizer(x_train, truncation=True, padding=True), y_train)
    val_dataset = Dataset(tokenizer(x_test, truncation=True, padding=True), y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    total_epochs_losses = []
    total_epochs_eval = []
    for epoch in range(epochs):
        epoc_losses = train_epoch(model=model, data_loader=train_loader, optimizer=optimizer,dev = dev)
        total_epochs_losses.append(epoc_losses)
        eval_result =evaluate_model(model=model, data_loader=val_loader,dev = dev, metric=metric)
        total_epochs_eval.append(eval_result['accuracy'])

    return total_epochs_losses , total_epochs_eval


if __name__ == "__main__":
    # Q1- Single Layer perceptron
    portions = [0.1, 0.2, 0.5, 1.0]
    print("start running - Single layer perceptron for Q1")
    all_train_losses = []
    all_val_accuracies = []

    for portion in portions:
        print(f"Running for portion={portion}")
        train_losses, val_accuracies = MLP_classification(portion=portion, model=Perceptron)
        all_train_losses.append(train_losses)
        all_val_accuracies.append(val_accuracies)

    # Plot results
    plot_results(portions, all_train_losses, all_val_accuracies)

    # Q2 - multi-layer MLP
    pass

    # Q3 - Transformer
    print("\nTransformer results:")
    for p in portions[:2]:
        print(f"Portion: {p}")
        transformer_classification(portion=p)
