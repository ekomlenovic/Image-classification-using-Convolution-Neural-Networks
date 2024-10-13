import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from optuna.samplers import TPESampler
from optuna.pruners import SuccessiveHalvingPruner
from torch.nn import functional as F

from sklearn.metrics import confusion_matrix
import seaborn as sns
import optuna
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



# Load and preprocess the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Define the CNN model
class CNN(nn.Module):
    def __init__(self, conv1_filters, conv2_filters, fc_units, dropout_rate):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, conv1_filters, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(conv1_filters)
        self.conv2 = nn.Conv2d(conv1_filters, conv2_filters, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(conv2_filters)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(conv2_filters * 7 * 7, fc_units)
        self.fc2 = nn.Linear(fc_units, 10)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# Define the objective function for Optuna
def objective(trial):
    writer = SummaryWriter()
    # Hyperparameters to be tuned
    best_model = None
    conv1_filters = trial.suggest_categorical('conv1_filters', [16, 32, 64])
    conv2_filters = trial.suggest_categorical('conv2_filters', [32, 64, 128])
    fc_units = trial.suggest_categorical('fc_units', [64, 128, 256])
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [0.9, 0.1])
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # Create model, loss function, and optimizer
    model = CNN(conv1_filters, conv2_filters, fc_units, dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(10):
        print(f"Epoch {epoch+1}")
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        print(f"Loss: {loss.item()}")
        writer.add_scalar('Loss', loss.item(), epoch)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()

        val_accuracy = val_correct / val_total
        print(f"Validation Accuracy: {val_accuracy}")
        writer.add_scalar('Validation Accuracy', val_accuracy, epoch)
        trial.report(val_accuracy, epoch)

        if trial.should_prune():
            raise optuna.TrialPruned()


    if best_model is None or val_accuracy > best_model['val_accuracy']:
        best_model = {
            'model': model,
            'conv1_filters': conv1_filters,
            'conv2_filters': conv2_filters,
            'fc_units': fc_units,
            'dropout_rate': dropout_rate,
            'lr': lr,
            'batch_size': batch_size
        }



        model.eval()
        correct = 0
        total = 0
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                c = (predicted == target).squeeze()
                for i in range(len(target)):
                    label = target[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        accuracy = correct / total

        writer.add_hparams(
            {
                'conv1_filters': conv1_filters,
                'conv2_filters': conv2_filters,
                'fc_units': fc_units,
                'dropout_rate': dropout_rate,
                'lr': lr,
                'batch_size': batch_size
            },
            {
                'val_accuracy': val_accuracy,
                'accuracy': accuracy
            }
        )
        
        writer.add_scalar('Test Accuracy', accuracy, 0)

        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

        for i in range(10):
            print('Accuracy of %5s : %2d %%' % (class_names[i], 100 * class_correct[i] / class_total[i]))


        y_pred = []
        y_true = []

        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            output = model(inputs)

            output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
            y_pred.extend(output)
            
            labels = labels.data.cpu().numpy()
            y_true.extend(labels)

        # Build confusion matrix
        cf_matrix = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(12, 8))
        sns.heatmap(cf_matrix, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.savefig('data/confusion_matrix.png')
    return accuracy


        

# Run Optuna optimization
sampler = TPESampler()

pruner = SuccessiveHalvingPruner()

study = optuna.create_study(
    direction='maximize',
    sampler=sampler,
    pruner=pruner
)

study.optimize(objective, n_trials=2)

print('Number of finished trials:', len(study.trials))
print('Best trial:')
trial = study.best_trial
print('  Value: ', trial.value)
print('  Params: ')
for key, value in trial.params.items():
    print('    {}: {}'.format(key, value))