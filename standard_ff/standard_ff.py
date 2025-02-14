import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
from IPython.display import clear_output
import pandas as pd
import os
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np


data_path = os.path.join(os.path.dirname(__file__), 'data')

def MNIST_loaders(train_batch_size, test_batch_size):

    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        Lambda(lambda x: torch.flatten(x))])

    train_loader = DataLoader(
        MNIST(data_path, train=True,
              download=True,
              transform=transform),
        batch_size=train_batch_size, shuffle=True)

    test_loader = DataLoader(
        MNIST(data_path, train=False,
              download=True,
              transform=transform),
        batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader


def overlay_y_on_x(x, y):
    """Replace the first 10 pixels of data [x] with one-hot-encoded label [y]
    """
    x_ = x.clone()
    x_[:, :10] *= 0.0
    x_[range(x.shape[0]), y] = x.max()
    return x_


class Net(torch.nn.Module):

    def __init__(self, dims, threshold, epochs, learning_rate, activation_function):
        super().__init__()
        self.layers = []
        for d in range(len(dims) - 1):
            self.layers += [Layer(dims[d], dims[d + 1], threshold, epochs, learning_rate, activation_function)]

    def predict(self, x):
        goodness_per_label = []
        for label in range(10):
            h = overlay_y_on_x(x, label)
            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness += [h.pow(2).mean(1)]
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label.argmax(1)

    def train(self, x_pos, x_neg):
        h_pos, h_neg = x_pos, x_neg
        for i, layer in enumerate(self.layers):
            
            #print('training layer', i, '...')
            h_pos, h_neg = layer.train(h_pos, h_neg)


class Layer(nn.Linear):
    def __init__(self, in_features, out_features, threshold, epochs, learning_rate, activation_function,
                 bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.relu = activation_function
        self.opt = Adam(self.parameters(), lr=learning_rate)
        self.threshold = threshold
        self.num_epochs = epochs
        self.losses = []  # Lista per salvare le loss

    def forward(self, x):
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        return self.relu(
            torch.mm(x_direction, self.weight.T) +
            self.bias.unsqueeze(0))

    def train(self, x_pos, x_neg):
        g_pos = self.forward(x_pos).pow(2).mean(1)
        g_neg = self.forward(x_neg).pow(2).mean(1)
        loss = torch.log(1 + torch.exp(torch.cat([-g_pos + self.threshold, 
                                                    g_neg - self.threshold]))).mean()
            
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
            
            
        self.losses.append(loss.item())
        
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()

    
def visualize_sample(data, name='', idx=0):
    reshaped = data[idx].cpu().reshape(28, 28)
    plt.figure(figsize = (4, 4))
    plt.title(name)
    plt.imshow(reshaped, cmap="gray")
    plt.show()
    
params_df = pd.read_csv('hyperparameters.csv')

if __name__ == "__main__":
    #torch.manual_seed(1234)
    
    #loss1, loss2, loss3 = [], [], []
    # DataFrame per salvare tutti i risultati
    all_results_df = pd.DataFrame()
    
    for index,row in params_df.iterrows():
        loss1, loss2, loss3 = [], [], []
        
        print(f"Test Numero:{row['trial_number']}")

        trial_number=row['trial_number'] 
        train_batch_size=row['train_batch_size']
        test_batch_size=row['test_batch_size']
        learning_rate=row['learning_rate']
        activation_function= eval(row['activation_function'])
        epochs=row['epochs']
        threshold=row['threshold']
        size_input=row['size_input']
        size_layer_uno=row['size_layer_uno']
        size_layer_due=row['size_layer_due']
        size_layer_tre=row['size_layer_tre']

        data_iperparametri = [
            ("Trial Number", trial_number),
            ("Train Batch Size", train_batch_size),
            ("Test Batch Size", test_batch_size),
            ("Learning Rate", learning_rate),
            ("Activation Function", activation_function),
            ("Epochs", epochs),
            ("Threshold", threshold),
            ("Size Input", size_input),
            ("Size Metalayer Uno", size_layer_uno),
            ("Size Metalayer Due", size_layer_due),
            ("Size Metalayer Tre", size_layer_tre),
        ]
        train_loader, test_loader = MNIST_loaders(train_batch_size, test_batch_size)

        # Crea directory per i risultati del trial
        trial_dir = f"trial_{trial_number}"
        if not os.path.exists(trial_dir):
            os.makedirs(trial_dir)

        # Training e valutazione
        net = Net([size_input, size_layer_uno, size_layer_due, size_layer_tre], 
                 threshold=threshold, 
                 epochs=epochs, learning_rate=learning_rate, 
                 activation_function=activation_function)
        
        # Liste per salvare predizioni e valori reali
        all_predictions = []
        all_true_labels = []
        
        for i in range(1, epochs+1):
            print(f"Epoch {i}")
            # Prima esegui il training che popola losses
            for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {i}", leave=False):
                x, y = batch_x, batch_y
                x_pos = overlay_y_on_x(x, y)
                rnd = torch.randperm(x.size(0))
                x_neg = overlay_y_on_x(x, y[rnd])
                
                net.train(x_pos, x_neg)
            
            # Poi stampa le losses
            for l, layer in enumerate(net.layers):
                if layer.losses:  # Verifica che ci siano losses da stampare
                    print(f'Loss at step {i}: {layer.losses[-1]}')
                    if l==0:
                        loss1.append(layer.losses[-1])
                    elif l==1:
                        loss2.append(layer.losses[-1])
                    elif l==2:
                        loss3.append(layer.losses[-1])
            
        # Salva grafico delle loss
        plt.figure(figsize=(10, 6))
        epochs_range = np.arange(1, len(loss1) + 1)  # Crea un array che parte da 1
        plt.plot(epochs_range, loss1, label='Layer 1')
        plt.plot(epochs_range, loss2, label='Layer 2')
        plt.plot(epochs_range, loss3, label='Layer 3')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Loss per Layer - Trial {trial_number}')
        plt.legend()
        plt.savefig(os.path.join(trial_dir, 'loss_plot.png'))
        plt.close()

        # Calcola metriche complete sul test set
        test_predictions = []
        test_true = []
        test_error_sum = 0
        total_test_samples = 0

        for x_te, y_te in test_loader:
            pred = net.predict(x_te)
            batch_test_error = 1.0 - pred.eq(y_te).float().mean().item()
            test_error_sum += batch_test_error * len(x_te)
            total_test_samples += len(x_te)
            test_predictions.extend(pred.cpu().numpy())
            test_true.extend(y_te.cpu().numpy())

        # Calcola l'errore medio sul test set
        test_error = test_error_sum / total_test_samples

        # Calcola l'errore sul training set
        train_predictions = []
        train_true = []
        train_error_sum = 0
        total_train_samples = 0

        for x_tr, y_tr in train_loader:
            pred = net.predict(x_tr)
            batch_train_error = 1.0 - pred.eq(y_tr).float().mean().item()
            train_error_sum += batch_train_error * len(x_tr)
            total_train_samples += len(x_tr)
            train_predictions.extend(pred.cpu().numpy())
            train_true.extend(y_tr.cpu().numpy())

        # Calcola l'errore medio sul training set
        train_error = train_error_sum / total_train_samples

        # Calcola metriche finali
        conf_matrix = confusion_matrix(test_true, test_predictions)
        classification_metrics = classification_report(
            test_true, 
            test_predictions, 
            output_dict=True,
            zero_division=0
        )
            
        # Salva matrice di confusione come immagine
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - Trial {trial_number}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(trial_dir, 'confusion_matrix.png'))
        plt.close()

        # Prepara i risultati dettagliati
        results_dict = {
            'trial_number': trial_number,
            'train_batch_size': train_batch_size,
            'test_batch_size': test_batch_size,
            'learning_rate': learning_rate,
            'activation_function': str(activation_function),
            'epochs': epochs,
            'threshold': threshold,
            'size_input': size_input,
            'size_layer_uno': size_layer_uno,
            'size_layer_due': size_layer_due,
            'size_layer_tre': size_layer_tre,
            'final_loss_layer1': loss1[-1] if loss1 else None,
            'final_loss_layer2': loss2[-1] if loss2 else None,
            'final_loss_layer3': loss3[-1] if loss3 else None,
            'train_error': train_error,
            'test_error': test_error,
            'train_accuracy': (1-train_error)*100,
            'test_accuracy': (1-test_error)*100,
            'macro_precision': classification_metrics['macro avg']['precision'],
            'macro_recall': classification_metrics['macro avg']['recall'],
            'macro_f1': classification_metrics['macro avg']['f1-score'],
            'weighted_precision': classification_metrics['weighted avg']['precision'],
            'weighted_recall': classification_metrics['weighted avg']['recall'],
            'weighted_f1': classification_metrics['weighted avg']['f1-score']
        }

        # Salva risultati del trial corrente
        trial_results_df = pd.DataFrame([results_dict])
        trial_results_df.to_csv(os.path.join(trial_dir, 'detailed_results.csv'), index=False)
            
    # Aggiungi i risultati al DataFrame generale
    all_results_df = pd.concat([all_results_df, trial_results_df], ignore_index=True)

    # Dopo ogni trial, salva il DataFrame completo
    all_results_df.to_excel('all_trials_results.xlsx', index=False)