import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
from IPython.display import clear_output
from torch.utils.data import random_split, Subset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import pandas as pd
import os
import copy

# import spike generation
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import spikegen


data_path = os.path.join(os.path.dirname(__file__), 'data')

# Parametri SNN
beta = 0.65  # decay rate
spike_grad = surrogate.fast_sigmoid()  # surrogate gradient function

""" Iperparametri"""
# Parametri visualizzazione
stampa_evaluation= False
stampa_training= False

""" Dichiarazione funzioni -- le sposteremo in un file a parte """
def loaders(train_batch_size, test_batch_size, train_set_size, test_set_size):

    """transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        Lambda(lambda x: torch.flatten(x))])"""
    
    transform = transforms.Compose([
    transforms.ToTensor(),  # Convert image to tensor with values in [0, 1]
    #transforms.Lambda(lambda x: 1 - x),  # Invert the pixel values
        transforms.Lambda(lambda x: torch.flatten(x))  # Flatten the image into a 1D vector
    ])
    
    # Carica il dataset MNIST
    train_dataset = MNIST(data_path, train=True, download=True, transform=transform)
    test_dataset = MNIST(data_path, train=False, download=True, transform=transform)

    # Crea sottoinsiemi dei dataset con le dimensioni desiderate
    train_subset, _ = random_split(train_dataset, [train_set_size, len(train_dataset) - train_set_size])
    test_subset, _ = random_split(test_dataset, [test_set_size, len(test_dataset) - test_set_size])

    # Crea i DataLoader per i sottoinsiemi dei dataset
    train_loader = DataLoader(train_subset, batch_size=train_batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_subset, batch_size=test_batch_size, shuffle=False, drop_last=True)
    return train_loader, test_loader

def visualize_sample(data, name='', idx=0):
    reshaped = data[idx].cpu().reshape(28, 28)
    plt.figure(figsize = (4, 4))
    plt.title(name)
    plt.imshow(reshaped, cmap="gray")
    plt.show()

def overlay_y_on_x(x,y):
    x_ = x.clone()
    x_[:, :10]*= 0.0
    x_[range(x.shape[0]), y] = x.max()
    return x_

class Layer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias, device, dtype)
        self.lif = snn.Leaky(beta=beta, spike_grad=spike_grad)
            
    def forward(self, x, mem):
        cur = self.linear(x)
        spk, mem = self.lif(cur, mem)
        return spk, mem

class MetaLayer(nn.Module):
    def __init__(self, self_size, pre_size, post_size, out_size):
        super(MetaLayer, self).__init__()
        self.layer_pre = Layer(pre_size, out_size)
        self.layer_post = Layer(post_size, out_size)
        self.layer_self = Layer(self_size, out_size)
        
        self.loss_history = list()
        self.threshold = threshold
        self.num_epochs = epochs
        self.opt = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
    def forward(self, x, mem_pre, mem_post, mem_self):
        in_self, in_pre, in_post = x  # unpacking della tupla
        
        # Forward pass attraverso i layer con gestione della memoria
        spk_pre, mem_pre_new = self.layer_pre(in_pre, mem_pre)
        spk_post, mem_post_new = self.layer_post(in_post, mem_post)
        spk_self, mem_self_new = self.layer_self(in_self, mem_self)
        
        # Combina gli output
        spk_out = 0.7 * (spk_pre + spk_post) + 0.3 * spk_self
        
        return spk_out, (mem_pre_new, mem_post_new, mem_self_new)

    def train(self, x_pos, x_neg):
        self.loss_history.append(list())
        
        # Inizializza stati di memoria
        batch_size = x_pos[0].size(0)
        mem_pre_pos = torch.zeros(batch_size, self.layer_pre.linear.out_features)
        mem_post_pos = torch.zeros(batch_size, self.layer_post.linear.out_features)
        mem_self_pos = torch.zeros(batch_size, self.layer_self.linear.out_features)
        
        mem_pre_neg = torch.zeros(batch_size, self.layer_pre.linear.out_features)
        mem_post_neg = torch.zeros(batch_size, self.layer_post.linear.out_features)
        mem_self_neg = torch.zeros(batch_size, self.layer_self.linear.out_features)

        

        for i in range(self.num_epochs):

            # Forward pass con x_pos
            spk_pos, (mem_pre_pos, mem_post_pos, mem_self_pos) = self.forward(x_pos, mem_pre_pos, mem_post_pos, mem_self_pos)
                
            # Forward pass con x_neg
            spk_neg, (mem_pre_neg, mem_post_neg, mem_self_neg) = self.forward(x_neg, mem_pre_neg, mem_post_neg, mem_self_neg)
                
            # Aggiorna gli stati di memoria per la prossima iterazione
            mem_pre_pos, mem_post_pos, mem_self_pos = mem_pre_pos, mem_post_pos, mem_self_pos
            mem_pre_neg, mem_post_neg, mem_self_neg = mem_pre_neg, mem_post_neg, mem_self_neg
                
            # Calcola loss usando spike rate
            g_pos = spk_pos
            g_neg = spk_neg
                
            # Calcolo della loss in modo compatibile con backward
            loss = torch.mean((g_pos - self.threshold)**2) - torch.mean((g_neg - self.threshold)**2)
                
            self.opt.zero_grad()
            loss.backward(retain_graph=True)
            self.opt.step()
            
            
            self.loss_history[-1].append(loss.item())
            
                
            if stampa_training:
                print(f"Loss all'epoca {i}: {loss.item()}")
        
        return spk_pos.detach(), spk_neg.detach()


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = []
 
        #Metalayer (self_size - pre_size - post_size - out_size)
        self.layers += [MetaLayer(size_Metalayer_uno,size_input,size_Metalayer_due,size_Metalayer_uno)]      
        self.layers += [MetaLayer(size_Metalayer_due,size_Metalayer_uno,size_Metalayer_tre,size_Metalayer_due)]    
        self.layers += [MetaLayer(size_Metalayer_tre,size_Metalayer_due,0,size_Metalayer_tre)] 

    def predict(self, x, test_batch_size):
        goodness_per_label = []
        result_tensor = torch.ones(test_batch_size, 1)
        
        for label in range(10):
            if stampa_evaluation:
                print("\n----LABEL NUMERO:", label,"-----")
            h = overlay_y_on_x(x, torch.full((x.size(0),), label))
            goodness = torch.zeros(test_batch_size)
            goodness_m1 = torch.zeros(test_batch_size, 1)
            goodness_m2 = torch.zeros(test_batch_size, 1)
            goodness_m3 = torch.zeros(test_batch_size, 1) 
            goodness_intime = []
            
            # Inizializza stati di memoria per ogni layer
            self_evaluation_list_next = [
                torch.zeros(h.shape[0], size_Metalayer_uno),
                torch.zeros(h.shape[0], size_Metalayer_due),
                torch.zeros(h.shape[0], size_Metalayer_tre),
                torch.zeros(h.shape[0], 0)
            ]
            
            # Simulazione temporale
            for time in range(test_time):
                if stampa_evaluation:
                    print("\n----ISTANTE TEMPORALE:", time,"-----")

                self_evaluation_list = self_evaluation_list_next.copy()
                
                # Forward pass attraverso i layer
                for i, layer in enumerate(self.layers):
                    pre_evaluation = self_evaluation_list[i-1] if i>0 else h
                    self_evaluation = self_evaluation_list[i]
                    post_evaluation = self_evaluation_list[i+1]
                    
                    # Calcola spikes e aggiorna memoria
                    spk_out, mems = layer.forward(
                        (self_evaluation, pre_evaluation, post_evaluation),
                        *mems if 'mems' in locals() else (None, None, None)
                    )
                    self_evaluation_list_next[i] = spk_out
                    
                    # Accumula goodness per timestep specifici
                    if time in range(1,test_time):
                        if stampa_evaluation:
                            print(f'.......................Evaluating Metalayer {i} .....................')
                        
                        spike_rate = spk_out.mean(dim=1, keepdim=True)
                        if i == 0:
                            goodness_m1 += spike_rate
                        elif i == 1:
                            goodness_m2 += spike_rate
                        elif i == 2:
                            goodness_m3 += spike_rate
                        
                        goodness = goodness_m1 + goodness_m2 + goodness_m3
                
                goodness_intime.append(goodness)
            
            # Assicurati che tutti i tensori abbiano la stessa dimensione
            test_batch_size = x.size(0)  # Ottieni la dimensione del batch dall'input
            goodness_intime = [g.view(test_batch_size, -1) for g in goodness_intime]
            total_goodness = torch.stack(goodness_intime).sum(dim=0)
            goodness_per_label.append(total_goodness)
        
        # Trova il label con la massima goodness
        goodness_stack = torch.stack(goodness_per_label, dim=1)
        result_tensor = torch.argmax(goodness_stack, dim=1, keepdim=True)
        
        return result_tensor

    def train(self, x_pos, x_neg):
        # Inizializza liste per gli stati di memoria
        self_pos_list_next = [
            torch.zeros(x_pos.shape[0], size_Metalayer_uno),
            torch.zeros(x_pos.shape[0], size_Metalayer_due),
            torch.zeros(x_pos.shape[0], size_Metalayer_tre),
            torch.zeros(x_pos.shape[0], 0)
        ]
        self_neg_list_next = [x.clone() for x in self_pos_list_next]

        # Training temporale
        for t in range(training_time):
            if(stampa_training==True):
                print(f"\n----ISTANTE TEMPORALE: {t} -----")
            
            self_pos_list = copy.deepcopy(self_pos_list_next)
            self_neg_list = copy.deepcopy(self_neg_list_next)

            # Training per ogni layer
            for i, layer in enumerate(self.layers):
                pre_pos = self_pos_list[i-1] if i>0 else x_pos
                self_pos = self_pos_list[i]
                post_pos = self_pos_list[i+1]
                            
                pre_neg = self_neg_list[i-1] if i>0 else x_neg
                self_neg = self_neg_list[i]
                post_neg = self_neg_list[i+1] 
                
                if(stampa_training==True):
                    print(f'.......................Training Metalayer {i} .....................')

                # Train layer e aggiorna stati
                spk_pos, spk_neg = layer.train(
                    (self_pos, pre_pos, post_pos),
                    (self_neg, pre_neg, post_neg)
                )
                
                self_pos_list_next[i] = spk_pos
                self_neg_list_next[i] = spk_neg

def evaluate_model(net, loader, batch_size, mode='test'):
    predicted_list = []
    real_list = []
    
    print(f"Numero di Batch nel {mode}_loader:", len(loader))
    
    for x_te, y_te in loader:
        # Predizione
        result = net.predict(x_te, batch_size).int()
        result_list = result.view(-1).tolist()
        labels = y_te.tolist()
        
        predicted_list.append(result_list)
        real_list.append(labels)
    
    # Appiattisci le liste
    predicted_flat = [item for sublist in predicted_list for item in sublist]
    real_flat = [item for sublist in real_list for item in sublist]
    
    # Calcola metriche
    correct = sum(p == r for p, r in zip(predicted_flat, real_flat))
    accuracy = round((correct / len(predicted_flat)) * 100, 2)
    precision = round(precision_score(real_flat, predicted_flat, average='macro', zero_division=1), 2)
    recall = round(recall_score(real_flat, predicted_flat, average='macro'), 2)
    f1 = round(f1_score(real_flat, predicted_flat, average='macro', zero_division=1), 3)
    
    if stampa_evaluation:
        print(f"\n{mode.capitalize()} Accuracy: {accuracy}%")
        print(f"{mode.capitalize()} Precision: {precision}")
        print(f"{mode.capitalize()} Recall: {recall}")
        print(f"{mode.capitalize()} F1 Score: {f1}")
    
    return {
        f'{mode}_accuracy': accuracy,
        f'{mode}_error': round(100 - accuracy, 2),
        f'{mode}_precision': precision,
        f'{mode}_recall': recall,
        f'{mode}_f1': f1
    }, confusion_matrix(real_flat, predicted_flat)

# Funzione per salvare e mostrare i grafici
def save_and_show_plot(fig, path, filename, show=False):
    plt.savefig(os.path.join(path, filename))
    if show:
        plt.show()
    plt.close(fig)

"""MAIN"""
trials_results = []
params_df = pd.read_csv('hyperparameters_spikenn.csv')

# Ciclo su tutte le righe
for index, row in params_df.iterrows():
    print(f"Test Numero:{row['trial_number']}")

    # Carica parametri dal CSV
    trial_number = row['trial_number'] 
    train_batch_size = row['train_batch_size']
    test_batch_size = row['test_batch_size']
    learning_rate = row['learning_rate']
    activation_function = eval(row['activation_function'])
    epochs = row['epochs']
    threshold = row['threshold'] 
    test_time = row['test_time']
    training_time = row['training_time']
    train_set_size = row['train_set_size']
    test_set_size = row['test_set_size']
    size_input = row['size_input']
    size_Metalayer_uno = row['size_metalayer_uno']
    size_Metalayer_due = row['size_metalayer_due']
    size_Metalayer_tre = row['size_metalayer_tre']

    # Stampa tabella parametri
    data_iperparametri = [
        ("Trial Number", trial_number),
        ("Train Batch Size", train_batch_size),
        ("Test Batch Size", test_batch_size),
        ("Learning Rate", learning_rate),
        ("Activation Function", activation_function),
        ("Epochs", epochs),
        ("Threshold", threshold),
        ("Test Time (s)", test_time),
        ("Training Time (s)", training_time),
        ("Train Set Size", train_set_size),
        ("Test Set Size", test_set_size),
        ("Size Input", size_input),
        ("Size Metalayer Uno", size_Metalayer_uno),
        ("Size Metalayer Due", size_Metalayer_due),
        ("Size Metalayer Tre", size_Metalayer_tre),
    ]

    # Formattazione e stampa della tabella
    col_width = max(len(str(value)) for row in data_iperparametri for value in row) + 2
    separator = '-' * (col_width * 2 + 1)
    print(separator)
    print(f"{'Parameter'.ljust(col_width)}| {'Value'.ljust(col_width)}")
    print(separator)
    for name, value in data_iperparametri:
        print(f"{name.ljust(col_width)}| {str(value).ljust(col_width)}")
    print(separator)

    # Setup data loaders
    train_loader, test_loader = loaders(train_batch_size, test_batch_size, train_set_size, test_set_size)

    if stampa_training:
        print("Lunghezza train loader", len(train_loader))
        print("Lunghezza test loader", len(test_loader))

    # Inizializzazione e training della rete
    net = Net()
    
    
    for x, y in tqdm(train_loader):
        x_pos = overlay_y_on_x(x, y)
        rnd = torch.randperm(x.size(0))
        x_neg = overlay_y_on_x(x, y[rnd])

        # Training
        net.train(x_pos, x_neg)

    # Creazione directory per i risultati
    nome_directory = f"test_{trial_number}"
    path_directory = os.path.join("testing_result", nome_directory)
    if not os.path.exists(path_directory):
        os.makedirs(path_directory)

    # Salvataggio grafico delle loss
    fig_loss = plt.figure(figsize=(8, 6))
    for layer_id in range(3):
        l = []
        for i in net.layers[layer_id].loss_history:
            l += i
        plt.plot(l, label=f'Layer {layer_id}')

    plt.xlabel('Epoche * Tempo')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss per Layer')
    save_and_show_plot(fig_loss, path_directory, "Loss media per Layer.png", show=False)

    # Calcolo loss finali
    loss_list_layer_zero = [item for sublist in net.layers[0].loss_history for item in sublist]
    loss_list_layer_uno = [item for sublist in net.layers[1].loss_history for item in sublist]
    loss_list_layer_due = [item for sublist in net.layers[2].loss_history for item in sublist]

    loss_layer_zero = loss_list_layer_zero[-1]
    loss_layer_uno = loss_list_layer_uno[-1]
    loss_layer_due = loss_list_layer_due[-1]

    # Valutazione sul training set
    train_metrics, train_conf_matrix = evaluate_model(net, train_loader, train_batch_size, 'training')
    
    # Valutazione sul test set
    test_metrics, test_conf_matrix = evaluate_model(net, test_loader, test_batch_size, 'testing')

    # Salvataggio matrici di confusione
    class_names = ["Zero", "Uno", "Due", "Tre", "Quattro", "Cinque", "Sei", "Sette", "Otto", "Nove"]
    
    # Training confusion matrix
    fig_train = plt.figure(figsize=(8, 6))
    sns.heatmap(train_conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Etichette Predette')
    plt.ylabel('Etichette Reali')
    plt.title('Matrice di Confusione in Training')
    save_and_show_plot(fig_train, path_directory, "Matrice di confusione in training.png", show=False)

    # Testing confusion matrix
    fig_test = plt.figure(figsize=(8, 6))
    sns.heatmap(test_conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Etichette Predette')
    plt.ylabel('Etichette Reali')
    plt.title('Matrice di Confusione Testing')
    save_and_show_plot(fig_test, path_directory, "Matrice di confusione in testing.png", show=False)

    # Preparazione risultati del trial
    trial_result = {
        'trial_number': trial_number,
        **train_metrics,
        **test_metrics,
        'Loss layer zero': loss_layer_zero,
        'Loss layer uno': loss_layer_uno,
        'Loss layer due': loss_layer_due,
        'overfitting_degree': train_metrics['training_accuracy'] - test_metrics['testing_accuracy']
    }

    # Salvataggio tabella risultati
    fig_table, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')
    table = ax.table(cellText=list(trial_result.items()), 
                    colLabels=['Parameter', 'Value'], 
                    loc='center', 
                    cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    save_and_show_plot(fig_table, path_directory, "Tabella_parametri.png", show=False)

    # Aggiungi i risultati alla lista
    trials_results.append(trial_result)

# Salvataggio finale dei risultati
df_results = pd.DataFrame(trials_results)
df_results.to_csv('trials_results.csv', index=False)
df_results.to_excel('trials_results.xlsx', index=False)
print("File CSV ed Excel con i risultati delle prove creati con successo.")
print("File CSV ed Excel con i risultati delle prove creati con successo.")