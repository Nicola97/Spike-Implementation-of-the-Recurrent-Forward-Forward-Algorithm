import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import Adam
from torchvision.datasets import MNIST
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
import numpy as np
import statistics

data_path = os.path.join(os.path.dirname(__file__), 'data')

# Definizione del device e setup CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True  # Per ottimizzare le prestazioni

# Funzione per monitorare l'uso della memoria GPU
def print_gpu_memory():
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved()/1024**2:.2f} MB")

# Funzione per liberare la memoria GPU
def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA disponibile: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU in uso: {torch.cuda.get_device_name(0)}")

""" Iperparametri"""
# Parametri visualizzazione
stampa_evaluation= False
stampa_training= False

""" Dichiarazione funzioni -- le sposteremo in un file a parte """
def loaders(train_batch_size, test_batch_size, train_set_size, test_set_size):

    transform = Compose([
        ToTensor(),
        Normalize((0.1307,), (0.3081,)),
        Lambda(lambda x: torch.flatten(x))])
    
    # Carica il dataset MNIST
    train_dataset = MNIST(data_path, train=True, download=True, transform=transform)
    test_dataset = MNIST(data_path, train=False, download=True, transform=transform)

    # Crea sottoinsiemi dei dataset con le dimensioni desiderate
    train_subset, _ = random_split(train_dataset, [train_set_size, len(train_dataset) - train_set_size])
    test_subset, _ = random_split(test_dataset, [test_set_size, len(test_dataset) - test_set_size])

    # Crea i DataLoader per i sottoinsiemi dei dataset
    train_loader = DataLoader(train_subset, batch_size=train_batch_size, shuffle=True, drop_last=True, pin_memory=True)
    test_loader = DataLoader(test_subset, batch_size=test_batch_size, shuffle=False, drop_last=True, pin_memory=True)
    return train_loader, test_loader

def visualize_sample(data, name='', idx=0):
    reshaped = data[idx].cpu().reshape(28, 28)
    plt.figure(figsize = (4, 4))
    plt.title(name)
    plt.imshow(reshaped, cmap="gray")
    plt.show()

def create_mask(size, blur_iterations=5):
    """
    Crea una maschera binaria semplice per generare dati negativi
    """
    # Crea una maschera random
    mask = torch.rand(size, device=device)
    
    # Applica una semplice sfocatura media
    blur_kernel = torch.ones(3, 3, device=device) / 9.0
    blur_kernel = blur_kernel.view(1, 1, 3, 3)
    
    # Reshape per la convoluzione
    mask = mask.view(1, 1, size[0], size[1])
    
    # Applica sfocatura
    for _ in range(blur_iterations):
        mask = torch.nn.functional.conv2d(
            mask, 
            blur_kernel, 
            padding=1
        )
    
    # Applica threshold per ottenere una maschera binaria
    mask = mask.squeeze()
    mask = (mask > 0.5).float()
    
    return mask

def create_hybrid_image(image1, image2, mask):
    """
    Crea un'immagine ibrida combinando due immagini usando una maschera.
    """
    mask = mask.to(device)
    reverse_mask = 1 - mask
    hybrid_image = image1 * mask + image2 * reverse_mask
    return hybrid_image

def overlay_y_on_x(x, y):
    x = x.to(device)
    y = y.to(device)
    batch_size = x.shape[0]
    
    # Reshape x per applicare la maschera (da flat a 28x28)
    x_2d = x.view(batch_size, 28, 28)
    
    # Crea una maschera per ogni immagine nel batch
    masks = torch.stack([create_mask((28, 28)) for _ in range(batch_size)]).to(device)
    
    # Crea le immagini ibride
    x_hybrid = torch.zeros_like(x_2d)
    for i in range(batch_size):
        # Prendi un'immagine casuale dal batch come seconda immagine
        random_idx = torch.randint(0, batch_size, (1,)).item()
        x_hybrid[i] = create_hybrid_image(x_2d[i], x_2d[random_idx], masks[i])
    
    # Ritorna alla forma originale (flat)
    x_hybrid_flat = x_hybrid.view(batch_size, -1)
    
    # Aggiungi l'etichetta come prima
    x_out = x_hybrid_flat.clone()
    x_out[:, :10] *= 0.0
    x_out[range(x.shape[0]), y] = x.max()
    
    return x_out

class Layer(nn.Linear):
    def __init__(self, in_features, out_features,
                 bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.relu = activation_function 
        self.to(device)
            
    def forward(self, x):
        # Assicurati che x sia sullo stesso device del layer
        x = x.to(self.weight.device)
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        
        return self.relu(
            torch.mm(x_direction, self.weight.T) +
            self.bias.unsqueeze(0))

class MetaLayer(torch.nn.Module):
    def __init__(self, self_size, pre_size, post_size, out_size):
      super(MetaLayer, self).__init__()
      self.layer_pre  = Layer(pre_size, out_size).to(device)
      self.layer_post = Layer(post_size, out_size).to(device)
      self.layer_self = Layer(self_size, out_size).to(device)
      
      self.loss_history = []
      self.threshold = threshold
      self.num_epochs = epochs
      self.opt = torch.optim.Adam(self.parameters(), lr=learning_rate)
      
    def forward(self, x):
        in_self, in_pre, in_post = x  # unpacking della tupla
        # Assicurati che tutti gli input siano su GPU
        in_self = in_self.to(device)
        in_pre = in_pre.to(device) if in_pre is not None else None
        in_post = in_post.to(device) if in_post is not None else None
        return 0.7*(self.layer_pre(in_pre) + self.layer_post(in_post)) + 0.3*(self.layer_self(in_self))

    def train(self, x_pos, x_neg):
        self.loss_history.append(list())

        g_pos = self.forward(x_pos).pow(2).mean(1)
        g_neg = self.forward(x_neg).pow(2).mean(1)
            
        # The following loss pushes pos (neg) samples to
         # values larger (smaller) than the self.threshold.
            
        loss = torch.log(1 + torch.exp(torch.cat([
            -g_pos + self.threshold,
            g_neg - self.threshold]))).mean()
        self.opt.zero_grad()

        self.loss_history[-1].append(loss.item())

        # this backward just compute the derivative and hence
        # is not considered backpropagation.
        loss.backward()
        self.opt.step()
            
        if(stampa_training==True):
          print(f"questa è la Loss all'epoca {self.num_epochs}",loss.item())
        
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()


class Net(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = []
 
        #Metalayer (self_size - pre_size - post_size - out_size)
        self.layers += [MetaLayer(size_Metalayer_uno,size_input,size_Metalayer_due,size_Metalayer_uno).to(device)]      
        self.layers += [MetaLayer(size_Metalayer_due,size_Metalayer_uno,size_Metalayer_tre,size_Metalayer_due).to(device)]    
        self.layers += [MetaLayer(size_Metalayer_tre,size_Metalayer_due,0,size_Metalayer_tre).to(device)] 
        self.to(device)  # Assicurati che tutto il modello sia su GPU

    def predict(self, x, test_batch_size):
        # Assicurati che x sia su GPU
        x = x.to(device)
        goodness_per_label = []
        result_tensor = torch.ones(test_batch_size, 1).to(device)
        
        for label in range(10):
            if(stampa_evaluation==True):
                print("\n----LABEL NUMERO:", label,"-----")
            h = overlay_y_on_x(x, torch.full((x.size(0),), label).to(device))
            goodness = torch.zeros(test_batch_size).to(device)
            goodness_m1 = torch.zeros(test_batch_size,1).to(device)
            goodness_m2 = torch.zeros(test_batch_size,1).to(device)
            goodness_m3 = torch.zeros(test_batch_size,1).to(device) 
            goodness_intime = []
            self_evaluation_list_next = [torch.zeros(h.shape[0],size_Metalayer_uno).to(device), torch.zeros(h.shape[0],size_Metalayer_due).to(device), torch.zeros(h.shape[0],size_Metalayer_tre).to(device), torch.zeros(h.shape[0],0).to(device)]
            
            for time in range(test_time):
                if(stampa_evaluation==True):
                    print("\n----ISTANTE TEMPORALE:", time,"-----")

                self_evaluation_list = self_evaluation_list_next
            
                for i, layer in enumerate(self.layers):
              
                    pre_evaluation  = self_evaluation_list[i-1] if i>0 else h
                    self_evaluation = self_evaluation_list[i]
                    post_evaluation = self_evaluation_list[i+1]
                            

                    self_evaluation_list_next[i] = layer(     
                        (self_evaluation, pre_evaluation, post_evaluation)
                    )
                     
                    if time == 2 or time==3 or time==4: #234
                        if(stampa_evaluation==True):
                            print('.......................Evaluating Metalayer', i, '.....................')
                        
                        
                        if i== 0:
                            goodness_m1 += self_evaluation_list_next[i].detach().pow(2).mean(dim=1, keepdim=True)
                            
                        if i== 1:
                            goodness_m2 += self_evaluation_list_next[i].detach().pow(2).mean(dim=1, keepdim=True)
                            
                        if i== 2:
                            goodness_m3 += self_evaluation_list_next[i].detach().pow(2).mean(dim=1, keepdim=True)
                        
                    
                        goodness = goodness_m1 + goodness_m2 + goodness_m3
                        

                goodness_intime.append(goodness)
                        
            goodness_per_label.append(goodness_intime[0]+ goodness_intime[1] + goodness_intime[2])
        
        for i in range(test_batch_size):
            # Creare una lista di valori per l'i-esima posizione da tutti i tensori
            values = [tensor[i][0] for tensor in goodness_per_label]
            # Trovare l'indice del valore massimo
            max_index = max(range(len(values)), key=values.__getitem__)
            # Assegnare l'indice del valore massimo al tensore risultato
            result_tensor[i][0] = max_index
        
        return result_tensor #goodness_per_label

    def train(self, x_pos, x_neg):

        self_pos_list_next = [torch.zeros(x_pos.shape[0],size_Metalayer_uno).to(device), torch.zeros(x_pos.shape[0],size_Metalayer_due).to(device), torch.zeros(x_pos.shape[0],size_Metalayer_tre).to(device), torch.zeros(x_pos.shape[0],0).to(device)]
        self_neg_list_next = [torch.zeros(x_pos.shape[0],size_Metalayer_uno).to(device), torch.zeros(x_pos.shape[0],size_Metalayer_due).to(device), torch.zeros(x_pos.shape[0],size_Metalayer_tre).to(device), torch.zeros(x_pos.shape[0],0).to(device)]
        
        losses_per_layer = []  # Lista per memorizzare le loss di ogni layer

        for t in range(int(training_time)):
          if(stampa_training==True):
            print("\n----ISTANTE TEMPORALE:", t,"-----")
          self_pos_list = copy.deepcopy(self_pos_list_next)
          self_neg_list = copy.deepcopy(self_neg_list_next)
        
          layer_losses = []

          for i, layer in enumerate(self.layers):
              
              pre_pos  = self_pos_list[i-1] if i>0 else x_pos
              self_pos = self_pos_list[i]
              post_pos = self_pos_list[i+1]
                            
              pre_neg =  self_neg_list[i-1] if i>0 else x_neg
              self_neg = self_neg_list[i]
              post_neg = self_neg_list[i+1] 
              
              if(stampa_training==True):
                print('.......................Training Metalayer', i, '.....................')

              self_pos_list_next[i], self_neg_list_next[i] = layer.train(     
                  (self_pos, pre_pos, post_pos), 
                  (self_neg, pre_neg, post_neg)
              )

              # Assicurati che layer.loss_history[-1] esista e abbia almeno un elemento
              if layer.loss_history and layer.loss_history[-1]:
                  layer_losses.append(layer.loss_history[-1][-1])
              else:
                  layer_losses.append(0.0)  # o un altro valore predefinito appropriato
        
          losses_per_layer.append(layer_losses)
        
        return losses_per_layer[-1] if losses_per_layer else [0.0, 0.0, 0.0]  # Ritorna l'ultimo set di losses o valori predefiniti



trials_results = []


"""MAIN"""
params_df = pd.read_csv('hyperparameters.csv')

# Ciclo su tutte le righe
for index,row in params_df.iterrows():
    # Esempio di accesso ai valori della riga
    print(f"Test Numero:{row['trial_number']}")

    trial_number=row['trial_number'] 
    train_batch_size=row['train_batch_size']
    test_batch_size=row['test_batch_size']
    learning_rate=row['learning_rate']
    activation_function= eval(row['activation_function'])
    epochs=row['epochs']
    threshold=row['threshold'] 
    test_time=row['test_time']
    training_time=row['training_time']
    train_set_size=row['train_set_size']
    test_set_size=row['test_set_size']
    size_input=row['size_input']
    size_Metalayer_uno=row['size_metalayer_uno']
    size_Metalayer_due=row['size_metalayer_due']
    size_Metalayer_tre=row['size_metalayer_tre']

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

    # Larghezza delle colonne
    col_width = max(len(str(value)) for row in data_iperparametri for value in row) + 2  # padding
    # Separatore orizzontale
    separator = '-' * (col_width * 2 + 1)
    # Stampa della tabella
    print(separator)
    print(f"{'Parameter'.ljust(col_width)}| {'Value'.ljust(col_width)}")
    print(separator)
    for name, value in data_iperparametri:
        print(f"{name.ljust(col_width)}| {str(value).ljust(col_width)}")
    print(separator)

    """
    MAIN ORIGINALE
    """
    train_loader, test_loader = loaders(train_batch_size,test_batch_size, train_set_size, test_set_size)
    

    loss_layer_zero_per_epoch = []
    loss_layer_uno_per_epoch = []
    loss_layer_due_per_epoch = []
    if(stampa_training==True):
        print("Lunghezza train loader", len(train_loader))
        print("Lunghezza test loader", len(test_loader))

    net = Net()
    net = net.to(device)  # Sposta esplicitamente il modello su GPU
    for epoch in range(epochs):
        loss_layer_zero = []
        loss_layer_uno = []
        loss_layer_due = []
        print("Epoch", epoch)

        for x, y in tqdm(train_loader):
            x = x.to(device)
            y = y.to(device)
            
            x_pos = overlay_y_on_x(x, y)
            # Non abbiamo più bisogno di rnd e x_neg separati
            # La funzione overlay_y_on_x ora crea direttamente immagini ibride
            x_neg = x_pos  # L'ibridazione è già avvenuta nella funzione overlay_y_on_x
            
            losses_per_layer = net.train(x_pos, x_neg)
            loss_layer_zero.append(losses_per_layer[0])    
            loss_layer_uno.append(losses_per_layer[1])
            loss_layer_due.append(losses_per_layer[2])
        
        # Sposta questi append DOPO il ciclo di training, quando le liste contengono effettivamente i dati
        loss_layer_zero_per_epoch.append(statistics.mean(loss_layer_zero))
        loss_layer_uno_per_epoch.append(statistics.mean(loss_layer_uno))
        loss_layer_due_per_epoch.append(statistics.mean(loss_layer_due))
        
        print("loss_layer_zero", loss_layer_zero_per_epoch)
        print("loss_layer_uno", loss_layer_uno_per_epoch)
        print("loss_layer_due", loss_layer_due_per_epoch)

        

        # Aggiungi linee verticali per separare le epoche
        plt.axvline(x=epoch * len(train_loader), color='gray', linestyle='--', alpha=0.3)

    """ Creo una directory nuova per inserire le immagini per ogni test effettuato"""

    nome_directory = "test_" + str(trial_number)
    path_directory = os.path.join("testing_result", nome_directory)
    if not os.path.exists(path_directory):
        os.makedirs(path_directory)
    
    """ IMMAGINE LOSS PER LAYER """
    plt.figure(figsize=(12, 6))
    epochs_range = range(1, epochs + 1)

    # Plot delle loss per ogni layer
    plt.plot(epochs_range, loss_layer_zero_per_epoch, label='Layer 0', marker='o')
    plt.plot(epochs_range, loss_layer_uno_per_epoch, label='Layer 1', marker='s')
    plt.plot(epochs_range, loss_layer_due_per_epoch, label='Layer 2', marker='^')

    plt.xlabel('Epoche')
    plt.ylabel('Loss')
    plt.title('Andamento della Loss per Layer')
    plt.legend()
    plt.grid(True)

    # Aggiungi questo se vuoi salvare il grafico
    path_immagine = os.path.join(path_directory, "Loss_per_Layers.png")
    plt.savefig(path_immagine)
    plt.close()


    """
    SCRIPT DI TESTING SU TRAINING SET
    
    predicted_list_training=[]
    real_list_training=[]
    print("Numero di Batch nel test_loader:", len(train_loader))

    for x_te, y_te in tqdm(train_loader, desc="Testing su Training Set"):
        x_te = x_te.to(device)
        y_te = y_te.to(device)
     METRICHE
        training_result = net.predict(x_te, train_batch_size).int()
        training_result_trasposto=training_result.view(-1).tolist()
        labels= y_te.tolist()
        predicted_list_training.append(training_result_trasposto)
        real_list_training.append(labels)
    
    predicted_list_flattened_training = [item for sublist in predicted_list_training for item in sublist]
    real_list_flattened_training= [item for sublist in real_list_training for item in sublist]

    count = 0
    # Confronta le liste elemento per elemento
    for i in range(len(predicted_list_flattened_training)):
        if predicted_list_flattened_training[i] == real_list_flattened_training[i]:
            count += 1
    if(stampa_evaluation==True):
        print("Numero di predizioni", len(predicted_list_flattened_training))
        print("Numero di labels", len(real_list_flattened_training))
        print("Numero di valori uguali nelle stesse posizioni:", count)

    METRICHE 
    training_accuracy = round((count / len(predicted_list_flattened_training)) * 100,2)
    training_error = round(100- training_accuracy,2)
    training_precision = round(precision_score(real_list_flattened_training, predicted_list_flattened_training, average='macro',zero_division=1),2)
    training_recall = round(recall_score(real_list_flattened_training, predicted_list_flattened_training, average='macro'),2)
    training_f1 = round(f1_score(real_list_flattened_training, predicted_list_flattened_training, average='macro', zero_division=1),3)

    if(stampa_evaluation==True):
        print(f"\nTraining_Accuracy:{training_accuracy} %")
        print(f"Training Error: {training_error} %")
        print(f"Training_Precision: {training_precision}")
        print(f"Training_Recall:{training_recall}")
        print(f"Training F1 Score: {training_f1}")

    Matrice di confusione
    training_conf_matrix = confusion_matrix(predicted_list_flattened_training, real_list_flattened_training)

    class_names=["Zero","Uno","Due","Tre","Quattro","Cinque","Sei","Sette","Otto","Nove"]
    if(stampa_evaluation==True):
        plt.figure(figsize=(8, 4))
        sns.heatmap(training_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Etichette Predette')
        plt.ylabel('Etichette Reali')
        plt.title('Matrice di Confusione Training')
        #plt.show()

    # Ricreiamo la confusion matrix in trainig per salvarla
    plt.figure(figsize=(8, 6))
    sns.heatmap(training_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Etichette Predette')
    plt.ylabel('Etichette Reali')
    plt.title('Matrice di Confusione in Training ')
    path_immagine = os.path.join(path_directory, "Matrice di confusione in training.png")
    plt.savefig(path_immagine)

    # Report di classificazione completo
    report = classification_report(real_list_flattened_training, predicted_list_flattened_training, zero_division=1)
    if(stampa_evaluation==True):
        print("Classification Report:\n", report)"""

    """
    SCRIPT DI TESTING
    """
    predicted_list_testing=[]
    real_list_testing=[]
    print("Numero di Batch nel test_loader:", len(test_loader))

    for x_te, y_te in tqdm(test_loader, desc="Testing su Test Set"):
        x_te = x_te.to(device)
        y_te = y_te.to(device)
        """ CALCOLO METRICHE """
        testing_result = net.predict(x_te, test_batch_size).int()
        testing_result_trasposto=testing_result.view(-1).tolist()
        testing_labels= y_te.tolist()
        predicted_list_testing.append(testing_result_trasposto)
        real_list_testing.append(testing_labels)
    
    predicted_list_flattened_testing = [item for sublist in predicted_list_testing for item in sublist]
    real_list_flattened_testing= [item for sublist in real_list_testing for item in sublist]

    if(stampa_evaluation==True):
        print("Numero di predizioni", len(predicted_list_flattened_testing))
        print("Numero di labels", len(real_list_flattened_testing))

    count = 0
    # Confronta le liste elemento per elemento
    for i in range(len(predicted_list_flattened_testing)):
        if predicted_list_flattened_testing[i] == real_list_flattened_testing[i]:
            count += 1

    if(stampa_evaluation==True):
        print("Numero di valori uguali nelle stesse posizioni:", count)

    """ METRICHE """
    testing_accuracy =round((count / len(predicted_list_flattened_testing)) * 100,2)
    testing_error = round(100- testing_accuracy,2)
    testing_precision = round(precision_score(real_list_flattened_testing, predicted_list_flattened_testing, average='macro'),2)
    testing_recall = round(recall_score(real_list_flattened_testing, predicted_list_flattened_testing, average='macro'),2)
    testing_f1 = round(f1_score(real_list_flattened_testing, predicted_list_flattened_testing, average='macro', zero_division=1),3)
    #overfitting_degree = round(training_accuracy - testing_accuracy,3)

    if(stampa_evaluation==True):
        print(f"\nTesting Accuracy:{testing_accuracy}%")
        print(f"Testing Error: {testing_error}%")
        print(f"Testing Precisione:{testing_precision}")
        print(f"Testing Recall:{testing_recall}")
        print(f"Testing f1-score:{testing_f1}")
        #print(f"Grado di overfitting:{overfitting_degree}") # Se positivo abbiamo overfitting se vicino a zero è ok 
    # Lo stesso ragionamento lo possiamo fare con le Loss

    """Matrice di confusione"""
    testing_conf_matrix = confusion_matrix(predicted_list_flattened_testing, real_list_flattened_testing) 

    class_names=["Zero","Uno","Due","Tre","Quattro","Cinque","Sei","Sette","Otto","Nove"]

    if(stampa_evaluation==True):
        plt.figure(figsize=(8, 4))
        image = sns.heatmap(testing_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Etichette Predette')
        plt.ylabel('Etichette Reali')
        plt.title('Matrice di Confusione Testing')
        #plt.show()
        

    # Ricrea il grafico per salvarlo
    plt.figure(figsize=(8, 6))
    sns.heatmap(testing_conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Etichette Predette')
    plt.ylabel('Etichette Reali')
    plt.title('Matrice di Confusione Testing')
    path_immagine = os.path.join(path_directory, "Matrice di confusione in testing.png")
    plt.savefig(path_immagine)

    # Report di classificazione completo
    report = classification_report(real_list_flattened_testing, predicted_list_flattened_testing, zero_division=1)
    if(stampa_evaluation==True):
        print("Classification Report:\n", report)
    
    # Inizializzazione della lista dei risultati delle prove

    """Salvataggio risultati"""

    # Dizionario dei risultati
    trial_result = {
        'trial_number': trial_number,
        #'training_accuracy': training_accuracy,
        #'training_error': training_error,
        #'training_precision': training_precision,
        #'training_recall': training_recall,
        #'training_f1': training_f1,
        'testing_accuracy': testing_accuracy,
        'testing_error': testing_error,
        'testing_precision': testing_precision,
        'testing_recall': testing_recall,
        'testing_f1': testing_f1,
        'Loss layer zero': loss_layer_zero_per_epoch[-1],
        'Loss layer uno': loss_layer_uno_per_epoch[-1],
        'Loss layer due': loss_layer_due_per_epoch[-1],
        #'overfitting_degree': overfitting_degree
    }
    

    """ STAMPO A VIDEO LE METRICHE DEL TEST"""

    # Lista di parametri e valori
    data = list(trial_result.items())

    # Larghezza delle colonne
    col_width = max(len(str(value)) for row in data for value in row) + 2  # padding

    # Creazione della figura e degli assi
    fig, ax = plt.subplots(figsize=(8, 6))  # Crea una nuova figura per la tabella
    ax.axis('off')

    # Creazione della tabella
    table = ax.table(cellText=data, colLabels=['Parameter', 'Value'], loc='center', cellLoc='left')

    # Imposta lo stile della tabella
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)  # Regola le dimensioni della tabella

    # Salva l'immagine
    path_immagine = os.path.join(path_directory, "Tabella_parametri.png")
    plt.savefig(path_immagine)
    plt.close()  # Chiudi anche questa figura

    """ AGGIUNGO I RISULTATI ALLA LISTA DEI RISULTATI ED AL FILE CSV"""

    # Aggiungi i risultati alla lista delle prove
    trials_results.append(trial_result)

# Creazione del DataFrame dei risultati
df_results = pd.DataFrame(trials_results) 
# Salvataggio in un file CSV
df_results.to_csv('trials_results.csv', index=False)
print("File CSV con i risultati delle prove creato con successo.")

# Salvataggio in Excel
df_results.to_excel('Risultati_FF_ricorrente.xlsx', index=False)
print("File Excel con i risultati delle prove creato con successo.")

