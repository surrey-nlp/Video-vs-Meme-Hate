import copy
import torch
import torch.nn as nn
import pickle
from torch.utils import data
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.utils.multiclass import unique_labels
import wandb
from tqdm import tqdm
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

from datasets import load_dataset
dataset = load_dataset('limjiayi/hateful_memes_expanded')

FOLDER_NAME = '/backup/girish_datasets/HateMM/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Text_Model(nn.Module):
    def __init__(self, input_size, fc1_hidden, fc2_hidden, output_size, dropout_rate=0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.network=nn.Sequential(
            nn.Linear(input_size, fc1_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fc1_hidden, fc2_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fc2_hidden, fc2_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fc2_hidden, output_size),
        )
    def forward(self, xb):
        return self.dropout(self.network(xb))

class Image_Model(nn.Module):
    def __init__(self, input_size, fc1_hidden, fc2_hidden, output_size, dropout_rate=0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.network=nn.Sequential(
            nn.Linear(input_size, fc1_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fc1_hidden, fc2_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fc2_hidden, fc2_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fc2_hidden, output_size),
        )
    def forward(self, xb):
        return self.dropout(self.network(xb))

class Combined_model(nn.Module):
    def __init__(self, text_model, image_model, num_classes, dropout_rate=0.2):
        super().__init__()
        self.text_model = text_model
        self.image_model = image_model
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout_rate)

        # Additional hidden layers
        self.fc1 = nn.Linear(128, 256)  # Adjust the input size of fc1 to match the output size of the element-wise product
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 128)
        self.relu3 = nn.ReLU()
        self.fc_output = nn.Linear(128, 1)

    def forward(self, x_text, x_img):
        if x_text is not None:
            tex_out = self.text_model(x_text.to(x_text.device))
        else:
            tex_out = torch.zeros(x_img.size(0), 64).to(x_img.device) if x_img is not None else torch.zeros(1, 64)

        if x_img is not None:
            img_out = self.image_model(x_img.to(x_img.device))
        else:
            img_out = torch.zeros(x_text.size(0), 64).to(x_text.device) if x_text is not None else torch.zeros(1, 64)

        # Check the number of dimensions for tex_out and img_out
        if tex_out.ndim != img_out.ndim:
            # Unsqueeze or squeeze to make them compatible
            if tex_out.ndim < img_out.ndim:
                tex_out = tex_out.unsqueeze(dim=1)
            elif tex_out.ndim > img_out.ndim:
                img_out = img_out.unsqueeze(dim=1)
        
        inp = torch.cat((tex_out, img_out), dim=1)
        # inp = torch.cat((torch.zeros_like(tex_out), img_out), dim=1)
        # inp = tex_out * img_out # Element-wise multiplication
        inp = inp.view(inp.size(0), -1)  # Flatten the second dimension

        # Ensure that the input tensor shape matches the linear layer's weight tensor shape
        expected_input_size = 128
        if inp.size(1) != expected_input_size:
            # print(f"Warning: Input tensor shape ({inp.size()}) does not match the expected shape for self.fc1. Padding the input tensor.")
            padding_size = expected_input_size - inp.size(1)
            inp = torch.cat([inp, torch.zeros(inp.size(0), padding_size, device=inp.device)], dim=1)
        # #     inp = F.pad(inp, (0, padding_size), "constant", 0)
        # #     inp = inp.view(inp.size(0), 128)

        # self.fc_output = nn.Linear(inp.size(1), self.num_classes).to(inp.device)  # Create the linear layer on the same device as inp
        # inp = inp.to(device)  # Move inp tensor to the same device as the model

        # Pass through additional hidden layers
        inp = self.fc1(inp)
        inp = self.relu1(inp)
        inp = self.dropout(inp)
        inp = self.fc2(inp)
        inp = self.relu2(inp)
        inp = self.dropout(inp)
        inp = self.fc3(inp)
        inp = self.relu3(inp)
        inp = self.dropout(inp)

        logits = self.fc_output(inp).squeeze(1)
        return logits

class Dataset_ViT(data.Dataset):
    def __init__(self, dataset, split='train'):
        "Initialization"
        self.dataset = dataset
        self.split = split
    
    def load_data_for_image(self, image_id):
        # Load text and image data
        try:
            if self.split == 'train':
                text_data = torch.tensor(np.array(TextEmbedding_train[image_id]))
                # text_data = text_data.mean(dim=0)  # Average the embeddings for all tokens (CLIP)
                # text_data = text_data[0]
                image_data = torch.tensor(np.array(ImgEmbedding_train[self.modify_image_id(image_id)]))
            elif self.split == 'validation':
                text_data = torch.tensor(np.array(TextEmbedding_val[image_id]))
                # text_data = text_data.mean(dim=0)  # Average the embeddings for all tokens (CLIP)
                # text_data = text_data[0]
                image_data = torch.tensor(np.array(ImgEmbedding_val[self.modify_image_id(image_id)]))
            else:
                text_data = torch.tensor(np.array(TextEmbedding_test[image_id]))
                # text_data = text_data.mean(dim=0)  # Average the embeddings for all tokens (CLIP)
                # text_data = text_data[0]
                image_data = torch.tensor(np.array(ImgEmbedding_test[self.modify_image_id(image_id)]))
        except KeyError:
            print(f"KeyError: {image_id}")
            # Assign default values for missing data
            text_data = torch.zeros(512)    # 512 for CLIP, 768 for BERT and HXP
            image_data = torch.zeros(512)   # 768 for CLIP and VIT, 384 for DINOv2

        return text_data, image_data
    
    def modify_image_id(self, image_id):
        # Append '.png' if not already present
        if not image_id.endswith(('.png', '.jpg')):
            image_id += '.png'
        return image_id

    def __len__(self):
        return len(self.dataset[self.split])

    def __getitem__(self, index):
        "Generates one sample of data"
        image_id = self.dataset[self.split]['id'][index]
        # Load data
        X_text, X_img = self.load_data_for_image(image_id)
        # Load label
        y = self.dataset[self.split]['label'][index]

        return X_text, X_img, y


# with open(FOLDER_NAME + 'hatememes_ext_train_VITembedding.pkl', 'rb') as fp:
# with open(FOLDER_NAME + 'hatememes_ext_train_CLIPembedding.pkl', 'rb') as fp:
# with open(FOLDER_NAME + 'hatememes_ext_train_DINOv2embedding.pkl', 'rb') as fp:
with open(FOLDER_NAME + 'hatememes_ext_train_CLIP_proj_embedding.pkl', 'rb') as fp:
    ImgEmbedding_train = pickle.load(fp)

# with open(FOLDER_NAME + 'all_hatememes_ext_train_rawBERTembedding.pkl', 'rb') as fp:
# with open(FOLDER_NAME + 'all_hatememes_ext_train_hatexplain_embedding.pkl', 'rb') as fp:
# with open(FOLDER_NAME + 'all_hatememes_ext_train_clip_embedding.pkl', 'rb') as fp:
with open(FOLDER_NAME + 'all_hatememes_ext_train_clip_proj_embedding.pkl', 'rb') as fp:
    TextEmbedding_train = pickle.load(fp)

# with open(FOLDER_NAME + 'all_hatememes_ext_validation_rawBERTembedding.pkl', 'rb') as fp:
# with open(FOLDER_NAME + 'all_hatememes_ext_validation_hatexplain_embedding.pkl', 'rb') as fp:
# with open(FOLDER_NAME + 'all_hatememes_ext_validation_clip_embedding.pkl', 'rb') as fp:
with open(FOLDER_NAME + 'all_hatememes_ext_validation_clip_proj_embedding.pkl', 'rb') as fp:
    TextEmbedding_val = pickle.load(fp)

# with open(FOLDER_NAME + 'hatememes_ext_validation_VITembedding.pkl', 'rb') as fp:
# with open(FOLDER_NAME + 'hatememes_ext_validation_CLIPembedding.pkl', 'rb') as fp:
# with open(FOLDER_NAME + 'hatememes_ext_validation_DINOv2embedding.pkl', 'rb') as fp:
with open(FOLDER_NAME + 'hatememes_ext_validation_CLIP_proj_embedding.pkl', 'rb') as fp:
    ImgEmbedding_val = pickle.load(fp)

# with open(FOLDER_NAME + 'all_hatememes_ext_test_rawBERTembedding.pkl', 'rb') as fp:
# with open(FOLDER_NAME + 'all_hatememes_ext_test_hatexplain_embedding.pkl', 'rb') as fp:
# with open(FOLDER_NAME + 'all_hatememes_ext_test_clip_embedding.pkl', 'rb') as fp:
with open(FOLDER_NAME + 'all_hatememes_ext_test_clip_proj_embedding.pkl', 'rb') as fp:
    TextEmbedding_test = pickle.load(fp)

# with open(FOLDER_NAME + 'hatememes_ext_test_VITembedding.pkl', 'rb') as fp:
# with open(FOLDER_NAME + 'hatememes_ext_test_CLIPembedding.pkl', 'rb') as fp:
# with open(FOLDER_NAME + 'hatememes_ext_test_DINOv2embedding.pkl', 'rb') as fp:
with open(FOLDER_NAME + 'hatememes_ext_test_CLIP_proj_embedding.pkl', 'rb') as fp:
    ImgEmbedding_test = pickle.load(fp)


def eval_metrics(y_true, y_pred):
    try:
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, labels = np.unique(y_pred), average='macro', zero_division='warn')
        precision = precision_score(y_true, y_pred, labels = np.unique(y_pred), average='macro', zero_division='warn')
        recall = recall_score(y_true, y_pred, labels = np.unique(y_pred), average='macro', zero_division='warn')
        
        # Check if there is only one class present in y_true
        num_classes = len(unique_labels(y_true))
        if num_classes == 1:
            roc_auc = 0.5  # Return a default value for binary classification
            print("Warning: ROC AUC score is not defined for a single class. Returning default value of 0.5.")
        else:
            roc_auc = roc_auc_score(y_true, y_pred, average='macro')
    except Exception as e:
        print(f"Error in eval_metrics: {e}")
        return 0, 0, 0, 0, 0
    
    return accuracy, f1, precision, recall, roc_auc


def collate_fn(batch):
    text, image, label = zip(*[(t, i, l) for t, i, l in batch if torch.any(t != 0) and torch.any(i != 0)])
    # text, image, label = zip(*batch)
    # Make sure all text tensors have the same shape
    text = [t.squeeze(0) if t.ndim > 1 else t for t in text]
    text = torch.stack(text)

    # Make sure all image tensors have the same shape
    image = [img.unsqueeze(0) if img.ndim == 1 else img for img in image]
    image = torch.stack(image)
    label = torch.tensor(label)
    return text, image, label

def label_smoothing_loss(inputs, targets, epsilon=0.1):
    """Applies label smoothing to the cross-entropy loss"""
    num_classes = inputs.size(-1)
    log_probs = F.log_softmax(inputs, dim=-1)
    # targets = targets.to(dtype=torch.float32)
    # targets = (1.0 - epsilon) * targets + epsilon / inputs.size(-1)
    targets = torch.zeros_like(inputs).scatter_(1, targets.unsqueeze(1), 1)
    targets = (1 - epsilon) * targets + (epsilon / num_classes)
    loss = (-targets * log_probs).sum(-1).mean()
    return loss

def l1_regularized_loss(outputs, labels, model, l1_lambda=0.001):
    """Compute the cross-entropy loss with L1 regularization"""
    loss = F.cross_entropy(outputs, labels)

    # Compute the L1 regularization term
    l1_reg = 0
    for param in model.parameters():
        l1_reg += torch.sum(torch.abs(param))

    # Add the L1 regularization term to the loss
    loss += l1_lambda * l1_reg

    return loss

input_text_size = 512   # 512 for CLIP, 768 for BERT and HXP
input_image_size = 512  # 768 for CLIP and VIT, 384 for DINOv2
fc1_hidden = 128
fc2_hidden = 128

# training parameters
num_classes = 2
initial_lr = 1e-4
num_epochs = 20
batch_size = 64


wandb.init(
    project="hate-memes-classification",
    config={
        "learning_rate": initial_lr,
        "architecture": "CLIP Text + CLIP Image (Concat - Projection Layer)",
        "dataset": "Hateful Memes",
        "epochs": num_epochs,
        "batch_size": batch_size,
    },
)

ext_data = {}

# DataLoaders
for split in dataset.keys():
    # consider only the first 8.5k samples for training, 500 for validation, and 1k for testing (hateful memes dataset)
    if split == 'train':
        dataset[split] = dataset[split].select(list(range(8500)))
    elif split == 'validation':
        dataset[split] = dataset[split].select(list(range(500)))
    elif split == 'test':
        dataset[split] = dataset[split].select(list(range(1000)))
        
    # get label statistics
    # label_stats = np.array(dataset[split]['label'])
    # print(f"Label statistics for {split}: {np.unique(label_stats, return_counts=True)}")

    ext_data[split] = Dataset_ViT(dataset, split)

train_loader = data.DataLoader(ext_data['train'], batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = data.DataLoader(ext_data['validation'], batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = data.DataLoader(ext_data['test'], batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Model
text_model = Text_Model(input_text_size, fc1_hidden, fc2_hidden, num_classes).to(device)
image_model = Image_Model(input_image_size, fc1_hidden, fc2_hidden, num_classes).to(device)

model = Combined_model(text_model, image_model, num_classes).to(device)

# Parallelize model to multiple GPUs
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
weight_decay = 1e-4
# l1_lambda = 0.001  # L1 regularization strength, make weight_decay = 0
optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr, weight_decay=weight_decay)

# Train the model
def train_model(model, train_loader, val_loader, num_epochs, criterion, optimizer, initial_lr, patience=5):
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=patience // 2, factor=0.1, verbose=True)
    best_val_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_loss = 0
        for text, image, labels in train_loader:
            text = text.to(device)
            image = image.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(text, image)
            loss = criterion(logits, labels.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        val_y_true = []
        val_y_pred = []
        with torch.no_grad():
            for text, image, labels in val_loader:
                text = text.to(device)
                image = image.to(device)
                labels = labels.to(device)

                logits = model(text, image)
                loss = criterion(logits, labels.float())
                val_loss += loss.item()

                val_y_true += labels.cpu().numpy().tolist()
                val_y_pred += torch.sigmoid(logits).cpu().numpy().tolist()

        val_y_true = np.array(val_y_true)
        val_y_pred = np.array(val_y_pred)
        val_y_pred = np.where(val_y_pred >= 0.6, 1, 0)

        accuracy, f1, precision, recall, roc_auc = eval_metrics(val_y_true, val_y_pred)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')
        print(f'Val Accuracy: {accuracy:.4f}, Val F1: {f1:.4f}, Val Precision: {precision:.4f}, Val Recall: {recall:.4f}, Val ROC AUC: {roc_auc:.4f}')

        wandb.log({"Train Loss": train_loss/len(train_loader), "Validation Loss": val_loss/len(val_loader), "Validation Accuracy": accuracy, "Val F1": f1, 
                    "Validation Precision": precision, "Validation Recall": recall, "Validation ROC AUC": roc_auc})

        lr_scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

train_model(model, train_loader, val_loader, num_epochs, criterion, optimizer, initial_lr)

# Test the model
def test_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for text, image, labels in test_loader:
            text = text.to(device)
            image = image.to(device)
            labels = labels.to(device)

            logits = model(text, image)
            loss = criterion(logits, labels.float())
            test_loss += loss.item()

            y_true += labels.cpu().numpy().tolist()
            y_pred += torch.sigmoid(logits).cpu().numpy().tolist()

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred = np.where(y_pred >= 0.6, 1, 0)

    accuracy, f1, precision, recall, roc_auc = eval_metrics(y_true, y_pred)
    print(f'Test Loss: {test_loss/len(test_loader):.4f}')
    print(f'Test Accuracy: {accuracy:.4f}, Test F1: {f1:.4f}, Test Precision: {precision:.4f}, Test Recall: {recall:.4f}, Test ROC AUC: {roc_auc:.4f}')

    wandb.log({"Test Loss": test_loss/len(test_loader), "Test Accuracy": accuracy, "Test F1": f1,
                "Test Precision": precision, "Test Recall": recall, "Test ROC AUC": roc_auc})

test_model(model, test_loader, criterion)
