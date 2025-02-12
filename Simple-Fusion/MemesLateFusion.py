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

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

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
        self.output_size = output_size

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
        self.output_size = output_size

    def forward(self, xb):
        return self.dropout(self.network(xb))

class RulesFromProbabilities(nn.Module):
    def __init__(self, visual_model, textual_model, num_classes, dropout_rate=0.2):
        super(RulesFromProbabilities, self).__init__()
        self.visual_model = visual_model
        self.textual_model = textual_model
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout_rate)
        self.visual_classifier = nn.Linear(visual_model.output_size, num_classes)
        self.textual_classifier = nn.Linear(textual_model.output_size, num_classes)
        self.final_classifier = nn.Linear(num_classes * 2, num_classes)

    def forward(self, x_text, x_img):
        visual_output = self.visual_model(x_img)
        textual_output = self.textual_model(x_text)

        visual_probabilities = self.visual_classifier(visual_output)
        textual_probabilities = self.textual_classifier(textual_output)

        visual_probabilities = visual_probabilities.squeeze(1)

        combined_probabilities = torch.cat((visual_probabilities, textual_probabilities), dim=1)
        combined_probabilities = self.dropout(combined_probabilities)

        output = self.final_classifier(combined_probabilities)

        return output

class WeightingTechnique(nn.Module):
    def __init__(self, visual_model, textual_model, num_classes):
        super(WeightingTechnique, self).__init__()
        self.visual_model = visual_model
        self.textual_model = textual_model
        self.num_classes = num_classes
        self.weight_visual = nn.Parameter(torch.randn(1, requires_grad=True))
        self.weight_textual = nn.Parameter(torch.randn(1, requires_grad=True))

        # print(f"visual_model output_size: {visual_model.output_size}, textual_model output_size: {textual_model.output_size}")
        self.classifier = nn.Linear(visual_model.output_size + textual_model.output_size, num_classes)

    def forward(self, x_text, x_img):
        visual_output = self.visual_model(x_img)
        textual_output = self.textual_model(x_text)

        weighted_visual = self.weight_visual * visual_output
        weighted_textual = self.weight_textual * textual_output

        weighted_visual = weighted_visual.squeeze(1)

        # Ensure that the same dimensions are being concatenated
        # if weighted_textual.ndim != weighted_visual.ndim:
        #     # Unsqueeze or squeeze to make them compatible
        #     if weighted_textual.ndim < weighted_visual.ndim:
        #         weighted_textual = weighted_textual.unsqueeze(dim=1)
        #     elif weighted_textual.ndim > weighted_visual.ndim:
        #         weighted_visual = weighted_visual.unsqueeze(dim=1)

        # combined_output = weighted_visual + weighted_textual
        combined_output = torch.cat((weighted_visual, weighted_textual), dim=1)
        output = self.classifier(combined_output)

        return output

class Combined_model(nn.Module):
    def __init__(self, text_model, image_model, num_classes, dropout_rate=0.2):
        super().__init__()
        self.text_model = text_model
        self.image_model = image_model
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout_rate)

        # self.weighting_technique = WeightingTechnique(self.image_model, self.text_model, num_classes)
        self.rules_from_probabilities = RulesFromProbabilities(self.image_model, self.text_model, num_classes, dropout_rate)

    def forward(self, x_text, x_img):
        # outputs = self.weighting_technique(x_text, x_img)
        outputs = self.rules_from_probabilities(x_text, x_img)
        return outputs

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
                image_data = torch.tensor(np.array(ImgEmbedding_train[self.modify_image_id(image_id)]))
            elif self.split == 'validation':
                text_data = torch.tensor(np.array(TextEmbedding_val[image_id]))
                image_data = torch.tensor(np.array(ImgEmbedding_val[self.modify_image_id(image_id)]))
            else:
                text_data = torch.tensor(np.array(TextEmbedding_test[image_id]))
                image_data = torch.tensor(np.array(ImgEmbedding_test[self.modify_image_id(image_id)]))
        except KeyError:
            print(f"KeyError: {image_id}")
            # Assign default values for missing data
            text_data = torch.zeros(768)
            image_data = torch.zeros(768)

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


with open(FOLDER_NAME + 'hatememesext_train_VITembedding.pkl', 'rb') as fp:
    ImgEmbedding_train = pickle.load(fp)

with open(FOLDER_NAME + 'all_hatememesext_train_rawBERTembedding.pkl', 'rb') as fp:
# with open(FOLDER_NAME + 'all_hatememesext_train_hatexplain_embedding.pkl', 'rb') as fp:
    TextEmbedding_train = pickle.load(fp)

with open(FOLDER_NAME + 'all_hatememesext_validation_rawBERTembedding.pkl', 'rb') as fp:
# with open(FOLDER_NAME + 'all_hatememesext_validation_hatexplain_embedding.pkl', 'rb') as fp:
    TextEmbedding_val = pickle.load(fp)

with open(FOLDER_NAME + 'hatememesext_validation_VITembedding.pkl', 'rb') as fp:
    ImgEmbedding_val = pickle.load(fp)

with open(FOLDER_NAME + 'all_hatememesext_test_rawBERTembedding.pkl', 'rb') as fp:
# with open(FOLDER_NAME + 'all_hatememesext_test_hatexplain_embedding.pkl', 'rb') as fp:
    TextEmbedding_test = pickle.load(fp)

with open(FOLDER_NAME + 'hatememesext_test_VITembedding.pkl', 'rb') as fp:
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


def handle_imbalance(dataset, oversampling=True, undersampling=True):
    smote = SMOTE()
    undersampler = RandomUnderSampler()
    X_text = []
    X_img = []
    y = []

    for i in range(len(dataset)):
        X_text_sample, X_img_sample = dataset[i][0], dataset[i][1]
        y_sample = dataset[i][2]

        X_text.append(X_text_sample)
        X_img.append(X_img_sample)
        y.append(y_sample)

    if oversampling:
        X_text_res, y_res = smote.fit_resample(X_text, y)
        # print(f"X_text_res: {len(X_text_res)}, y_res: {len(y_res)}")
        X_img_array = np.stack(X_img)
        X_img_flattened = X_img_array.reshape(X_img_array.shape[0], -1)
        X_img_res, _ = smote.fit_resample(X_img_flattened, y)
        # X_text_res = [torch.from_numpy(x) for x in X_text_res]
        # X_img_res = [torch.from_numpy(x) for x in X_img_res]
    else:
        X_text_res, X_img_res, y_res = X_text, X_img, y

    if undersampling:
        X_text_res, X_img_res, y_res = undersampler.fit_resample(X_text_res, X_img_res, y_res)
        # X_text_res = [torch.from_numpy(x) for x in X_text_res]
        # X_img_res = [torch.from_numpy(x) for x in X_img_res]

    return X_text_res, X_img_res, y_res


def collate_fn(batch):
    text, image, label = zip(*[(t, i, l) for t, i, l in batch if torch.any(t != 0) and torch.any(i != 0)])
    # text, image, label = zip(*batch)
    # Convert to tensor if it's not already one
    text = [torch.tensor(t) if not isinstance(t, torch.Tensor) else t for t in text]
    image = [torch.tensor(img) if not isinstance(img, torch.Tensor) else img for img in image]
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


input_text_size = 768
input_image_size = 768
fc1_hidden = 128
fc2_hidden = 128

# training parameters
num_classes = 2
initial_lr = 1e-4
num_epochs = 30
batch_size = 32


wandb.init(
    project="hate-memes-classification",
    config={
        "learning_rate": initial_lr,
        "architecture": "BERT + ViT (Trained Probs - LF)",
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
    label_stats = np.array(dataset[split]['label'])
    print(f"Label statistics for {split}: {np.unique(label_stats, return_counts=True)}")

    ext_data[split] = Dataset_ViT(dataset, split)

# Apply oversampling and undersampling
# X_text_train, X_img_train, y_train = handle_imbalance(ext_data['train'], oversampling=True, undersampling=False)
# train_loader = data.DataLoader(list(zip(X_text_train, X_img_train, y_train)), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# X_text_val, X_img_val, y_val = handle_imbalance(ext_data['validation'], oversampling=False, undersampling=False)
# val_loader = data.DataLoader(list(zip(X_text_val, X_img_val, y_val)), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# X_text_test, X_img_test, y_test = handle_imbalance(ext_data['test'], oversampling=False, undersampling=False)
# test_loader = data.DataLoader(list(zip(X_text_test, X_img_test, y_test)), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

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
criterion = nn.CrossEntropyLoss()
weight_decay = 1e-4
# l1_lambda = 0.001  # L1 regularization strength, make weight_decay = 0
optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr, weight_decay=weight_decay)

# Train the model
def train_model(model, train_loader, val_loader, num_epochs, criterion, optimizer, initial_lr, patience=5):
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=patience // 2, factor=0.1, verbose=True)
    best_val_loss = float('inf')
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        # Use tqdm for tracking progress
        progress_bar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}")

        for i, (text, image, labels) in enumerate(progress_bar):
            if text is None or image is None:
                # Skip samples with missing data
                continue
            train_y_true = []
            train_y_pred = []
            text = text.to(device)
            image = image.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(text, image)
            # loss = l1_regularized_loss(outputs, labels, model, l1_lambda)
            # loss = label_smoothing_loss(outputs, labels)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            train_y_true.extend(labels.cpu().numpy())
            train_y_pred.extend(predicted.cpu().numpy())

            # Update the progress bar description
            progress_bar.set_postfix(loss=train_loss / (i + 1))

        train_loss /= len(train_loader)

        val_loss = 0
        model.eval()
        with torch.no_grad():
            val_y_true = []
            val_y_pred = []
            for text, image, labels in val_loader:
                text = text.to(device)
                image = image.to(device)
                labels = labels.to(device)

                outputs = model(text, image)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                val_y_true.extend(labels.cpu().numpy())
                val_y_pred.extend(predicted.cpu().numpy())

        val_loss /= len(val_loader)
        lr_scheduler.step(val_loss)  # Update learning rate based on validation loss

        # Log the weights after each epoch
        # if isinstance(model, nn.DataParallel):
        #     # If the model is wrapped in DataParallel, the original model is accessed with .module
        #     weight_visual = model.module.weighting_technique.weight_visual.item()
        #     weight_textual = model.module.weighting_technique.weight_textual.item()
        # else:
        #     weight_visual = model.weighting_technique.weight_visual.item()
        #     weight_textual = model.weighting_technique.weight_textual.item()

        wandb.log({"Train Loss": train_loss, "Validation Loss": val_loss, 
                   "Train Accuracy": eval_metrics(train_y_true, train_y_pred)[0], 
                   "Validation Accuracy": eval_metrics(val_y_true, val_y_pred)[0],
                #    "Weight Visual": weight_visual, "Weight Textual": weight_textual,
                   "Train ROC AUC": eval_metrics(train_y_true, train_y_pred)[4], "Validation ROC AUC": eval_metrics(val_y_true, val_y_pred)[4]})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # torch.save(model.state_dict(), 'best_model.pth')
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping after {epoch + 1} epochs.")
                break

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Train Accuracy: {eval_metrics(train_y_true, train_y_pred)[0]:.4f}, Validation Accuracy: {eval_metrics(val_y_true, val_y_pred)[0]:.4f}')

    if best_model_state is not None:
        torch.save(best_model_state, 'best_model.pth')

train_model(model, train_loader, val_loader, num_epochs, criterion, optimizer, initial_lr)

# Test the model
def test_model(model, test_loader, criterion):
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    test_loss = 0
    with torch.no_grad():
        y_true = []
        y_pred = []
        for text, image, labels in test_loader:
            text = text.to(device)
            image = image.to(device)
            labels = labels.to(device)

            outputs = model(text, image)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

        accuracy, f1, precision, recall, roc_auc = eval_metrics(y_true, y_pred)
        wandb.log({"Test Accuracy": accuracy, "Test F1": f1, "Test Precision": precision, "Test Recall": recall, "Test ROC AUC": roc_auc})

        print(f'Test Accuracy: {accuracy:.4f}, Test F1: {f1:.4f}, Test Precision: {precision:.4f}, Test Recall: {recall:.4f}, Test ROC AUC: {roc_auc:.4f}')

test_model(model, test_loader, criterion)
