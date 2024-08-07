import pickle
import torch
import numpy as np
from torch import nn
from torch.utils import data
from transformers import BartTokenizerFast
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

from multimodal_bart_downstream import MultimodalBartForSequenceClassification
from transformers import BartModel
from audio_video_first import MultimodalAudio

FOLDER_NAME = '/backup/girish_datasets/HateMM/'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = 1e-4

SOURCE_MAX_LEN = 768 # 500
ACOUSTIC_DIM = 768
ACOUSTIC_MAX_LEN = 1000
VISUAL_DIM = 384 # 2048
VISUAL_MAX_LEN = 100 # 480


# with open(FOLDER_NAME + 'all__video_vosk_audioMap.pkl', 'rb') as f:
with open(FOLDER_NAME+'all_whisper_tiny_transcripts.pkl','rb') as f:
    transcript = pickle.load(f)

with open(FOLDER_NAME + 'Wav2Vec2_features_chunked.pkl', 'rb') as fo:
# with open(FOLDER_NAME+'CLAP_features.pkl','rb') as fo:
    audio_data = pickle.load(fo)

with open(FOLDER_NAME + 'noFoldDetails.pkl', 'rb') as fp:
    video_labels = pickle.load(fp)

def pad_seq(tensor, dim, max_len):
    if max_len > tensor.shape[0] :
        return torch.cat([tensor, torch.zeros(max_len - tensor.shape[0], dim)])
    else:
        return tensor[:max_len]
  

model = MultimodalBartForSequenceClassification.from_pretrained("facebook/bart-base")       # for text first models

tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-base')
# print("Tokenizer : ", tokenizer)

# num_param = sum(p.numel() for p in model.parameters())
# print("Total parameters : ", num_param/1e6)


p = {
        'additional_special_tokens' : ['[CONTEXT]', '[UTTERANCE]']
        # 'additional_special_tokens' : ['[UTTERANCE]']
    }

tokenizer.add_special_tokens(p)
# print("Tokenizer after adding special tokens : ", tokenizer)

# print(model.resize_token_embeddings(len(tokenizer)))

# model = MultimodalAudio()     # for audio/video first models


class HateMMDataset(data.Dataset):
    "Characterizes a dataset for PyTorch"
    def __init__(self, folders, labels):
        "Initialization"
        # print(folders, labels)
        self.labels = labels
        self.folders = folders

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.folders)

    def load_data_for_video(self, video):
        video_file_name_without_extension, _ = os.path.splitext(video)
        # pickle_file_path = os.path.join(FOLDER_NAME, "VITF_new", video_file_name_without_extension + "_vit.p")
        pickle_file_path = os.path.join(FOLDER_NAME, "DINOv2_lhs", video_file_name_without_extension + "_DINOv2_features.pkl")
        # pickle_file_path = os.path.join(FOLDER_NAME, "CLIP_pooled", video_file_name_without_extension + "_clip.pkl")
        
        # Load text data
        if video in transcript:
            # text_features = torch.tensor(np.array(transcript[video]), dtype=torch.float32)
            text_features = tokenizer(transcript[video], max_length = SOURCE_MAX_LEN, padding = 'max_length', truncation = True)
        else:
            # text_features = torch.zeros(768, dtype=torch.float32)
            raise ValueError(f"Text data not found for {video}")
        
        # Load video data
        try:
            with open(pickle_file_path, 'rb') as fp:
                video_data = pickle.load(fp)
                video_features = torch.tensor(np.array(list(video_data.values())), dtype=torch.float32)     # for last hidden state features
                # video_features = torch.stack([tensor.detach() for tensor in video_data.values()])     # for pooled features
                # video_features = torch.tensor(np.array(list(video_data.values().mean(dim=0))), dtype=torch.float32)
                video_features = video_features.mean(dim=1)
        except FileNotFoundError:
            raise ValueError(f"Video data file not found: {pickle_file_path}")
        
        # Load audio data
        if video in audio_data:
            audio_features = torch.tensor(np.array(audio_data[video]), dtype=torch.float32)
            audio_features = audio_features.mean(dim=0) # for wav2vec2
            # audio_features, _ = audio_features.max(dim=0)
            # audio_features = audio_features.view(audio_features.size(0), -1) # for CLAP
        else:
            # audio_features = torch.zeros(768, dtype=torch.float32)
            raise ValueError(f"Audio data not found for {video}")
        
        return text_features, video_features, audio_features

    def __getitem__(self, index):
        "Generates one sample of data"
        try:
            # Select sample
            folder = self.folders[index]
            # Load data
            X_text, X_vid, X_audio = self.load_data_for_video(folder)
            y = torch.LongTensor([self.labels[index]]) 
            
            # return X_text, X_vid, X_audio, y
            return torch.tensor(X_text['input_ids'], dtype=torch.long), torch.tensor(X_text['attention_mask'], dtype=torch.bool), X_audio, X_vid, y
        
        except Exception as e:
            # traceback.print_exc()
            print(f"Error loading data for index {index}: {e}")                        
            return None
        

all_train_data, all_train_label = video_labels['train']
all_val_data, all_val_label = video_labels['val']
all_test_data, all_test_label = video_labels['test']

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:  # Check if the batch is empty after filtering
        return None

    return torch.utils.data.dataloader.default_collate(batch)

# training parameters
k = 2            # number of target category
epochs = 20
batch_size = 32
log_interval = 100

# import wandb
# wandb.init(
#     project="hate-video-classification",
#     config={
#         "learning_rate": LEARNING_RATE,
#         "architecture": "Audio -> Video -> Text",
#         "dataset": "HateMM",
#         # "features": "BART + ViT + Wav2Vec2",
#         "epochs": epochs,
#         "batch_size": batch_size,
#     },
# )

params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True} if torch.cuda.is_available() else {}
valParams = {'batch_size': batch_size, 'shuffle': False, 'num_workers': 2, 'pin_memory': True} if torch.cuda.is_available() else {}

train_set, val_set, test_set = HateMMDataset(all_train_data, all_train_label), HateMMDataset(all_val_data, all_val_label), HateMMDataset(all_test_data, all_test_label)
train_loader = data.DataLoader(train_set, collate_fn = collate_fn, **params)
test_loader = data.DataLoader(test_set, collate_fn = collate_fn, **valParams)
valid_loader = data.DataLoader(val_set, collate_fn = collate_fn, **valParams)

model.to(DEVICE)

# Parallelize model to multiple GPUs
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
criterion = torch.nn.CrossEntropyLoss()


def train_epoch(model, data_loader):
    model.train()
    epoch_train_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    all_preds = []
    all_labels = []

    for step, batch in enumerate(tqdm(data_loader, desc='Training Iteration')):
        input_ids, attention_mask, acoustic_input, visual_input, labels = batch

        input_ids, attention_mask, acoustic_input, visual_input, labels = input_ids.to(DEVICE), attention_mask.to(DEVICE), acoustic_input.to(DEVICE), visual_input.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        acoustic_input=acoustic_input,
                        visual_input=visual_input,
                        labels=labels)

        loss = outputs['loss']
        loss = loss.mean()
        epoch_train_loss += loss.item()

        logits = outputs['logits']
        preds = torch.argmax(logits, dim=-1)
        correct_predictions += (preds == labels).sum().item()
        total_predictions += labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        loss.backward()
        optimizer.step()

        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')

        # wandb.log({"loss": epoch_train_loss, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1})
    # print("Epoch train loss : ", epoch_train_loss, "Accuracy: ", accuracy, "Precision: ", precision, "Recall: ", recall, "F1 Score: ", f1)


def valid_epoch(model, data_loader):
    model.eval()
    valid_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for step, batch in enumerate(tqdm(data_loader, desc='Validation Iteration')):
            input_ids, attention_mask, acoustic_input, visual_input, labels = batch
            input_ids, attention_mask, acoustic_input, visual_input, labels = input_ids.to(DEVICE), attention_mask.to(DEVICE), acoustic_input.to(DEVICE), visual_input.to(DEVICE), labels.to(DEVICE)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            acoustic_input=acoustic_input,
                            visual_input=visual_input,
                            labels=labels)

            logits = outputs['logits']
            loss = outputs['loss']
            loss = loss.mean()

            valid_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            correct_predictions += (preds == labels).sum().item()
            total_predictions += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            accuracy = accuracy_score(all_labels, all_preds)
            precision = precision_score(all_labels, all_preds, average='macro')
            recall = recall_score(all_labels, all_preds, average='macro')
            f1 = f1_score(all_labels, all_preds, average='macro')

            # wandb.log({"Valid_loss": valid_loss, "Valid_accuracy": accuracy, "Valid_precision": precision, "Valid_recall": recall, "Valid_f1": f1})

    return valid_loss, all_preds, all_labels


def test_epoch(model, data_loader):
    model.eval()
    predictions = []
    gold = []

    correct = 0
    with torch.no_grad():
        for step, batch in enumerate(tqdm(data_loader)):
            # batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, attention_mask, acoustic_input, visual_input, labels = batch
            input_ids, attention_mask, acoustic_input, visual_input, labels = input_ids.to(DEVICE), attention_mask.to(DEVICE), acoustic_input.to(DEVICE), visual_input.to(DEVICE), labels.to(DEVICE)

            outputs = model(input_ids = input_ids,
                            attention_mask = attention_mask,
                            # context_input_ids = context_input_ids,
                            # context_attention_mask = context_attention_mask,

                            acoustic_input = acoustic_input,
                            visual_input = visual_input,
                            labels = labels)

            logits = outputs['logits']
            pred = logits.argmax(dim = -1)

            predictions.extend(pred.tolist())
            gold.extend(labels.tolist())

            # correct += int((pred == labels).sum())

    return predictions, gold


class EarlyStopping:
  def __init__(self, patience, min_delta):
    self.patience = patience
    self.min_delta = min_delta
    self.counter = 0
    self.min_validation = np.inf

  def early_stop(self, valid_loss):
    if valid_loss < self.min_validation:
      self.min_validation = valid_loss
      self.counter = 0
    elif valid_loss > (self.min_validation + self.min_delta):
      self.counter += 1
      if self.counter >= self.patience:
        return True
    return False
  
early_stopper = EarlyStopping(patience = 15, min_delta = 0.2)


def train_and_validation(model, train_loader, valid_loader):
    # lowest_loss = 1e6
    best_f1 = 0.0
    # min_loss = 1e6
    for epoch in range(epochs):
      print("\n=============Epoch : ", epoch)
      train_epoch(model, train_loader)
      valid_loss, valid_pred, valid_gold = valid_epoch(model, valid_loader)

      if early_stopper.early_stop(valid_loss):
        break

      print("Length of predictions : ", len(valid_pred))
      print("Length of gold : ", len(valid_gold))
      print("Valid loss : ", valid_loss)
      print("\n Valid Accuracy : ", accuracy_score(valid_gold, valid_pred))
      print("\n Valid Precision : ", precision_score(valid_gold, valid_pred, average = 'weighted'))
      print("\n Valid Recall : ", recall_score(valid_gold, valid_pred, average = 'weighted'))
      print("\n Valid F1 score : ", f1_score(valid_gold, valid_pred, average = 'weighted'))


      curr_f1 = f1_score(valid_gold, valid_pred, average = 'weighted')

      # curr_loss = valid_loss
      # if((curr_f1 > best_f1) and (epoch>=4)):
      if(curr_f1 > best_f1):
      # if(curr_loss < min_loss):
      # if(curr_loss < lowest_loss):
        best_f1 = curr_f1
        # min_loss = curr_loss

        torch.save(model.state_dict(), 'bart_w2v_dino_model.pth')
        print("model saved\n")

    return model


model = train_and_validation(model, train_loader, valid_loader)

test_pred, test_gold = test_epoch(model, test_loader)

test_accuracy = accuracy_score(test_gold, test_pred)
test_precision = precision_score(test_gold, test_pred, average = 'weighted')
test_recall = recall_score(test_gold, test_pred, average = 'weighted')
test_f1 = f1_score(test_gold, test_pred, average = 'weighted')

# wandb.log({"Test_accuracy": test_accuracy, "Test_precision": test_precision, "Test_recall": test_recall, "Test_f1": test_f1})

print("Test accuracy : ", test_accuracy)
print("Test Precision : ", test_precision)
print("Test Recall : ", test_recall)
print("Test F1 score : ", test_f1)
          


# Test the model for specific videos
test_videos = ['hate_video_1', 'hate_video_2', 'hate_video_10', 'hate_video_16', 'hate_video_17', 'hate_video_21', 'hate_video_26', 'hate_video_34',
'hate_video_41', 'hate_video_46', 'hate_video_103', 'hate_video_426', 'non_hate_video_4', 'non_hate_video_7', 'non_hate_video_10', 'non_hate_video_62', 'non_hate_video_130',
'non_hate_video_328', 'non_hate_video_406', 'non_hate_video_593']

model.load_state_dict(torch.load('bart_w2v_dino_model.pth'))
model.eval()
with torch.no_grad():
    test_dataset_instance = HateMMDataset(all_test_data, all_test_label)  # Create an instance of the test dataset
    val_dataset_instance = HateMMDataset(all_val_data, all_val_label)  # Create an instance of the validation dataset
    for video_id in test_videos:
        video_id = video_id + '.mp4'
        dataset_instance = test_dataset_instance if video_id in all_test_data else val_dataset_instance
        data_list = all_test_data if video_id in all_test_data else all_val_data
        label_list = all_test_label if video_id in all_test_data else all_val_label
        
        if video_id not in data_list:
            continue
        try:
            # Load data for the video
            X_text, X_vid, X_audio = dataset_instance.load_data_for_video(video=video_id)
            true_label_index = data_list.index(video_id)
            true_label = label_list[true_label_index]
            
            # Convert data to tensors and move to device, handling missing modalities
            if X_text is not None and X_vid is not None and X_audio is not None:
                X_input_ids = torch.tensor(X_text['input_ids'], dtype=torch.long).unsqueeze(0).to(DEVICE)
                X_attention_masks = torch.tensor(X_text['attention_mask'], dtype=torch.bool).unsqueeze(0).to(DEVICE)
                X_vid = X_vid.to(DEVICE)
                X_audio = X_audio.unsqueeze(0).to(DEVICE)
                # true_label = torch.tensor([true_label]).unsqueeze(0).to(DEVICE)
            else:
                continue

            # Make predictions
            outputs = model(X_input_ids, X_attention_masks, X_audio, X_vid)
            _, predicted = torch.max(outputs['logits'], 1)

            # Print the results
            print(f"Video ID: {video_id}, Predicted Label: {predicted.item()}, True Label: {true_label}")

        except Exception as e:
            print(f"Error processing video {video_id}: {e}")
            continue
        