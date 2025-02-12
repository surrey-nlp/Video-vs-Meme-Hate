FOLDER_NAME = '/backup/girish_datasets/HateMM/'

import os
import pickle
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, recall_score, precision_score
from tqdm import tqdm
import random

def fix_the_random(seed_val = 2021):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

fix_the_random(2024)

class Text_Model(nn.Module):
    def __init__(self, input_size, fc1_hidden, fc2_hidden, output_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, fc1_hidden),
            nn.ReLU(),
            nn.Linear(fc1_hidden, fc2_hidden),
            nn.ReLU(),
            nn.Linear(fc2_hidden, output_size),
        )
        self.output_size = output_size

    def forward(self, xb):
        return self.network(xb)
 
class LSTM(nn.Module):
    def __init__(self, input_emb_size = 768, no_of_frames = 100):   # 768 for CLIP and ViT, 384 for DINOv2
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_emb_size, 128)
        self.fc = nn.Linear(128*no_of_frames, 64)
        self.output_size = 64
        
    def forward(self, x):
        x = x.squeeze(1)  # Remove the 2nd dimension
        x, _ = self.lstm(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x 

class Aud_Model(nn.Module):
    def __init__(self, input_size, fc1_hidden, fc2_hidden, output_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, fc1_hidden),
            nn.ReLU(),
            nn.Linear(fc1_hidden, fc2_hidden),
            nn.ReLU(),
            nn.Linear(fc2_hidden, output_size),
        )
        self.output_size = output_size

    def forward(self, xb):
        return self.network(xb)

class RulesFromProbabilities(nn.Module):
    def __init__(self, visual_model, textual_model, audio_model, num_classes):
        super(RulesFromProbabilities, self).__init__()
        self.visual_model = visual_model
        self.textual_model = textual_model
        self.audio_model = audio_model
        self.num_classes = num_classes
        # self.dropout = nn.Dropout(dropout_rate)
        self.visual_classifier = nn.Linear(visual_model.output_size, num_classes)
        self.textual_classifier = nn.Linear(textual_model.output_size, num_classes)
        self.audio_classifier = nn.Linear(audio_model.output_size, num_classes)
        self.final_classifier = nn.Linear(num_classes * 3, num_classes)

    def forward(self, x_text, x_img, x_audio):
        visual_output = self.visual_model(x_img)
        textual_output = self.textual_model(x_text)
        audio_output = self.audio_model(x_audio)

        visual_probabilities = self.visual_classifier(visual_output)
        textual_probabilities = self.textual_classifier(textual_output)
        audio_probabilities = self.audio_classifier(audio_output)

        # visual_probabilities = visual_probabilities.squeeze(1)

        combined_probabilities = torch.cat((visual_probabilities, textual_probabilities, audio_probabilities), dim=1)
        combined_probabilities = self.dropout(combined_probabilities)

        output = self.final_classifier(combined_probabilities)

        return output

class WeightingTechnique(nn.Module):
    def __init__(self, visual_model, textual_model, audio_model, num_classes):
        super(WeightingTechnique, self).__init__()
        self.visual_model = visual_model
        self.textual_model = textual_model
        self.audio_model = audio_model
        self.num_classes = num_classes
        self.weight_visual = nn.Parameter(torch.randn(1, requires_grad=True))
        self.weight_textual = nn.Parameter(torch.randn(1, requires_grad=True))
        self.weight_audio = nn.Parameter(torch.randn(1, requires_grad=True))

        # print(f"visual_model output_size: {visual_model.output_size}, textual_model output_size: {textual_model.output_size}")
        self.classifier = nn.Linear(visual_model.output_size + textual_model.output_size + audio_model.output_size, num_classes)

    def forward(self, x_text, x_img, x_audio):
        visual_output = self.visual_model(x_img)
        textual_output = self.textual_model(x_text)
        audio_output = self.audio_model(x_audio)

        weighted_visual = self.weight_visual * visual_output
        weighted_textual = self.weight_textual * textual_output
        weighted_audio = self.weight_audio * audio_output

        # weighted_visual = weighted_visual.squeeze(1)

        combined_output = torch.cat((weighted_visual, weighted_textual, weighted_audio), dim=1)
        output = self.classifier(combined_output)

        return output

class Combined_model(nn.Module):
    def __init__(self, text_model, video_model, audio_model, num_classes):
        super().__init__()
        self.text_model = text_model
        self.audio_model = audio_model
        self.video_model = video_model
        self.num_classes = num_classes
        self.fc_output = nn.Linear(3*64, num_classes)

        # self.weighting_technique = WeightingTechnique(self.video_model, self.text_model, self.audio_model, num_classes)
        # self.rules_from_probabilities = RulesFromProbabilities(self.video_model, self.text_model, self.audio_model, num_classes)

    def forward(self, x_text, x_vid, x_audio):
        # out = self.weighting_technique(x_text, x_vid, x_audio)
        # out = self.rules_from_probabilities(x_text, x_vid, x_audio)

        if x_text is not None:
            tex_out = self.text_model(x_text)
        else:
            tex_out = torch.zeros(x_vid.size(0), 64).to(x_vid.device) if x_vid is not None else torch.zeros(x_audio.size(0), 64).to(x_audio.device)

        if x_vid is not None:
            vid_out = self.video_model(x_vid)
        else:
            vid_out = torch.zeros(x_text.size(0), 64).to(x_text.device) if x_text is not None else torch.zeros(x_audio.size(0), 64).to(x_audio.device)

        if x_audio is not None:
            aud_out = self.audio_model(x_audio)
        else:
            aud_out = torch.zeros(x_text.size(0), 64).to(x_text.device) if x_text is not None else torch.zeros(x_vid.size(0), 64).to(x_vid.device)

        if len(tex_out.shape) > 2:
            tex_out = tex_out.view(tex_out.size(0), -1)
        if len(vid_out.shape) > 2:
            vid_out = vid_out.view(vid_out.size(0), -1)
        if len(aud_out.shape) > 2:
            aud_out = aud_out.view(aud_out.size(0), -1)

        # Element-wise multiplication
        # inp = tex_out * vid_out * aud_out
        # inp = inp.view(inp.size(0), -1)

        # Ensure that the input tensor shape matches the linear layer's weight tensor shape
        # expected_input_size = 3*64
        # if inp.size(1) < expected_input_size:
        #     padding_size = expected_input_size - inp.size(1)
        #     inp = torch.cat([inp, torch.zeros(inp.size(0), padding_size, device=inp.device)], dim=1)

        # inp = torch.cat((tex_out, vid_out, aud_out), dim = 1)
        # inp = torch.cat((torch.zeros_like(tex_out), vid_out, aud_out), dim = 1)
        # inp = torch.cat((tex_out, vid_out, torch.zeros_like(aud_out)), dim = 1)
        # inp = torch.cat((tex_out, torch.empty_like(vid_out), aud_out), dim = 1)
        # inp = torch.cat((torch.zeros_like(tex_out), torch.zeros_like(vid_out), torch.zeros_like(aud_out)), dim = 1)
        # inp = torch.cat((tex_out, torch.zeros_like(vid_out), torch.zeros_like(aud_out)), dim = 1)
        # inp = torch.cat((torch.zeros_like(tex_out), vid_out, torch.zeros_like(aud_out)), dim = 1)
        inp = torch.cat((torch.zeros_like(tex_out), torch.zeros_like(vid_out), aud_out), dim = 1)
        # print("Input tensor: ", inp)
        out = self.fc_output(inp)
        return out

class HateMM_Dataset(data.Dataset):
    "Characterizes a dataset for PyTorch"
    def __init__(self, folders, labels):
        "Initialization"
        self.labels = labels
        self.folders = folders

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.folders)

    def load_data_for_video(self, selected_folder):
        video_file_name_without_extension, _ = os.path.splitext(selected_folder)
        # pickle_file_path = os.path.join(FOLDER_NAME, "VITF_new", video_file_name_without_extension + "_vit.pkl")
        pickle_file_path = os.path.join(FOLDER_NAME, "CLIP_pooled", video_file_name_without_extension + "_clip.pkl")
        # pickle_file_path = os.path.join(FOLDER_NAME, "DINOv2_lhs", video_file_name_without_extension + "_DINOv2_features.pkl")
        
        # Load text data
        if selected_folder in textData:
            text_features = torch.tensor(np.array(textData[selected_folder]), dtype=torch.float32)
        else:
            raise ValueError(f"Text data not found for {selected_folder}")
        
        # Load video data
        try:
            with open(pickle_file_path, 'rb') as fp:
                video_data = pickle.load(fp)
                # video_features = torch.tensor(np.array(list(video_data.values())), dtype=torch.float32)     # for last hidden state features
                video_features = torch.stack([tensor.detach() for tensor in video_data.values()])     # for pooled features
        except FileNotFoundError:
            raise ValueError(f"Video data file not found: {pickle_file_path}")
        
        # Load audio data
        if selected_folder in audData:
            audio_features = torch.tensor(np.array(audData[selected_folder]), dtype=torch.float32)
            # audio_features = audio_features.mean(dim=0).unsqueeze(0)      # mean of wav2vec2 features across all frames
            # audio_features = audio_features[0]    # first frame of wav2vec2 features
            audio_features = audio_features.view(audio_features.size(0), -1)    # for CLAP
        else:
            raise ValueError(f"Audio data not found for {selected_folder}")
        
        return text_features, video_features, audio_features

    def __getitem__(self, index):
        "Generates one sample of data"
        try:
            folder = self.folders[index]
            X_text, X_vid, X_audio = self.load_data_for_video(folder)
            y = torch.LongTensor([self.labels[index]]) 
            return X_text, X_vid, X_audio, y
        except Exception as e:
            # traceback.print_exc()
            print(f"Error loading data for index {index}: {e}")                        
            return None

def evalMetric(y_true, y_pred):
    try:
        accuracy = accuracy_score(y_true, y_pred)
        mf1Score = f1_score(y_true, y_pred, average='macro', zero_division='warn')
        f1Score  = f1_score(y_true, y_pred, labels = np.unique(y_pred), zero_division='warn')
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        area_under_c = auc(fpr, tpr)
        recallScore = recall_score(y_true, y_pred, labels = np.unique(y_pred), zero_division='warn')
        precisionScore = precision_score(y_true, y_pred, labels = np.unique(y_pred), zero_division='warn')
    except:
        return dict({"accuracy": 0, 'mF1Score': 0, 'f1Score': 0, 'auc': 0,'precision': 0, 'recall': 0})
    return dict({"accuracy": accuracy, 'mF1Score': mf1Score, 'f1Score': f1Score, 'auc': area_under_c, 'precision': precisionScore, 'recall': recallScore})

# with open(FOLDER_NAME+'all_HateXPlainembedding_vosk.pkl','rb') as fp:
with open(FOLDER_NAME+'all_HateXPlainembedding_whisper.pkl','rb') as fp:
# with open(FOLDER_NAME+'all_rawBERTembedding.pkl','rb') as fp:
# with open(FOLDER_NAME+'all_hatemm_clap_embedding_truncated.pkl','rb') as fp:
# with open(FOLDER_NAME+'all_hatemm_clip_embedding_truncated.pkl','rb') as fp:
    textData = pickle.load(fp)

# with open(FOLDER_NAME+'vgg19_audFeatureMap.pkl','rb') as fp:
# with open(FOLDER_NAME+'MFCCFeaturesNew.pkl','rb') as fp:
# with open(FOLDER_NAME+'Wav2Vec2_features_chunked.pkl','rb') as fp:
with open(FOLDER_NAME+'CLAP_features.pkl','rb') as fp:
    audData = pickle.load(fp)
  

# Audio parameters
input_size_text = 768   # 512 for CLIP, 768 for HXP, BERT and CLAP

input_size_audio = 49152   # 49152 for CLAP, 768 for Wav2Vec2, 1000 for AudioVGG19, 40 for MFCC

fc1_hidden_audio, fc2_hidden_audio = 128, 128

# training parameters
k = 2            # number of target category
epochs = 20
batch_size = 32
learning_rate = 1e-4
log_interval = 100

wandb.init(
    project="hate-video-classification",
    config={
        "learning_rate": learning_rate,
        "architecture": "CLAP concat with empty tensors",
        "dataset": "HateMM",
        "epochs": epochs,
        "batch_size": batch_size,
    },
)

def train(log_interval, model, device, train_loader, optimizer, epoch):
    model.train()

    losses = []
    scores = []
    N_count = 0   # counting total trained sample in one epoch

    for batch_idx, batch in enumerate(train_loader):
        # print("Batch:", batch)
        if batch is None:  # Skip the batch if collate_fn returned None
            continue
        X_text, X_vid, X_aud, y = batch

        # distribute data to device 
        X_text, X_vid, X_aud, y = (X_text.float()).to(device), (X_vid.float()).to(device), (X_aud.float()).to(device), y.to(device)
    
        N_count += X_text.size(0)

        optimizer.zero_grad()
        output = model(X_text, X_vid, X_aud)  # output size = (batch, number of classes)

        y = y.squeeze()
        loss = F.cross_entropy(output, y, weight=torch.FloatTensor([0.41, 0.59]).to(device))
        # loss = F.cross_entropy(output, y)
        losses.append(loss.item())

        # to compute accuracy
        y_pred = torch.max(output, 1)[1]  # y_pred != output
        metrics = evalMetric(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy())
        scores.append(metrics)  

        loss.backward()
        optimizer.step()

        wandb.log({"loss": loss.item(), "accuracy": metrics['accuracy'], "f1": metrics['f1Score'], "mF1": metrics['mF1Score'], 
                   "auc": metrics['auc'], "precision": metrics['precision'], "recall": metrics['recall']})

        # if (batch_idx + 1) % log_interval == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}, Accu: {:.4f}, MF1 Score: {:.4f}, F1 Score: {:.4f}, Area Under Curve: {:.4f}, Precision: {:.4f}, Recall Score: {:.4f}'.format(epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item(), metrics['accuracy'], metrics['mF1Score'], metrics['f1Score'], metrics['auc'], metrics['precision'], metrics['recall']))

    return losses, scores

def validation(model, device, test_loader, testingType = "Test"):
    model.eval()

    test_loss = 0
    all_y = []
    all_y_pred = []
    with torch.no_grad():
        for batch in test_loader:
            if batch is None:
                continue
            X_text, X_vid, X_aud, y = batch
            # distribute data to device
            X_text, X_vid, X_aud, y = (X_text.float()).to(device), (X_vid.float()).to(device), (X_aud.float()).to(device), y.to(device)

            output = model(X_text, X_vid, X_aud)

            y = y.squeeze()
            loss = F.cross_entropy(output, y, reduction='sum')
            test_loss += loss.item()                 # sum up batch loss
            y_pred = output.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-probability

            # collect all y and y_pred in all batches
            all_y.extend(y)
            all_y_pred.extend(y_pred)

    test_loss /= len(test_loader.dataset)

    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)

    print("====================")
    metrics = evalMetric(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())

    wandb.log({f"{testingType}_loss": test_loss, f"{testingType}_accuracy": metrics['accuracy'], f"{testingType}_f1": metrics['f1Score'], f"{testingType}_mF1": metrics['mF1Score'],
                f"{testingType}_auc": metrics['auc'], f"{testingType}_precision": metrics['precision'], f"{testingType}_recall": metrics['recall']})

    print('\n '+testingType+' set -> Average loss: {:.4f}, Accuracy: {:.4f}, MF1 Score: {:.4f}, F1 Score: {:.4f}, Area Under Curve: {:.4f}, Precision: {:.4f}, Recall Score: {:.4f}'.format(
                test_loss, metrics['accuracy'], metrics['mF1Score'], metrics['f1Score'], metrics['auc'], metrics['precision'], metrics['recall']))

    return test_loss, metrics, list(all_y_pred.cpu().data.squeeze().numpy())


# Detect devices
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU

params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True} if use_cuda else {}
valParams = {'batch_size': batch_size, 'shuffle': False, 'num_workers': 2, 'pin_memory': True} if use_cuda else {}

# with open(FOLDER_NAME+'allFoldDetails.pkl', 'rb') as fp:
with open(FOLDER_NAME+'noFoldDetails.pkl', 'rb') as fp:
    allDataAnnotation = pickle.load(fp)


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:  # Check if the batch is empty after filtering
        return None

    return torch.utils.data.dataloader.default_collate(batch)


all_train_data, all_train_label = [] , []
all_val_data, all_val_label = [] , []
all_test_data, all_test_label = [] , []

tex = Text_Model(input_size_text, fc1_hidden_audio, fc2_hidden_audio, 64).to(device)
vid = LSTM().to(device)
aud = Aud_Model(input_size_audio, fc1_hidden_audio, fc2_hidden_audio, 64).to(device)
comb = Combined_model(tex, vid, aud, k).to(device)


# Parallelize model to multiple GPUs
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    comb = nn.DataParallel(comb)

optimizer = torch.optim.Adam(comb.parameters(), lr=learning_rate) 

all_train_data, all_train_label = allDataAnnotation['train']
all_val_data, all_val_label = allDataAnnotation['val']
all_test_data, all_test_label = allDataAnnotation['test']

print("Train data size:", len(all_train_data), "Val data size:", len(all_val_data), "Test data size:", len(all_test_data))
# print("Train label size:", len(all_train_label), "Val label size:", len(all_val_label), "Test label size:", len(all_test_label))

train_set, valid_set , test_set = HateMM_Dataset(all_train_data, all_train_label), HateMM_Dataset(all_val_data, all_val_label), HateMM_Dataset(all_test_data, all_test_label)
train_loader = data.DataLoader(train_set, collate_fn = collate_fn, **params)
test_loader = data.DataLoader(test_set, collate_fn = collate_fn, **valParams)
valid_loader = data.DataLoader(valid_set, collate_fn = collate_fn, **valParams)


epoch_train_losses = []
epoch_train_scores = []
epoch_test_losses = []
epoch_test_scores = []

validFinalValue = None
testFinalValue = None
finalScoreAcc = 0
prediction  = None

# start training
for epoch in tqdm(range(epochs)):
    train_losses, train_scores = train(log_interval, comb, device, train_loader, optimizer, epoch)
    test_loss1, test_scores1, veValid_pred = validation(comb, device, valid_loader, 'Valid')
    if (test_scores1['mF1Score']>finalScoreAcc):
        finalScoreAcc = test_scores1['mF1Score']
        validFinalValue = test_scores1

    # save results
    epoch_train_losses.append(train_losses)
    epoch_train_scores.append(list(x['accuracy'] for x in train_scores))

# torch.save(comb.state_dict(), 'hxp_clap_clip_pooled_lstm.pth')

test_loss, test_scores, veTest_pred = validation(comb, device, test_loader, 'Test')
        