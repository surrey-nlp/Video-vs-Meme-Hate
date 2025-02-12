import pickle
import torch
from Simple-Fusion.HateMM_Fusion import Text_Model, Aud_Model, Combined_model, LSTM
from Simple-Fusion.HateMM_Fusion import HateMM_Dataset
import torch.nn as nn

FOLDER_NAME = '/backup/girish_datasets/HateMM/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Audio parameters
input_size_text = 768   # 512 for CLIP, 768 for HXP, BERT and CLAP

input_size_audio = 49152   # 49152 for CLAP, 768 for Wav2Vec2, 1000 for AudioVGG19, 40 for MFCC

fc1_hidden_audio, fc2_hidden_audio = 128, 128
k = 2 # Number of classes

tex = Text_Model(input_size_text, fc1_hidden_audio, fc2_hidden_audio, 64).to(device)
vid = LSTM().to(device)
aud = Aud_Model(input_size_audio, fc1_hidden_audio, fc2_hidden_audio, 64).to(device)
comb = Combined_model(tex, vid, aud, k).to(device)

# Parallelize model to multiple GPUs
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    comb = nn.DataParallel(comb)

with open(FOLDER_NAME+'all_HateXPlainembedding_whisper.pkl','rb') as fp:
    textData = pickle.load(fp)

with open(FOLDER_NAME+'CLAP_features.pkl','rb') as fp:
    audData = pickle.load(fp)

with open(FOLDER_NAME+'noFoldDetails.pkl', 'rb') as fp:
    allDataAnnotation = pickle.load(fp)

all_train_data = []
all_train_label = []
all_val_data = []
all_val_label = []
all_test_data = []
all_test_label = []

all_train_data, all_train_label = allDataAnnotation['train']
all_val_data, all_val_label = allDataAnnotation['val']
all_test_data, all_test_label = allDataAnnotation['test']

def load_model(model_path, model, device):
    # Load the model state dict that was saved after training
    state_dict = torch.load(model_path, map_location=device)

    # If the model was trained using nn.DataParallel, which saves the model with a 'module.' prefix
    # We need to remove this prefix from each key
    new_state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}

    # Load the adjusted state dict into the model
    model.load_state_dict(new_state_dict)

    return model


# Test the model for specific videos from validation and test sets
test_videos = ['hate_video_1', 'hate_video_2', 'hate_video_10', 'hate_video_16', 'hate_video_17', 'hate_video_21', 'hate_video_26', 'hate_video_34',
'hate_video_41', 'hate_video_46', 'hate_video_103', 'hate_video_426', 'non_hate_video_4', 'non_hate_video_7', 'non_hate_video_10', 'non_hate_video_62', 'non_hate_video_130',
'non_hate_video_328', 'non_hate_video_406', 'non_hate_video_593']

comb = load_model('hxp_clap_clip_pooled_lstm.pth', comb, device)    # Load the finetuned model

comb.eval()
with torch.no_grad():
    test_dataset_instance = HateMM_Dataset(all_test_data, all_test_label)  # Create an instance of the test dataset
    val_dataset_instance = HateMM_Dataset(all_val_data, all_val_label)  # Create an instance of the validation dataset
    for video_id in test_videos:
        video_id = video_id + '.mp4'
        dataset_instance = test_dataset_instance if video_id in all_test_data else val_dataset_instance
        data_list = all_test_data if video_id in all_test_data else all_val_data
        label_list = all_test_label if video_id in all_test_data else all_val_label
        
        if video_id not in data_list:
            continue
        try:
            # Load data for the video
            X_text, X_vid, X_audio = dataset_instance.load_data_for_video(selected_folder=video_id)
            true_label_index = data_list.index(video_id)
            true_label = label_list[true_label_index]
            
            # Convert data to tensors and move to device, handling missing modalities
            if X_text is not None and X_vid is not None and X_audio is not None:
                X_text = X_text.unsqueeze(0).to(device)
                X_vid = X_vid.to(device)
                X_audio = X_audio.unsqueeze(0).to(device)
                # true_label = torch.tensor([true_label]).unsqueeze(0).to(DEVICE)
            else:
                continue

            # Make predictions
            outputs = comb(X_text, X_vid, X_audio)
            predicted = outputs.max(1, keepdim=True)[1]

            # Print the results
            print(f"Video ID: {video_id}, Predicted Label: {predicted.item()}, True Label: {true_label}")

        except Exception as e:
            print(f"Error processing video {video_id}: {e}")
            continue
        