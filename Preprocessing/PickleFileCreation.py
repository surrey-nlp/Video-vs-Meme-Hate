import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import pickle

def add_transcripts_to_pickle(directory, pickle_file):
    transcripts = {}
    for filename in os.listdir(directory):
        if filename.endswith("whisper_tiny.txt"):
            with open(os.path.join(directory, filename), 'r') as file:
                transcripts[filename.replace("_whisper_tiny.txt", ".mp4")] = file.read()
    
    if os.path.getsize(pickle_file) > 0:        
        with open(pickle_file, 'rb') as fp:
            existing_data = pickle.load(fp)
    else:
        existing_data = {}

    existing_data.update(transcripts)
    # print(existing_data)
    
    with open(pickle_file, 'wb') as fp:
        pickle.dump(existing_data, fp)

# add_transcripts_to_pickle('/backup/girish_datasets/HateMM/hate_videos/', '/backup/girish_datasets/HateMM/all_whisper_tiny_transcripts.pkl')


def add_audio_paths_to_pickle(directory, pickle_file):
    audio_paths = {}
    for filename in os.listdir(directory):
        if filename.endswith(".mp4"):
            audio_paths[filename.replace(".mp4", "")] = os.path.join(directory, filename)
    
    if os.path.getsize(pickle_file) > 0:        
        with open(pickle_file, 'rb') as fp:
            existing_data = pickle.load(fp)
    else:
        existing_data = {}

    existing_data.update(audio_paths)
    
    with open(pickle_file, 'wb') as fp:
        pickle.dump(existing_data, fp)

# add_audio_paths_to_pickle('/backup/girish_datasets/HateMM/non_hate_videos/', '/backup/girish_datasets/HateMM/final_allVideos.p')


def add_video_frames_paths_to_pickle(directory, pickle_file):
    video_frames_paths = {}
    for video_folder in os.listdir(directory):
        video_folder_path = os.path.join(directory, video_folder)
        if os.path.isdir(video_folder_path):
            frame_paths = []
            for frame_file in sorted(os.listdir(video_folder_path)):
                if frame_file.endswith((".jpg", ".png")):  # Assuming frames are in jpg or png format
                    frame_path = os.path.join(video_folder_path, frame_file)
                    frame_paths.append(frame_path)
            video_frames_paths[video_folder] = frame_paths
    
    # Check if the pickle file already exists and has content
    if os.path.exists(pickle_file) and os.path.getsize(pickle_file) > 0:
        with open(pickle_file, 'rb') as fp:
            existing_data = pickle.load(fp)
    else:
        existing_data = {}
    
    # Update the existing data with new video frames paths
    existing_data.update(video_frames_paths)
    
    # Write the updated data to the pickle file
    with open(pickle_file, 'wb') as fp:
        pickle.dump(existing_data, fp)

# add_video_frames_paths_to_pickle('/backup/girish_datasets/HateMM_Images/', '/backup/girish_datasets/HateMM/final_allImageFrames.p')



def prepare_folds_data(annotations_path, output_path):
    # Load the annotations
    df = pd.read_csv(annotations_path)

    # Perform integer encoding for 'label' column
    df['label'] = df['label'].apply(lambda x: 1 if x == 'Hate' else 0)

    # Prepare train, validation, and test data
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=2024)
    train_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_df['label'], random_state=2024)

    # Prepare the folds data dictionary
    folds_data = {
        'train': (train_df['video_file_name'].tolist(), train_df['label'].tolist()),
        'val': (val_df['video_file_name'].tolist(), val_df['label'].tolist()),
        'test': (test_df['video_file_name'].tolist(), test_df['label'].tolist())
    }

    print("Train size:", len(train_df))
    print("Validation size:", len(val_df))
    print("Test size:", len(test_df))

    # Save the fold details to a pickle file
    with open(output_path, 'wb') as fp:
        pickle.dump(folds_data, fp)

    print("Fold details saved to:", output_path)

# Specify the paths
annotations_path = '/backup/girish_datasets/HateMM/HateMM_annotation.csv'
output_path = '/backup/girish_datasets/HateMM/noFoldDetails.pkl'

# Call the function
# prepare_folds_data(annotations_path, output_path)



def convert_list_to_dict_in_pickle_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".pkl"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'rb') as fp:
                data_list = pickle.load(fp)
            
            # Extract the base filename without the .pkl extension, remove '_vit', and append .mp4
            base_filename = filename[:-4]  # Remove the .pkl extension
            if base_filename.endswith('_clip'):
                # base_filename = base_filename[:-4]  # Remove '_vit'
                base_filename = base_filename[:-5]  # Remove '_clip'
                # base_filename = base_filename[:-16]  # Remove '_DINOv2_features'
            video_name_key = base_filename + ".mp4"
            data_dict = {video_name_key: data_list}
            
            with open(file_path, 'wb') as fp:
                pickle.dump(data_dict, fp)
            print(f"Converted {filename} to dictionary format.")

# Specify the directory containing the .pkl files
VITF_FOLDER = '/backup/girish_datasets/HateMM/CLIP_lhs/'
# convert_list_to_dict_in_pickle_files(VITF_FOLDER)
