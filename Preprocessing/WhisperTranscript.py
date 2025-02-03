import os
import soundfile as sf

import torch
from transformers import pipeline
from transformers import AutoProcessor, WhisperForConditionalGeneration

processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

# Path to the Dataset folder containing video files
video_folder = "/backup/girish_datasets/HateMM/hate_videos"

# Path to the log file that keeps track of processed videos
processed_log_path = "processed_audios.txt"

device = "cuda:0" if torch.cuda.is_available() else "cpu"

pipe = pipeline(
  "automatic-speech-recognition",
  model="openai/whisper-tiny.en",
  chunk_length_s=30,
  device=device,
)

def load_processed_audios():
    if os.path.exists(processed_log_path):
        with open(processed_log_path, 'r') as file:
            return set(file.read().splitlines())
    return set()

def log_processed_audio(video_name):
    with open(processed_log_path, 'a') as file:
        file.write(video_name + "\n")

def process_audios(video_folder):
    transcripts = {}
    processed_audios = load_processed_audios()
    for audio_file in os.listdir(video_folder):
        if audio_file.endswith(".wav") and audio_file not in processed_audios:
            audio_path = os.path.join(video_folder, audio_file)
            transcript_path = audio_path.replace(".wav", "_whisper_tiny.txt")
            sample = sf.read(audio_path)
            print(f"Processing {audio_file}...")
            if sample:
                try:
                    transcript = pipe(sample[0].copy(), batch_size=8)["text"]
                    if transcript:
                        transcripts[audio_file] = transcript
                        with open(transcript_path, 'w') as f:
                            f.write(transcript)
                        log_processed_audio(audio_file)
                    else:
                        print(f"Failed to transcribe {audio_file}")
                except StopIteration:
                    print(f"StopIteration error occurred while processing {audio_file}")
            else:
                print(f"Failed to extract audio from {audio_file}")
        else:
            print(f"Skipping already processed file: {audio_file}")
    
    return transcripts

if __name__ == "__main__":
    transcripts = process_audios(video_folder)
    print("Transcription completed.")