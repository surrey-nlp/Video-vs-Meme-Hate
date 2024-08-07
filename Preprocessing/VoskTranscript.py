import os
from moviepy.editor import VideoFileClip, AudioFileClip
from vosk import Model, KaldiRecognizer
import json
from pydub import AudioSegment

# Path to the Dataset folder containing video files
video_folder = "/backup/girish_datasets/HateMM/non_hate_videos/"
#
# Initialize the Vosk model with the language code
model = Model("./vosk-model-en-us-0.22/")

# Path to the log file that keeps track of processed videos
processed_log_path = "processed_videos.txt"

def extract_audio_from_video(video_path, audio_path):
    try:
        video = VideoFileClip(video_path)
        # video.audio.write_audiofile(audio_path, codec='pcm_s16le', ffmpeg_params=["-ar", "16000"])
        if video.audio:
            audio = video.audio
            audio.write_audiofile(audio_path, codec='pcm_s16le', ffmpeg_params=["-ar", "16000"])
            return True
        else:
            print(f"No audio found in {video_path}")
            return False
    except Exception as e:
        # print(f"Error extracting audio from {video_path}: {e}")
        if 'video_fps' in str(e):
            print(f"Video contains no video content. Attempting direct audio extraction.")
            return extract_audio_directly(video_path, audio_path)
        else:
            print(f"Error extracting audio from {video_path}: {e}")
            return False

def extract_audio_directly(video_path, audio_path):
    try:
        audio = AudioFileClip(video_path)
        audio.write_audiofile(audio_path, codec='pcm_s16le', ffmpeg_params=["-ar", "16000"])
        return True
    except Exception as e:
        print(f"Error extracting audio directly from {video_path}: {e}")
        return False

def convert_to_mono(audio_path):
    """
    Checks if the audio file is stereo and converts it to mono if necessary.
    """
    sound = AudioSegment.from_wav(audio_path)
    if sound.channels > 1:
        sound = sound.set_channels(1)
        sound.export(audio_path, format="wav")

def transcribe_audio(audio_path):
    try:
        # Ensure the audio is in mono before transcribing
        convert_to_mono(audio_path)
        wf = open(audio_path, "rb")
        rec = KaldiRecognizer(model, 16000)
        text = ""
        while True:
            data = wf.read(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                jres = json.loads(rec.Result())
                text += " " + jres["text"]
        jres = json.loads(rec.FinalResult())
        text += " " + jres["text"]
        return text.strip()
    except Exception as e:
        print(f"Error transcribing audio from {audio_path}: {e}")
        return ""

def load_processed_videos():
    if os.path.exists(processed_log_path):
        with open(processed_log_path, 'r') as file:
            return set(file.read().splitlines())
    return set()

def log_processed_video(video_name):
    with open(processed_log_path, 'a') as file:
        file.write(video_name + "\n")

def process_videos(video_folder):
    transcripts = {}
    processed_videos = load_processed_videos()
    for video_file in os.listdir(video_folder):
        if video_file.endswith(".mp4") and video_file not in processed_videos:
            video_path = os.path.join(video_folder, video_file)
            audio_path = video_path.replace(".mp4", ".wav")
            transcript_path = video_path.replace(".mp4", ".txt")
            
            print(f"Processing {video_file}...")
            if extract_audio_from_video(video_path, audio_path):
                transcript = transcribe_audio(audio_path)
                if transcript:
                    transcripts[video_file] = transcript
                    
                    with open(transcript_path, 'w') as f:
                        f.write(transcript)
                    
                    # Log the processed video
                    log_processed_video(video_file)
                    
                    # Optionally, remove the audio file to save space
                    # os.remove(audio_path)
                else:
                    print(f"Failed to transcribe {video_file}")
            else:
                print(f"Failed to extract audio from {video_file}")
        else:
            print(f"Skipping already processed file: {video_file}")
    
    return transcripts

if __name__ == "__main__":
    transcripts = process_videos(video_folder)
    print("Transcription completed.")