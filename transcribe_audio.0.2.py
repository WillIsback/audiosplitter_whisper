import os
import yaml
import torch
import csv
import whisperx
import gc
import torch
from multiprocessing import Pool
import time  # Add this at the beginning of your script


# CONSTANTS
AUDIO_EXT = ".wav"  # The extension for audio files
CONFIG_PATH = "conf.yaml"  # The path to the configuration file
DATASET_DIR = "dataset"  # The directory containing the dataset
OUTPUT_DIR = "output"  # The directory to store the output files
LOG_FILE = "transcribe_log.txt"  # The file to log errors
PROCESSED_FILE = "processed.txt"  # The file to keep track of processed audio files
NUM_WORKERS = 4  # The number of worker processes
# FUNCTIONS

def load_settings(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)
    
def get_device_info():
    if torch.cuda.is_available():
        print('CUDA is available. Running on GPU.')
        return 'cuda', "float16"
    else:
        print('CUDA is not available. Running on CPU.')
        return 'cpu', "int8"
    
def load_audio_file(args):
    speaker, audio_file, speaker_dir = args
    audio_path = os.path.join(speaker_dir, audio_file)
    audio = whisperx.load_audio(audio_path)
    return audio, f"{speaker}/{audio_file}"

# Function transcribes the audio files
def run_whisperx(dataset_dir, output_dir, settings, device, compute_type, model):
    with open('train.txt', 'a', encoding='utf-8') as f, open(LOG_FILE, 'a') as log_file, open(PROCESSED_FILE, 'a') as pf:
        successful_transcriptions = 0  # The number of successful transcriptions
        # Load the processed files
        start_time = time.time()  # Start timing
        if os.path.exists(PROCESSED_FILE):
            with open(PROCESSED_FILE, 'r') as pf_read:
                processed_files = pf_read.read().splitlines()
        else:
            processed_files = []
        end_time = time.time()  # End timing
        print(f"Loading processed files took {end_time - start_time} seconds")  # Print the time taken


        with open(os.path.join(dataset_dir, 'info.csv'), 'r') as csv_file:
            reader = csv.reader(csv_file)
            next(reader)
            for row in reader:
                speaker, num_files, *_ = row
                speaker_dir = os.path.join(dataset_dir, speaker, 'wav')
                if not os.path.exists(speaker_dir):
                    log_file.write(f"Speaker directory {speaker_dir} does not exist.\n")
                    continue
                try:
                    audio_files = [file for file in os.listdir(speaker_dir) if file.endswith('.wav') and f"{speaker}/{file}" not in processed_files]
                except Exception as e:
                    log_file.write(f"Error reading directory {speaker_dir}: {str(e)}\n")
                    continue

            with Pool(NUM_WORKERS) as pool:
                args = [(speaker, audio_file, speaker_dir) for audio_file in audio_files[:int(num_files)]]
                start_time = time.time()  # Start timing
                for audio, processed_file in pool.imap_unordered(load_audio_file, args):
                    result = model.transcribe(audio, batch_size=16, language="ja", print_progress=True)
                    for segment in result["segments"]:
                        f.write(f"{processed_file}|{segment['text']}\n")
                        f.flush()
                    pf.write(processed_file + '\n')
                    pf.flush()
                end_time = time.time()  # End timing
                print(f"Transcribing audio files took {end_time - start_time} seconds\n")  # Print the time taken

                successful_transcriptions += 1

                log_file.write(f"Successfully transcribed {successful_transcriptions} files for speaker {speaker}\n")
                # Check if the number of successful transcriptions matches the number of files
                if successful_transcriptions == int(num_files):
                    log_file.write(f"All files for speaker {speaker} were successfully transcribed.\n")
                else:
                    log_file.write(f"Only {successful_transcriptions} out of {num_files} files for speaker {speaker} were successfully transcribed.\n")

def main():

    start_time = time.time()  # Start timing
    settings = load_settings(CONFIG_PATH)
    end_time = time.time()  # End timing
    print(f"Loading settings took {end_time - start_time} seconds")  # Print the time taken


    print("Getting device info")
    device, compute_type = get_device_info()
    print("Device info obtained.")

    start_time = time.time()  # Start timing
    model = whisperx.load_model("large-v2", device, compute_type=compute_type, language=settings["language"])
    end_time = time.time()  # End timing
    print(f"Loading model took {end_time - start_time} seconds")  # Print the time taken

    print('starting run_whisperx...')
    run_whisperx(DATASET_DIR, OUTPUT_DIR, settings, device, compute_type, model)
    print("Transcription complete.")

    # Free up memory
    del model
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
    print("Script complete.")

# The code only transcribe and generate the train.txt file of an already strutured dataset. The next step is to use the train.txt file to train the model.
