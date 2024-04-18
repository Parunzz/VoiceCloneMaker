import os
import torch
from transformers import pipeline
import glob

MODEL_NAME = "biodatlab/whisper-th-medium-combined"
lang = "th"

device = 0 if torch.cuda.is_available() else "cpu"

pipe = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_NAME,
    chunk_length_s=30,
    device=device,
)

path_prompts = "./th/data/"  # Directory to save the prompts.tsv file
path_new = "./wavs/"  # Output directory to save WAV files with the new sample rate
zip_file_path = "./wav.zip"
# Create the output directory if it doesn't exist
if not os.path.exists(path_prompts):
    os.makedirs(path_prompts)

# Check if prompts.tsv exists and its content is valid
prompts_file_path = os.path.join(path_prompts, "prompts.tsv")
if not os.path.exists(prompts_file_path) or os.path.getsize(prompts_file_path) == 0:
    # File doesn't exist or is empty, recreate it with the header
    with open(prompts_file_path, "w", encoding="utf-8") as prompts_file:
        prompts_file.write("")
    existing_filenames = set()  # Initialize existing_filenames as an empty set

# File exists and has content, read existing filenames from it
existing_filenames = set()
with open(os.path.join(path_prompts, "prompts.tsv"), "r", encoding="utf-8") as prompts_file:
    for line in prompts_file:
        filename, _ = line.strip().split("\t")
        existing_filenames.add(filename)

# Iterate over each WAV file in the input directory
for i, file_path in enumerate(glob.glob(os.path.join(path_new, "*.wav"))):
    filename = f"wavs_{i + 1}"  # Assign ascending filenames like wavs_1, wavs_2, ...

    # Check if filename already exists in prompts.tsv
    if filename in existing_filenames:
        print(f"Skipping transcription for {filename} as it already exists in prompts.tsv")
        continue

    print("computing this file : ",filename)   

    # Transcribe the audio file
    transcriptions = pipe(
        file_path,
        batch_size=16,
        return_timestamps=False,
        generate_kwargs={"language": "<|th|>", "task": "transcribe"}
    )["text"]
    
    # Write the output filename and transcription to the prompts.tsv file
    with open(os.path.join(path_prompts, "prompts.tsv"), "a", encoding="utf-8") as prompts_file:
        prompts_file.write(f"{filename}\t{transcriptions}\n")
    
    print(f"Transcribed {filename}: {transcriptions}")

