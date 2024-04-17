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

path_prompts = "./prompts/"  # Directory to save the prompts.tsv file
path_new = "./wavs/"  # Output directory to save WAV files with the new sample rate

# Create the output directory if it doesn't exist
if not os.path.exists(path_prompts):
    os.makedirs(path_prompts)

# Read existing filenames from prompts.tsv
existing_filenames = set()
with open(os.path.join(path_prompts, "prompts.tsv"), "r", encoding="utf-8") as prompts_file:
    for line in prompts_file:
        filename, _ = line.strip().split("\t")
        existing_filenames.add(filename)

# Iterate over each WAV file in the input directory
for file_path in glob.glob(os.path.join(path_new, "*.wav")):
    filename = os.path.basename(file_path)

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
