import os
from moviepy.editor import AudioFileClip
import glob
import zipfile

path_old = "./wavs-old/"  # Input directory containing WAV files
path_new = "./wavs/"  # Output directory to save WAV files with the new sample rate
zip_file_path = "./wavs.zip"

# Create the output directory if it doesn't exist
for path in [path_old, path_new]:
    if not os.path.exists(path):
        os.makedirs(path)

# Counter for renaming files
counter = 1

# Iterate over each WAV file in the input directory
for file in glob.glob(os.path.join(path_old, "*.wav")):
    # Load the audio file
    audio_clip = AudioFileClip(file)

    # Resample the audio to 16000 Hz
    audio_resampled = audio_clip.set_fps(16000)

    # Calculate the number of segments
    num_segments = int(audio_resampled.duration / 60) + 1

    # Split the audio into 1-minute segments
    for i in range(num_segments):
        start_time = i * 60
        end_time = min((i + 1) * 60, audio_resampled.duration)
        segment = audio_resampled.subclip(start_time, end_time)

        # Generate new filename with counter
        new_filename = f"wavs_{counter}.wav"

        # Write the segment to the output directory
        output_file = os.path.join(path_new, new_filename)
        segment.write_audiofile(output_file)

        print(f"Processed {os.path.basename(file)} ({start_time}-{end_time} seconds). Saved as {new_filename}")

        # Increment the counter
        counter += 1

print("Conversion completed successfully.")

# Zip all WAV files into a single archive
with zipfile.ZipFile(zip_file_path, 'w') as zipf:
    for root, _, files in os.walk(path_new):
        for file in files:
            zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), path_new))

print(f"All WAV files zipped into {zip_file_path}.")