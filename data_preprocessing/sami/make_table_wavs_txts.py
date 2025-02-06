import os

# Define the path to the folder containing the .wav and .txt files
folder_path = "/home/hiovain/DATASETS/MaijuSaijetsProcess/splits"

# Open the output file in the same directory as folder_path
output_file_path = "/home/hiovain/DATASETS/MaijuSaijetsProcess/splits/Maiju_splits_table.txt"
with open(output_file_path, 'w') as outfile:
    # Get a sorted list of all files in the folder
    sorted_filenames = sorted(os.listdir(folder_path))
    
    # Loop through all files in the folder, now sorted alphabetically
    for filename in sorted_filenames:
        # Check if the file is a .wav file
        if filename.endswith(".wav"):
            # Create the full file path
            wav_file_path = os.path.join(folder_path, filename)

            # Create the corresponding .txt file path
            txt_file_path = os.path.join(folder_path, filename.replace('.wav', '.txt'))

            # Check if the .txt file exists
            if os.path.exists(txt_file_path):
                # Read the text from the .txt file
                with open(txt_file_path, 'r') as txt_file:
                    text = txt_file.read().strip()

                # Write the .wav filename and the text to the output file
                outfile.write(f"{filename}|{text}\n")
