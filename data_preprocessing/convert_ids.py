import sys
# Read the data file
with open(sys.argv[1], "r") as file:
    lines = file.readlines()

# Extract speaker_ids and language_ids
speaker_ids = set()
language_ids = set()
for line in lines:
    try:
        _, _, _, _, speaker_id, language_id = line.strip().split("|")
    except:
        pass
    speaker_ids.add(speaker_id)
    language_ids.add(language_id)

# Create mappings
speaker_mapping = {speaker: idx for idx, speaker in enumerate(sorted(speaker_ids))}
language_mapping = {language: idx for idx, language in enumerate(sorted(language_ids))}


# Write mappings to a file
with open("mappings.txt", "w") as mapping_file:
    mapping_file.write("Speaker Mapping:\n")
    for speaker, idx in speaker_mapping.items():
        mapping_file.write(f"{speaker}: {idx}\n")
    mapping_file.write("\nLanguage Mapping:\n")
    for language, idx in language_mapping.items():
        mapping_file.write(f"{language}: {idx}\n")

        
# Convert data
converted_lines = []
for line in lines:
    wavname, pitchfile, text, speaker_num, speaker_id, language_id = line.strip().split("|")
    numerical_speaker = speaker_mapping[speaker_id]
    numerical_language = language_mapping[language_id]
    converted_line = f"{wavname}|{pitchfile}|{text}|{speaker_mapping[speaker_id]}|{language_mapping[language_id]}"
    print(converted_line)

