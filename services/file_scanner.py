import os

SUPPORTED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg"}


def scan_audio_files(folder_path):
    audio_files = []

    for root, _, files in os.walk(folder_path):
        for file_name in files:
            _, extension = os.path.splitext(file_name)
            if extension.lower() in SUPPORTED_AUDIO_EXTENSIONS:
                full_path = os.path.join(root, file_name)
                audio_files.append(full_path)

    return sorted(audio_files)