import os
from mutagen import File as MutagenFile


def get_audio_metadata(file_path):
    # Returns (duration, file_size_mb) — both stored as columns in the assets table
    duration = 0.0
    file_size_mb = 0.0

    try:
        audio_file = MutagenFile(file_path)
        if audio_file is not None and audio_file.info is not None:
            duration = round(audio_file.info.length, 2)
    except Exception:
        duration = 0.0

    try:
        file_size_bytes = os.path.getsize(file_path)
        file_size_mb = round(file_size_bytes / (1024 * 1024), 2)
    except Exception:
        file_size_mb = 0.0

    return duration, file_size_mb