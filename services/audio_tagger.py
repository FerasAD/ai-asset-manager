import os
import re

# Words that add no retrieval value and are stripped out during tag generation
STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on",
    "sound", "sounds", "sfx", "audio", "track", "clip"
}


def generate_filename_tags(file_path):
    # Cleans the filename and returns a list of meaningful tags
    # e.g. epic_sword_clash_2.mp3 → ["epic", "sword", "clash"]
    filename = os.path.basename(file_path)
    name_without_extension = os.path.splitext(filename)[0].lower()

    # Replace separators with spaces, then strip anything that isn't a letter/number
    cleaned_text = re.sub(r"[_\-]+", " ", name_without_extension)
    cleaned_text = re.sub(r"[^a-z0-9\s]", " ", cleaned_text)

    raw_tokens = cleaned_text.split()

    tags = []
    for token in raw_tokens:
        if token.isdigit():       # skip pure numbers like "2" or "007"
            continue
        if len(token) <= 1:       # skip single characters
            continue
        if token in STOPWORDS:    # skip generic words
            continue
        if token not in tags:     # skip duplicates
            tags.append(token)

    return tags