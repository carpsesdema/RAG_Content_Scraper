import os
import re
try:
    from config import DEFAULT_SNIPPET_FILENAME_SLUG_MAX_LENGTH
except ImportError:
    DEFAULT_SNIPPET_FILENAME_SLUG_MAX_LENGTH = 50


def _slugify(text: str, max_length: int = None) -> str:
    """
    Generates a filesystem-safe slug from the first line of the text.
    """
    actual_max_length = max_length if max_length is not None else DEFAULT_SNIPPET_FILENAME_SLUG_MAX_LENGTH
    # Get first non-empty line
    first_line = text.strip().splitlines()[0] if text.strip() else "snippet"
    # Replace non-word characters with underscores
    slug = re.sub(r"\W+", "_", first_line)
    # Trim leading/trailing underscores and lowercase
    slug = slug.strip("_").lower()
    return slug[:actual_max_length] or "snippet"

def save_snippets(snippets, directory):
    """
    Saves each snippet into a separately named .txt file in the given directory.
    Filenames are prefixed with a zero-padded index and a slug of the snippet's first line.
    """
    os.makedirs(directory, exist_ok=True)

    for idx, snippet in enumerate(snippets, start=1):
        slug = _slugify(snippet) # Uses the configured max_length by default
        base_name = f"{idx:03d}_{slug}"
        filename = os.path.join(directory, f"{base_name}.txt")

        # If a collision occurs, append a counter
        counter = 1
        while os.path.exists(filename):
            filename = os.path.join(directory, f"{base_name}_{counter}.txt")
            counter += 1

        with open(filename, "w", encoding="utf-8") as f:
            f.write(snippet)