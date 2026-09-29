from pathlib import Path


def unique_path(path: Path) -> Path:
    """Devuelve `path` si está libre; si no, inserta un contador antes de la extensión.

    `clip.wav.mp3` pasa a `clip.wav.1.mp3`, `clip.wav.2.mp3`, etc.
    """
    candidate = path
    counter = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}.{counter}{path.suffix}")
        counter += 1
    return candidate
