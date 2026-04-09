from __future__ import annotations

from app import (
    preload_classifier_bundle,
    resolve_speaker_model_cache_dir,
    resolve_speaker_model_name,
)


def main() -> int:
    model_name = resolve_speaker_model_name()
    cache_dir = resolve_speaker_model_cache_dir()
    print(f"Preparing SpeechBrain speaker model '{model_name}' in '{cache_dir}'...")

    classifier, classifier_error = preload_classifier_bundle()

    if classifier is None:
        print(f"SpeechBrain speaker model could not be prepared: {classifier_error}")
        return 1

    print("SpeechBrain speaker model is cached and ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
