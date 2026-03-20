"""
Download the full GTSinger English dataset from HuggingFace.

Usage:
    python scripts/download_gtsinger.py
    python scripts/download_gtsinger.py --output data/GTSinger_English
    python scripts/download_gtsinger.py --repo GTSinger/GTSinger
    python scripts/download_gtsinger.py --token hf_yourtoken
"""

import argparse
import time
from pathlib import Path


def download_english_gtsinger(output_dir="data/GTSinger_English", repo_id="GTSinger/GTSinger", token=None):
    """
    Download the English portion of the GTSinger dataset from HuggingFace.

    Args:
        output_dir: Local directory to save the dataset.
        repo_id: HuggingFace dataset repo ID.
        token: Optional HuggingFace API token (increases rate limits).
    """
    try:
        from huggingface_hub import snapshot_download, login
    except ImportError:
        print("Error: huggingface_hub is not installed.")
        print("Install it with: pip install huggingface_hub")
        return False

    if token:
        login(token=token)
        print("Logged in to HuggingFace.\n")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading GTSinger English dataset from {repo_id}...")
    print(f"Output directory: {output_path.resolve()}")
    print("This may take a while depending on your connection speed.")
    print("Downloads will resume automatically if interrupted.\n")

    max_attempts = 10
    retry_wait = 60  # seconds between retries

    for attempt in range(1, max_attempts + 1):
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                allow_patterns=["English/**"],
                local_dir=str(output_path),
                local_dir_use_symlinks=False,  # Windows-safe: write real files
                resume_download=True,           # pick up where we left off
                max_workers=8,                  # low concurrency avoids 429s
                token=token,
            )

            print(f"\nDownload complete!")

            # Verify
            wav_files = list(output_path.rglob("*.wav"))
            json_files = list(output_path.rglob("*.json"))
            xml_files = list(output_path.rglob("*.musicxml")) + list(output_path.rglob("*.xml"))

            print(f"  WAV files:      {len(wav_files)}")
            print(f"  JSON files:     {len(json_files)}")
            print(f"  MusicXML files: {len(xml_files)}")

            return True

        except Exception as e:
            print(f"\n[Attempt {attempt}/{max_attempts}] Download interrupted: {e}")
            if attempt < max_attempts:
                print(f"Resuming in {retry_wait}s... (progress is saved)\n")
                time.sleep(retry_wait)
            else:
                print("All attempts exhausted. Run the script again to resume.")
                return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download GTSinger English dataset")
    parser.add_argument('--output', default='data/GTSinger_English',
                        help='Output directory (default: data/GTSinger_English)')
    parser.add_argument('--repo', default='GTSinger/GTSinger',
                        help='HuggingFace repo ID (default: GTSinger/GTSinger)')
    parser.add_argument('--token', default=None,
                        help='HuggingFace API token (recommended — raises rate limits)')
    args = parser.parse_args()

    # Resolve relative to the emotion_pipelines directory
    base = Path(__file__).parent.parent
    output = base / args.output

    download_english_gtsinger(str(output), args.repo, args.token)