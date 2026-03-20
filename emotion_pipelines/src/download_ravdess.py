"""
Download RAVDESS dataset from Kaggle and organize into correct structure.
"""
import os
import shutil
from pathlib import Path

try:
    import kagglehub
except ImportError:
    print("kagglehub not installed. Installing now...")
    import subprocess
    subprocess.check_call(["pip", "install", "kagglehub"])
    import kagglehub

# Target directory for RAVDESS
PROJECT_ROOT = Path(__file__).parent.parent
RAVDESS_TARGET = PROJECT_ROOT / "data" / "raw" / "archive (1)"

def download_ravdess():
    """Download RAVDESS dataset from Kaggle."""
    print("=" * 60)
    print("DOWNLOADING RAVDESS DATASET FROM KAGGLE")
    print("=" * 60)
    
    # Download latest version
    print("\nDownloading dataset...")
    try:
        path = kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-song-audio")
        print(f"✓ Downloaded to: {path}")
    except Exception as e:
        print(f"✗ Download failed: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure you have a Kaggle account")
        print("2. Set up Kaggle API credentials:")
        print("   - Go to https://www.kaggle.com/settings")
        print("   - Click 'Create New API Token'")
        print("   - Place kaggle.json in: C:\\Users\\<username>\\.kaggle\\")
        return None
    
    return Path(path)

def organize_dataset(source_path, target_path):
    """Organize downloaded dataset into project structure."""
    print(f"\nOrganizing dataset...")
    print(f"Source: {source_path}")
    print(f"Target: {target_path}")
    
    # Create target directory
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Find all .wav files in source
    wav_files = list(source_path.rglob("*.wav"))
    print(f"\nFound {len(wav_files)} audio files")
    
    if len(wav_files) == 0:
        print("✗ No .wav files found in downloaded dataset")
        print(f"Contents of {source_path}:")
        for item in source_path.iterdir():
            print(f"  - {item}")
        return False
    
    # Copy files organized by actor
    actors_processed = set()
    for wav_file in wav_files:
        # RAVDESS filename format: 03-01-06-01-02-01-12.wav
        # Last two digits = actor number
        try:
            parts = wav_file.stem.split('-')
            if len(parts) >= 7:
                actor_num = parts[-1]
                actor_folder = target_path / f"Actor_{actor_num}"
                actor_folder.mkdir(exist_ok=True)
                
                # Copy file
                dest_file = actor_folder / wav_file.name
                if not dest_file.exists():
                    shutil.copy2(wav_file, dest_file)
                    actors_processed.add(actor_num)
        except Exception as e:
            print(f"Warning: Could not process {wav_file.name}: {e}")
    
    print(f"\n✓ Organized {len(wav_files)} files into {len(actors_processed)} actor folders")
    print(f"✓ Dataset ready at: {target_path}")
    
    return True

def verify_dataset(target_path):
    """Verify dataset structure and count files."""
    print("\n" + "=" * 60)
    print("VERIFYING DATASET")
    print("=" * 60)
    
    if not target_path.exists():
        print(f"✗ Target path does not exist: {target_path}")
        return False
    
    # Count actor folders
    actor_folders = sorted([d for d in target_path.iterdir() if d.is_dir() and d.name.startswith("Actor_")])
    print(f"\nActor folders found: {len(actor_folders)}")
    
    # Count total audio files
    total_files = len(list(target_path.rglob("*.wav")))
    print(f"Total audio files: {total_files}")
    
    # Show sample structure
    if len(actor_folders) > 0:
        print(f"\nSample structure:")
        for actor in actor_folders[:3]:
            files = list(actor.glob("*.wav"))
            print(f"  {actor.name}/")
            for f in files[:3]:
                print(f"    - {f.name}")
            if len(files) > 3:
                print(f"    ... and {len(files) - 3} more files")
        if len(actor_folders) > 3:
            print(f"  ... and {len(actor_folders) - 3} more actor folders")
    
    print("\n✓ Dataset verification complete")
    print(f"✓ Ready to train emotion classifier!")
    
    return True

def main():
    print("RAVDESS Dataset Setup")
    print("This script will download and organize the RAVDESS dataset.\n")
    
    # Check if already exists
    if RAVDESS_TARGET.exists():
        existing_files = len(list(RAVDESS_TARGET.rglob("*.wav")))
        if existing_files > 0:
            print(f"✓ RAVDESS dataset already exists at: {RAVDESS_TARGET}")
            print(f"  Found {existing_files} audio files")
            response = input("\nDo you want to re-download? (y/N): ").strip().lower()
            if response != 'y':
                print("Using existing dataset.")
                verify_dataset(RAVDESS_TARGET)
                return
            else:
                print("Removing existing dataset...")
                shutil.rmtree(RAVDESS_TARGET)
    
    # Download
    source_path = download_ravdess()
    if source_path is None:
        return
    
    # Organize
    success = organize_dataset(source_path, RAVDESS_TARGET)
    if not success:
        return
    
    # Verify
    verify_dataset(RAVDESS_TARGET)
    
    print("\n" + "=" * 60)
    print("SETUP COMPLETE!")
    print("=" * 60)
    print(f"\nDataset location: {RAVDESS_TARGET}")
    print("\nNext steps:")
    print("1. Run the emotion classification pipeline:")
    print("   python src\\run_emotion_pipeline.py")
    print("\n2. Or train the model separately:")
    print("   python src\\emotion_classification\\train_emotion_classifier.py")

if __name__ == "__main__":
    main()
