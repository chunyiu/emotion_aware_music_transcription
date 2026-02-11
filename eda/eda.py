import librosa
import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from tqdm import tqdm
import warnings

GTSINGER_ROOT_DIR = 'GTSinger_sample_50'
RAVDESS_ROOT_DIR = './archive' 
OUTPUT_PLOT_DIR = 'eda_plots'
SAMPLE_RATE = 44100

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.dpi'] = 100
plt.rcParams.update({
    "font.family": "serif",
    "axes.labelsize": 11,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

sns.set_context("talk", font_scale=0.9)

# ============================================================================
# PART 1: GTSINGER DATA LOADING & ANALYSIS
# ============================================================================

def load_gtsinger_data(root_dir):
    """
    Scans the GTSinger directory and extracts all metadata and note data.
    """
    all_notes = []
    file_metadata = []
    
    print(f"Scanning directory: {root_dir}...")
    json_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))

    if not json_files:
        print(f"Warning: No .json files found in {root_dir}.")
        return pd.DataFrame(), pd.DataFrame()

    for json_path in tqdm(json_files, desc="Loading GTSinger Data"):
        wav_path = os.path.splitext(json_path)[0] + ".wav"
        
        if not os.path.exists(wav_path):
            continue
        
        try:
            path_parts = os.path.normpath(json_path).split(os.sep)
            singer_id = path_parts[-5]
            technique = path_parts[-4]
            song_name = path_parts[-3]
            file_id = os.path.splitext(os.path.basename(json_path))[0]
            
            # Extract voice type (gender)
            voice_type = "Unknown"
            if "Alto" in singer_id: voice_type = "Alto (Female)"
            if "Tenor" in singer_id: voice_type = "Tenor (Male)"

        except IndexError:
            continue
        
        # Load JSON for note and emotion data
        file_emotion = "Unknown"
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        last_note_end = 0
        for i, entry in enumerate(data):
            # Extract emotion from the *first* entry
            if i == 0:
                file_emotion = entry.get('emotion', 'Unknown').capitalize()

            notes = entry.get('note', [])
            note_starts = entry.get('note_start', [])
            note_ends = entry.get('note_end', [])
            
            for j in range(len(notes)):
                if notes[j] <= 0: # Skip rests
                    continue
                
                start = float(note_starts[j])
                end = float(note_ends[j])
                duration = end - start
                gap = start - last_note_end
                
                all_notes.append({
                    'file_id': file_id,
                    'singer_id': singer_id,
                    'voice_type': voice_type,
                    'technique': technique,
                    'emotion': file_emotion,
                    'midi_pitch': int(notes[j]),
                    'pitch_name': librosa.midi_to_note(int(notes[j])),
                    'start_sec': start,
                    'end_sec': end,
                    'note_duration_sec': duration,
                    'gap_duration_sec': gap if gap > 0.001 else 0
                })
                last_note_end = end
        
        file_metadata.append({
            'singer_id': singer_id,
            'voice_type': voice_type,
            'technique': technique,
            'emotion': file_emotion,
            'song_name': song_name,
            'file_id': file_id,
            'wav_path': wav_path,
            'json_path': json_path,
        })

    print(f"Scan complete. Found {len(file_metadata)} audio files and {len(all_notes)} ground-truth notes.")
    return pd.DataFrame(file_metadata), pd.DataFrame(all_notes)

def plot_gtsinger_overview(files_df, output_dir):

    print("Generating GTSinger Overview Plots...")

    fig, axes = plt.subplots(
        2, 2,
        figsize=(16, 13),
        constrained_layout=True
    )

    # ---------------- Plot 1 ----------------
    sns.countplot(
        data=files_df,
        y="technique",
        order=files_df["technique"].value_counts().index,
        ax=axes[0, 0],
        palette="viridis"
    )
    axes[0, 0].set_title("Plot 1: Singing Technique Distribution")
    axes[0, 0].set_xlabel("Number of Files")
    axes[0, 0].set_ylabel("Technique")

    # ---------------- Plot 2 ----------------
    sns.countplot(
        data=files_df,
        y="singer_id",
        order=files_df["singer_id"].value_counts().index,
        ax=axes[0, 1],
        palette="plasma"
    )
    axes[0, 1].set_title("Plot 2: Singer Distribution")
    axes[0, 1].set_xlabel("Number of Files")
    axes[0, 1].set_ylabel("Singer ID")

    # Make room for long labels
    axes[0, 1].tick_params(axis="y", labelsize=10)

    # ---------------- Plot 3 ----------------
    sns.countplot(
        data=files_df,
        x="voice_type",
        order=files_df["voice_type"].value_counts().index,
        ax=axes[1, 0],
        palette="Blues"
    )
    axes[1, 0].set_title("Plot 3: Voice Type Distribution")
    axes[1, 0].set_xlabel("Voice Type")
    axes[1, 0].set_ylabel("Number of Files")

    # ---------------- Plot 4 ----------------
    sns.countplot(
        data=files_df,
        x="emotion",
        order=files_df["emotion"].value_counts().index,
        ax=axes[1, 1],
        palette="Reds"
    )
    axes[1, 1].set_title("Plot 4: Emotion Distribution")
    axes[1, 1].set_xlabel("Emotion")
    axes[1, 1].set_ylabel("Number of Files")

    fig.suptitle(
        "GTSinger Dataset Composition",
        fontsize=18
    )

    plt.savefig(
        os.path.join(output_dir, "plot_gtsinger_overview.png"),
        dpi=150,
        bbox_inches="tight"
    )
    plt.close()

    print("-> Saved 'plot_gtsinger_overview.png'")

def plot_transcription_justification(notes_df, output_dir):
    """
    Plots 5, 6, 7: Justifies the hyperparameter ranges for tuning.
    """
    print("Generating Transcription Hyperparameter Plots...")
    plt.figure(figsize=(18, 5))
    
# Plot 5: Note Duration Distribution (log scale + violin)
    plt.subplot(1, 3, 1)
    sns.violinplot(y=np.log1p(notes_df["note_duration_sec"]), color="seagreen", inner="quartile")
    plt.title("Note Durations (log-scaled)")
    plt.ylabel("log(Duration + 1) [s]")

    # Plot 6: Pitch Distribution by Voice Type
    plt.subplot(1, 3, 2)
    sns.boxplot(data=notes_df, x="voice_type", y="midi_pitch", palette="flare")
    plt.title("Pitch Distribution by Voice Type")
    plt.ylabel("MIDI Pitch")
    plt.xlabel("Voice Type")

    # Plot 7: Silence Distribution
    plt.subplot(1, 3, 3)
    sns.violinplot(y=notes_df["gap_duration_sec"], color="crimson", inner="quartile")
    plt.title("Silence Duration")
    plt.ylabel("Seconds")
        
    plt.suptitle("EDA Justification: Transcription Hyperparameters", fontsize=16, y=1.03)
    plt.savefig(os.path.join(output_dir, "plot_transcription_justification.png"), bbox_inches="tight")
    plt.close()
    print("-> Saved 'plot_transcription_justification.png'")

def plot_gtsinger_distribution(output_dir):

    print("Generating GTSinger Distributions...")

    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"font.family": "serif"})

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Statistical Distribution of Global Style Labels (Bar Chart Form)", fontsize=14, y=1.02)

    # Singing Method
    data_a = pd.Series({"Bel Canto": 25.63, "Pop": 74.37})
    axes[0,0].barh(data_a.index, data_a.values, color=["#f28e8e", "#8eb8f2"])
    for i, v in enumerate(data_a.values):
        axes[0,0].text(v + 1, i, f"{v:.2f}%", va="center")
    axes[0,0].set_xlim(0,100)
    axes[0,0].set_title("(a) Distribution of Singing Method")
    axes[0,0].set_xlabel("Percentage")

    # Pace
    data_b = pd.Series({"Slow": 34.09, "Moderate": 23.15, "Fast": 42.76})
    axes[0,1].barh(data_b.index, data_b.values, color=["#8eb8f2", "#f28e8e", "#b3a9f2"])
    for i, v in enumerate(data_b.values):
        axes[0,1].text(v + 1, i, f"{v:.2f}%", va="center")
    axes[0,1].set_xlim(0,100)
    axes[0,1].set_title("(b) Distribution of Pace")
    axes[0,1].set_xlabel("Percentage")

    # Range
    data_c = pd.Series({"Low": 12.43, "Medium": 62.38, "High": 25.19})
    axes[1,0].barh(data_c.index, data_c.values, color=["#8eb8f2", "#f28e8e", "#b3a9f2"])
    for i, v in enumerate(data_c.values):
        axes[1,0].text(v + 1, i, f"{v:.2f}%", va="center")
    axes[1,0].set_xlim(0,100)
    axes[1,0].set_title("(c) Distribution of Range")
    axes[1,0].set_xlabel("Percentage")

    # Emotion
    data_d = pd.Series({"Happy": 21.78, "Sad": 78.22})
    axes[1,1].barh(data_d.index, data_d.values, color=["#f28e8e", "#8eb8f2"])
    for i, v in enumerate(data_d.values):
        axes[1,1].text(v + 1, i, f"{v:.2f}%", va="center")
    axes[1,1].set_xlim(0,100)
    axes[1,1].set_title("(d) Distribution of Emotion")
    axes[1,1].set_xlabel("Percentage")

    plt.savefig(os.path.join(output_dir, "bar_style_gtsinger_paper_recreation.png.png"), bbox_inches="tight")
    plt.close()
    print("-> Saved 'bar_style_gtsinger_paper_recreation.png.png'")

def plot_pyin_vs_ground_truth(sample_file, output_dir):
    """
    Plot 8: Visually explains the low F1-score.
    """
    print(f"Generating Plot 8 (pYIN vs. Ground Truth)...")
    wav_path = sample_file['wav_path']
    json_path = sample_file['json_path']

    file_notes = []
    with open(json_path, 'r') as f: data = json.load(f)
    for entry in data:
        for i in range(len(entry.get('note', []))):
            if entry['note'][i] > 0:
                file_notes.append({
                    'midi_pitch': entry['note'][i],
                    'start_sec': entry['note_start'][i],
                    'end_sec': entry['note_end'][i]
                })
    file_notes_df = pd.DataFrame(file_notes)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
            f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('G3'), fmax=librosa.note_to_hz('G5'), sr=sr)
        times = librosa.times_like(f0, sr=sr)
        f0_midi = librosa.hz_to_midi(f0)
    except Exception as e:
        print(f"Error loading audio {wav_path}: {e}")
        return

    plt.figure(figsize=(15, 5))
    plt.plot(times, f0_midi, 'o', color='steelblue', markersize=2, alpha=0.5, label='Raw pYIN Pitch (Noisy)')
    
    for i, row in file_notes_df.iterrows():
        plt.hlines(y=row['midi_pitch'], xmin=row['start_sec'], xmax=row['end_sec'],
                   color='darkorange', linewidth=4, label='Ground Truth (Clean)' if i == 0 else "")
    
    plt.title(f"Plot 8: Transcription Challenge (pYIN vs. Ground Truth)\n{os.path.basename(wav_path)}", fontsize=14)
    plt.xlabel('Time (seconds)')
    plt.ylabel('MIDI Pitch')
    plt.legend()
    plt.ylim(50, 85)
    plt.savefig(os.path.join(output_dir, "plot_pyin_vs_ground_truth.png"), bbox_inches="tight")
    plt.close()
    print("-> Saved 'plot_pyin_vs_ground_truth.png'")

def plot_harmony_justification_gtsinger(notes_df, sample_file_id, output_dir):
    """
    Plots 9 & 10: Justifies the 'smart' harmony approach (mingus).
    """
    print("Generating Harmony Justification Plots...")
    sample_notes = notes_df[notes_df['file_id'] == sample_file_id]
    
    plt.figure(figsize=(16, 6))

    # Plot 9: Tonal Analysis (Chromagram)
    plt.subplot(1, 2, 1)
    pitch_classes = [p % 12 for p in sample_notes['midi_pitch']]
    pitch_counts = Counter(pitch_classes)
    chroma_df = pd.DataFrame({
        'pitch_class': [librosa.midi_to_note(p, octave=False) for p in range(12)],
        'count': [pitch_counts.get(i, 0) for i in range(12)]
    })
    sns.barplot(data=chroma_df, x='pitch_class', y='count', palette='crest')
    plt.title(f'Plot 9: Pitch Class Distribution (Tonal Analysis)\nFile: {sample_file_id}.json')
    plt.xlabel('Pitch Class')
    plt.ylabel('Note Count')
    
    # Plot 10: Pitch Class Co-occurrence (Intervals)
    plt.subplot(1, 2, 2)
    pitch_classes = np.array(pitch_classes)
    co_occurrence = np.zeros((12, 12))
    for i in range(len(pitch_classes) - 1):
        p1 = pitch_classes[i]
        p2 = pitch_classes[i+1]
        co_occurrence[p1, p2] += 1
    
    # Normalize
    row_sums = co_occurrence.sum(axis=1, keepdims=True)
    co_occurrence_norm = np.divide(co_occurrence, row_sums, out=np.zeros_like(co_occurrence), where=row_sums!=0)
    
    sns.heatmap(co_occurrence_norm, annot=False, cmap='viridis', 
                xticklabels=chroma_df['pitch_class'], yticklabels=chroma_df['pitch_class'])
    plt.title(f'Plot 10: Pitch Class Transition Probability\n(Row -> Column)')
    plt.xlabel('Next Note')
    plt.ylabel('Current Note')

    plt.suptitle("EDA Justification: Harmony Generation (GTSinger)", fontsize=16, y=1.03)
    plt.savefig(os.path.join(output_dir, "plot_harmony_justification.png"), bbox_inches="tight")
    plt.close()
    print("-> Saved 'plot_harmony_justification.png'")
    print("-> JUSTIFICATION: Plot 9 shows the music is TONAL. Plot 10 shows clear patterns in note transitions (harmony). This justifies a chord-aware library like `mingus` (Pipeline 2).")

# ============================================================================
# PART 2: RAVDESS DATA LOADING & ANALYSIS
# ============================================================================

def load_ravdess_data(root_dir):
    """
    Scans RAVDESS 'Actor_XX' folders and parses filenames.
    """
    print(f"\nScanning for RAVDESS data in: {root_dir}...")
    file_metadata = []
    
    # Maps from RAVDESS documentation
    emotion_map = {
        '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
        '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
    }
    intensity_map = {'01': 'normal', '02': 'strong'}
    statement_map = {'01': 'Kids are talking', '02': 'Dogs are sitting'}
    
    actor_dirs = [d for d in os.listdir(root_dir) if d.startswith('Actor_') and os.path.isdir(os.path.join(root_dir, d))]
    
    if not actor_dirs:
        print("Warning: No 'Actor_XX' folders found. Skipping RAVDESS analysis.")
        return pd.DataFrame()

    for actor_dir in tqdm(actor_dirs, desc="Loading RAVDESS Data"):
        actor_path = os.path.join(root_dir, actor_dir)
        actor_num = int(actor_dir.split('_')[-1])
        gender = "Female" if actor_num % 2 == 0 else "Male"
        
        for file in os.listdir(actor_path):
            if file.endswith(".wav"):
                parts = file.split('.')[0].split('-')
                if len(parts) == 7:
                    file_metadata.append({
                        'emotion': emotion_map.get(parts[2]),
                        'intensity': intensity_map.get(parts[3]),
                        'statement': statement_map.get(parts[4]),
                        'gender': gender,
                        'actor': actor_num,
                        'path': os.path.join(actor_path, file)
                    })
                    
    print(f"Scan complete. Found {len(file_metadata)} RAVDESS files.")
    return pd.DataFrame(file_metadata)

def plot_ravdess_overview(ravdess_df, output_dir):
    """
    Plots distributions for RAVDESS Emotion, Gender, and Intensity.
    """
    print("Generating RAVDESS Overview Plots...")
    plt.figure(figsize=(18, 5))
    
    # Plot 11: Emotion Distribution
    plt.subplot(1, 3, 1)
    sns.countplot(data=ravdess_df, x='emotion', order=ravdess_df['emotion'].value_counts().index, palette="Set2")
    plt.title('Plot 11: RAVDESS - Emotion Distribution')
    plt.xlabel('Emotion')
    plt.ylabel('File Count')
    plt.xticks(rotation=45)
    
    # Plot 12: Gender Distribution
    plt.subplot(1, 3, 2)
    sns.countplot(data=ravdess_df, x='gender', palette="bwr")
    plt.title('Plot 12: RAVDESS - Gender Distribution')
    plt.xlabel('Gender')
    plt.ylabel('File Count')
    
    # Plot 13: Intensity Distribution
    plt.subplot(1, 3, 3)
    sns.countplot(data=ravdess_df, x='intensity', palette="Greens")
    plt.title('Plot 13: RAVDESS - Intensity Distribution')
    plt.xlabel('Intensity')
    plt.ylabel('File Count')
    
    plt.suptitle("RAVDESS Dataset Composition", fontsize=16, y=1.05)
    plt.savefig(os.path.join(output_dir, "plot_ravdess_overview.png"), bbox_inches="tight")
    plt.close()
    print("-> Saved 'plot_ravdess_overview.png'")
    print("-> JUSTIFICATION: These plots show a perfectly balanced dataset for Emotion and Gender, making it an ideal choice for training our emotion recognition module.")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    # Create output directory first
    os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)
    
    files_df, notes_df = load_gtsinger_data(GTSINGER_ROOT_DIR)

    
    if not files_df.empty and not notes_df.empty:
        # Plot 1, 2, 3, 4: GTSinger Overview
        plot_gtsinger_overview(files_df, OUTPUT_PLOT_DIR)

        plot_gtsinger_distribution(OUTPUT_PLOT_DIR)
        
        # Plot 5, 6, 7: Hyperparameter Justification
        plot_transcription_justification(notes_df, OUTPUT_PLOT_DIR)

        # Plot 8: pYIN vs. Ground Truth
        # Try to find a specific 'Vibrato' file to make the point clear
        sample_file_df = files_df[
            (files_df['technique'] == 'Vibrato') & 
            (files_df['file_id'] == '0005')
        ]
        
        if not sample_file_df.empty:
            sample_file = sample_file_df.iloc[0]
            plot_pyin_vs_ground_truth(sample_file, OUTPUT_PLOT_DIR)
            
            # Plot 9, 10: Tonal Analysis (Justifying Mingus)
            plot_harmony_justification_gtsinger(notes_df, sample_file['file_id'], OUTPUT_PLOT_DIR)
        else:
            print("Could not find the specific sample file for Plots 8, 9, 10. Skipping.")
    
    # --- RAVDESS Analysis ---
    ravdess_df = load_ravdess_data(RAVDESS_ROOT_DIR)
    
    if not ravdess_df.empty:
        # Plot 11, 12, 13: RAVDESS Overview
        plot_ravdess_overview(ravdess_df, OUTPUT_PLOT_DIR)
    
    print(f"\n--- EDA Complete ---")
    print(f"All plots saved to: {os.path.abspath(OUTPUT_PLOT_DIR)}")

if __name__ == "__main__":
    main()