# Audio Classifier & M3U Playlist Generator

---

## Audio Classifier

### Overview

A Python command-line tool for classifying `.wav` and `.flac` audio files using rule-based audio feature analysis. It generates a CSV report with classification results.

---

### Features

- 🎧 **Audio Feature Extraction**: Uses `librosa` to extract tempo, RMS energy, spectral centroid, etc.
- 📊 **Rule-Based Classification**: Matches extracted features against predefined rules (e.g., "Happy" for 120-140 BPM).
- ⚡ **Parallel Processing**: Utilizes `ProcessPoolExecutor` for fast batch analysis.
- 📄 **CSV Output**: Generates a structured report with feature scores and matched labels.

---

### Requirements

- **Python 3.6+**
- **Libraries**:
  ```bash
  pip install numpy librosa
  ```

---

### Usage

```bash
python3 eval_to_csv.py --directory <path> --output <file> --min-score <value>
```

**Example**:
```bash
python3 eval_to_csv.py --directory ./music --output audio_index.csv --min-score 0.7
```

**Arguments**:
- `--directory`: Audio file directory (default: current directory).
- `--output`: Output CSV file (default: `audio_index.csv`).
- `--min-score`: Minimum score threshold for label matches (default: 0.66).

---

### Output CSV Structure

| Column               | Description                              |
|----------------------|------------------------------------------|
| `file_path`          | Full path to the audio file              |
| `best_label`         | Highest-scoring label                    |
| `best_score`         | Score for `best_label`                   |
| `error`              | Error message (empty if no error)        |
| `<label>_score`      | Match score for each label               |
| `<label>_passed`     | Number of rules passed for the label     |
| `<label>_total`      | Total rules for the label                |
| `<label>_matched`    | 1 if score ≥ threshold, else 0           |

---

## M3U Playlist Generator

### Overview

Generates `.m3u` playlists from an audio CSV index, supporting device-specific path transformations (e.g., HiBy, Shanling, generic drives).

---

### Features

- 📁 Detects labels from CSV headers (`<label>_matched`, `best_label`, etc.).
- 🎧 Creates playlists for matched labels, best labels, or highest scores.
- 🔄 Supports path transformations for different devices.

---

### Usage

1. **Prepare CSV**:
   Ensure it contains columns like `file_path`, `<label>_matched`, `<label>_score`.

2. **Run Script**:
   ```bash
   python3 csv_to_m3u.py <csv_file> --output-dir <dir> --base-path <prefix>
   ```

3. **Variants**:
   - `varianta`: HiBy devices (`a:\Music\`)
   - `variantA`: Shanling devices (`A:\Music\`)
   - `variantc`: Generic drives (`c:\Music\`)

---

### Example

**Input CSV (`audio_index.csv`)**:
```csv
file_path,genre_matched,artist_matched,best_label,genre_score,artist_score
/music/song1.flac,1,0,genre,0.9,0.8
/music/song2.wav,0,1,artist,0.7,0.9
```

**Command**:
```bash
python3 csv_to_m3u.py audio_index.csv --output-dir playlists --base-path /music/
```

**Output Structure**:
```
playlists/
├── varianta/
│   ├── genre.m3u
│   └── artist.m3u
├── variantA/
│   ├── genre.m3u
│   └── artist.m3u
└── variantc/
    ├── genre.m3u
    └── artist.m3u
```

---

### Notes

- Skips rows with errors.
- Falls back to highest score if no labels match.
- Normalizes file paths and applies variant-specific prefixes.

---

### m3u setup

On Hiby non-Android devices:
- Create a directory `playlist_data` and place m3u files inside
- On the device in Music > Folder Playlist icon > Playlists, select option "Load playlist" (the operation takes time depending on the size of the files)

On Shanling non-Android devices:
- Create a directory `_explaylist_data` and place m3u files inside
- On the device in Playlist click on "Import Playlist" button (the operation takes time depending on the size of the files)

TODO on hibyMusic and Poweramp on Android devices.

---

### License

MIT License. See [LICENSE](LICENSE).

