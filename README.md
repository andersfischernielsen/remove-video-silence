# Video Silence Removal Tool (Smart Speed)

A command-line tool that automatically detects and removes silence from video files using FFmpeg.

## Features

- ✨ **Automatic silence detection** using FFmpeg's `silencedetect` filter
- 🎯 **Configurable thresholds** for noise level and minimum silence duration
- 🎬 **Progressive output** - watch your video while it's being processed (MP4 format)

## Requirements

- **Python 3.8+**
- **FFmpeg** (must be installed and available in `PATH`)

## Usage

The script uses `uv` for dependency management, so you can run it directly:

```sh
# Basic usage
uv run remove-silence.py -i input.mp4
./remove-silence.py -i input.mp4

# Custom threshold and duration
uv run remove-silence.py -i input.mp4 -t -35dB -d 0.5
./remove-silence.py -i input.mp4 -t -35dB -d 0.5

# Specify output file
uv run remove-silence.py -i input.mp4 -o cleaned_video.mp4
./remove-silence.py -i input.mp4 -o cleaned_video.mp4

# Enable progressive output (watch while processing)
uv run remove-silence.py -i input.mp4 -o output.mp4 --progressive
./remove-silence.py -i input.mp4 -o output.mp4 --progressive

# Show help
uv run remove-silence.py --help
./remove-silence.py --help
```

### Command Line Options

| Option            | Description                                        | Default  |
| ----------------- | -------------------------------------------------- | -------- |
| `-t, --threshold` | Noise threshold (e.g., -30dB)                      | -30dB    |
| `-d, --duration`  | Minimum silence duration to cut (seconds)          | 0.1      |
| `-i, --input`     | Input video file (required)                        | -        |
| `-o, --output`    | Output file name (required)                        | -        |
| `--progressive`   | Enable progressive output (watch while processing) | disabled |

### Examples

```bash
# Remove silences quieter than -30dB lasting 0.1+ seconds
./remove-silence.py -i lecture.mp4

# More aggressive: remove silences quieter than -25dB lasting 0.5+ seconds
./remove-silence.py -i podcast.mp4 -t -25dB -d 0.5

# Process a webm file with custom output name
./remove-silence.py -i recording.webm -o clean_recording.mp4

# Enable progressive output for long videos (watch while processing)
./remove-silence.py -i long_lecture.mp4 -o processed_lecture.mp4 --progressive
```

## Implementation

1. **Silence Detection**: Uses FFmpeg's `silencedetect` filter to find silent segments
2. **Interval Building**: Creates a list of non-silent segments to keep
3. **Video Processing**: Uses FFmpeg's `filter_complex` to concatenate kept segments
4. **Progress Tracking**: Parses FFmpeg output to show real-time progress

## Output

The tool provides feedback including:

- Original and final video duration
- Number of silence periods found
- Time saved and percentage reduction
- File size of the output
- Processing statistics

Example output:

```
Processing: input.mp4
Threshold: -30dB, Min duration: 0.1s

   Original duration: 5m 23.2s
   Estimated final duration: 3m 45.1s
   Estimated time saved: 1m 38.1s (30.2%)

Success!
Output file: output.mp4
File size: 45.2MB
Segments processed: 23
Original duration: 5m 23.2s
Final duration: 3m 45.1s
Time saved: 1m 38.1s (30.2%)
```

## Troubleshooting

**"FFmpeg not found"**: Make sure FFmpeg is installed and available in your PATH.

**"No silences detected"**: Try lowering the threshold (e.g., -35dB) or reducing the minimum duration.

**Processing fails**: Check that the input file is a valid video format supported by FFmpeg.

## Installing FFmpeg

**macOS (Homebrew):**

```bash
brew install ffmpeg
```

**Ubuntu/Debian:**

```bash
sudo apt update
sudo apt install ffmpeg
```

**Windows:**
Download from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)
