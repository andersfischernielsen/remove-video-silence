#!/usr/bin/env -S uv run

# /// script
# dependencies = [
#   "rich"
# ]
# ///

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn


@dataclass
class ProcessResult:
    returncode: int
    stdout: str
    stderr: str


console = Console()


def format_duration(seconds: float):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    if hours > 0:
        return f"{hours}h {minutes}m {secs:.1f}s"
    elif minutes > 0:
        return f"{minutes}m {secs:.1f}s"
    else:
        return f"{secs:.1f}s"


def run_command(cmd: str, description: str = "", capture_output: bool = True):
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=capture_output, text=True
        )

        if result.returncode != 0 and description:
            error_msg = f"[red]Error in {description}[/red]"
            if capture_output and result.stderr:
                error_msg += f": {result.stderr.strip()}"
            console.print(error_msg)

        return result
    except Exception as e:
        if description:
            console.print(f"[red]Command failed in {description}:[/red] {e}")
        return None


def run_with_progress(description: str, task_func, show_percentage: bool = False):
    columns = [SpinnerColumn(), TextColumn("[progress.description]{task.description}")]
    if show_percentage:
        columns.extend(
            [BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%")]
        )

    with Progress(*columns, console=console) as progress:
        task = progress.add_task(
            description, total=None if not show_percentage else 100
        )
        return task_func(progress, task)


def parse_silence_log(log_file: str):
    try:
        with open(log_file, "r") as f:
            content = f.read()

        silence_starts = [
            float(t) for t in re.findall(r"silence_start:\s*([\d.]+)", content)
        ]
        silence_ends = [
            float(t) for t in re.findall(r"silence_end:\s*([\d.]+)", content)
        ]

        console.print(f"   [dim]Found {len(silence_starts)} silence start times[/dim]")
        console.print(f"   [dim]Found {len(silence_ends)} silence end times[/dim]")

        min_len = min(len(silence_starts), len(silence_ends))
        return silence_starts[:min_len], silence_ends[:min_len]

    except Exception as e:
        console.print(f"[red]Error parsing silence log:[/red] {e}")
        return [], []


def build_keep_intervals(
    silence_starts: list[float],
    silence_ends: list[float],
    total_duration: float,
    min_segment_duration: float = 0.1,
):
    keep_intervals = []

    if silence_starts and silence_starts[0] > min_segment_duration:
        keep_intervals.append((0.0, silence_starts[0]))

    for i in range(len(silence_ends)):
        segment_start = silence_ends[i]

        if i + 1 < len(silence_starts):
            segment_end = silence_starts[i + 1]
        else:
            segment_end = total_duration

        if segment_end - segment_start > min_segment_duration:
            keep_intervals.append((segment_start, segment_end))

    console.print(f"   [dim]Built {len(keep_intervals)} non-overlapping segments[/dim]")
    return keep_intervals


def get_video_duration(input_file: str):
    cmd = f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "{input_file}"'
    result = run_command(cmd, "getting video duration")

    if result and result.returncode == 0:
        try:
            return float(result.stdout.strip())
        except ValueError:
            console.print(f"[red]Could not parse duration: {result.stdout}[/red]")
    return None


def extract_segment(
    input_file: str,
    start_time: float,
    end_time: float,
    output_file: str,
    use_stream_copy: bool = True,
):
    if use_stream_copy:
        cmd = (
            f"ffmpeg -hide_banner -loglevel error -y "
            f'-i "{input_file}" '
            f"-ss {start_time} -to {end_time} "
            f"-c copy "
            f'"{output_file}"'
        )
    else:
        cmd = (
            f"ffmpeg -hide_banner -loglevel error -y "
            f'-i "{input_file}" '
            f"-ss {start_time} -to {end_time} "
            f"-c:v libx264 -crf 18 -preset medium "
            f"-c:a aac -b:a 192k "
            f'"{output_file}"'
        )

    result = run_command(cmd, capture_output=True)
    success = result and result.returncode == 0

    if not success and use_stream_copy:
        return extract_segment(
            input_file, start_time, end_time, output_file, use_stream_copy=False
        )

    return success


def create_filter_complex_cmd(
    input_file: str,
    keep_intervals: list[tuple[float, float]],
    output_file: str,
    progress_file: str | None = None,
):
    if not keep_intervals:
        return None

    filter_parts = []
    concat_inputs = []

    for i, (start_time, end_time) in enumerate(keep_intervals):
        filter_parts.append(
            f"[0:v]trim=start={start_time}:end={end_time},setpts=PTS-STARTPTS[v{i}]"
        )
        filter_parts.append(
            f"[0:a]atrim=start={start_time}:end={end_time},asetpts=PTS-STARTPTS[a{i}]"
        )
        concat_inputs.extend([f"[v{i}][a{i}]"])

    n_segments = len(keep_intervals)
    concat_filter = f"{''.join(concat_inputs)}concat=n={n_segments}:v=1:a=1[outv][outa]"

    filter_complex = ";".join(filter_parts + [concat_filter])

    cmd = (
        f"ffmpeg -hide_banner -loglevel error -y "
        f'-i "{input_file}" '
        f'-filter_complex "{filter_complex}" '
        f'-map "[outv]" -map "[outa]" '
        f"-c:v libx264 -crf 18 -preset medium "
        f"-c:a aac -b:a 192k "
    )

    if progress_file:
        cmd += f' -progress "{progress_file}"'

    cmd += f' "{output_file}"'

    return cmd


def run_ffmpeg_with_progress(
    cmd: str, total_duration: float, progress_obj: Progress, progress_task: int
):
    cmd_with_progress = cmd.replace("-loglevel error", "-loglevel info -stats")

    process = subprocess.Popen(
        cmd_with_progress,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    last_time = 0
    while True:
        output = process.stderr.readline()
        if output == "" and process.poll() is not None:
            break

        if output:
            time_match = re.search(r"time=(\d+):(\d+):(\d+)\.(\d+)", output)
            if time_match:
                hours, minutes, seconds, centiseconds = map(int, time_match.groups())
                current_time = (
                    hours * 3600 + minutes * 60 + seconds + centiseconds / 100
                )

                if current_time > last_time and total_duration > 0:
                    percentage = min((current_time / total_duration) * 100, 100)
                    progress_obj.update(
                        progress_task,
                        completed=percentage,
                        description=f"Processing video: {percentage:.1f}%",
                    )
                    last_time = current_time

    stdout, stderr = process.communicate()

    return ProcessResult(process.returncode, stdout, stderr)


def concatenate_segments(segment_list_file: str, output_file: str):
    cmd = (
        f"ffmpeg -hide_banner -y "
        f'-f concat -safe 0 -i "{segment_list_file}" '
        f'-c copy "{output_file}"'
    )

    result = run_command(cmd, capture_output=False)
    return result and result.returncode == 0


def detect_silences(input_path: str, threshold: str, duration: float):
    silence_log = "silence.log"
    cmd = (
        f'ffmpeg -hide_banner -y -i "{input_path}" '
        f'-af "silencedetect=noise={threshold}:d={duration}" '
        f"-f null - 2> {silence_log}"
    )

    def _detect(progress, task):
        result = run_command(cmd, capture_output=True)
        if not result or result.returncode != 0:
            progress.update(task, description="[red]Failed to detect silences[/red]")
            console.print("[red]Failed to detect silences[/red]")
            sys.exit(1)

        progress.update(task, description="[green]Silence detection complete[/green]")
        return parse_silence_log(silence_log)

    return run_with_progress("Detecting silences...", _detect)


def process_silence_data(
    silence_starts: list[float], silence_ends: list[float], total_duration: float
):
    if not silence_starts and not silence_ends:
        console.print(
            "[yellow]No silences detected. Output would be identical to input.[/yellow]"
        )
        sys.exit(0)

    def _process(progress, task):
        progress.update(
            task,
            description=f"[green]Found {len(silence_starts)} silence periods[/green]",
        )

        keep_intervals = build_keep_intervals(
            silence_starts, silence_ends, total_duration
        )
        if not keep_intervals:
            progress.update(task, description="[red]No segments to keep[/red]")
            console.print("[red]No segments found after filtering silences.[/red]")
            sys.exit(1)

        progress.update(
            task, description=f"[green]Built {len(keep_intervals)} segments[/green]"
        )
        return keep_intervals

    return run_with_progress("Processing silence data...", _process)


def print_processing_stats(
    silence_starts: list[float], silence_ends: list[float], total_duration: float
):
    total_silence_time = sum(
        min(end, total_duration) - start
        for start, end in zip(silence_starts, silence_ends)
        if start < total_duration
    )
    estimated_final_duration = total_duration - total_silence_time
    estimated_time_saved_percent = (
        (total_silence_time / total_duration * 100) if total_duration > 0 else 0
    )

    console.print(f"   [dim]Original duration: {format_duration(total_duration)}[/dim]")
    console.print(
        f"   [dim]Estimated final duration: {format_duration(estimated_final_duration)}[/dim]"
    )
    console.print(
        f"   [dim]Estimated time saved: {format_duration(total_silence_time)} ({estimated_time_saved_percent:.1f}%)[/dim]"
    )


def process_video(
    input_path: str,
    keep_intervals: list[tuple[float, float]],
    output_file: str,
    total_duration: float,
):
    def _process(progress, task):
        filter_cmd = create_filter_complex_cmd(input_path, keep_intervals, output_file)
        if not filter_cmd:
            console.print("[red]Could not create filter command[/red]")
            return False

        result = run_ffmpeg_with_progress(filter_cmd, total_duration, progress, task)
        if not result or result.returncode != 0:
            progress.update(task, description="[red]Processing failed[/red]")
            return False

        progress.update(task, description="[green]Processing complete[/green]")
        return True

    return run_with_progress(
        "Processing video with filter_complex...", _process, show_percentage=True
    )


def print_success_stats(
    output_file: str, keep_intervals: list[tuple[float, float]], total_duration: float
):
    if not os.path.exists(output_file):
        console.print("[red]Output file was not created[/red]")
        sys.exit(1)

    file_size = os.path.getsize(output_file)
    size_mb = file_size / (1024 * 1024)

    output_duration = get_video_duration(output_file)
    time_saved = total_duration - output_duration if output_duration else 0
    time_saved_percent = (
        (time_saved / total_duration * 100) if total_duration > 0 else 0
    )

    success_message = (
        f"[bold green]Success![/bold green]\n"
        f"[bold blue]Output file:[/bold blue] {output_file}\n"
        f"[bold blue]File size:[/bold blue] {size_mb:.1f}MB\n"
        f"[bold blue]Segments processed:[/bold blue] {len(keep_intervals)}\n"
        f"[bold blue]Original duration:[/bold blue] {format_duration(total_duration)}\n"
    )

    if output_duration:
        success_message += (
            f"[bold blue]Final duration:[/bold blue] {format_duration(output_duration)}\n"
            f"[bold green]Time saved:[/bold green] {format_duration(time_saved)} ({time_saved_percent:.1f}%)"
        )
    else:
        success_message += (
            "[bold yellow]Could not determine final duration[/bold yellow]"
        )

    console.print(
        Panel(success_message, title="Conversion Complete", border_style="green")
    )


def main():
    parser = argparse.ArgumentParser(description="Remove silence from video files")
    parser.add_argument("-i", "--input", required=True, help="Input video file")
    parser.add_argument(
        "-t",
        "--threshold",
        default="-30dB",
        help='Noise threshold (e.g., -30dB). Use quotes or equals: -t=-30dB or -t "-30dB"',
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=float,
        default=0.1,
        help="Minimum silence duration to cut (seconds, default: 0.1)",
    )
    parser.add_argument("-o", "--output", default="output.mp4", help="Output file name")
    parser.add_argument(
        "--max-segments",
        type=int,
        default=100,
        help="Maximum number of segments to process (default: 100)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        console.print(f"[red]Error: Input file '{args.input}' not found[/red]")
        sys.exit(1)

    input_path = os.path.abspath(args.input)

    console.print(
        Panel(
            f"[bold blue]Processing:[/bold blue] {args.input}\n"
            f"[bold green]Threshold:[/bold green] {args.threshold}, "
            f"[bold green]Min duration:[/bold green] {args.duration}s",
            title="Video Silence Removal",
            border_style="blue",
        )
    )

    total_duration = get_video_duration(input_path)
    if total_duration is None:
        sys.exit(1)

    silence_starts, silence_ends = detect_silences(
        input_path, args.threshold, args.duration
    )
    keep_intervals = process_silence_data(silence_starts, silence_ends, total_duration)

    print_processing_stats(silence_starts, silence_ends, total_duration)

    if not process_video(input_path, keep_intervals, args.output, total_duration):
        sys.exit(1)

    print_success_stats(args.output, keep_intervals, total_duration)


if __name__ == "__main__":
    main()
