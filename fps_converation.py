import os
import subprocess

import imageio_ffmpeg

ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

input_folder = (
    r"D:\bhanu\OneDrive - Imagevision.ai India Pvt Ltd\bhanu_iv061\GHD_TRAFFIC_COUNTING\Engineering\input_videos"
)
output_folder = (
    r"D:\bhanu\OneDrive - Imagevision.ai India Pvt Ltd\bhanu_iv061\GHD_TRAFFIC_COUNTING\Engineering\fps_downgradeed"
)
target_fps = 5

os.makedirs(output_folder, exist_ok=True)


def help():
    print("""
========================================================
Video FPS Downsampler using FFmpeg
========================================================

This script reduces the frame rate (FPS) of multiple videos in a folder
using FFmpeg. It processes videos in batch mode and saves new versions
with a lower FPS while preserving video quality.

FFmpeg is automatically provided via the imageio-ffmpeg package,
so no system-level FFmpeg installation is required.

--------------------------------------------------------
WORKING FLOW
--------------------------------------------------------

1. FFmpeg Binary Detection
   - Uses imageio_ffmpeg to locate a bundled FFmpeg executable.
   - Ensures FFmpeg works even if it is not installed system-wide.

2. Input & Output Folder Setup
   - Reads all video files from the input directory.
   - Creates the output directory automatically if it does not exist.

3. Video Filtering
   - Processes only valid video formats:
       • .mp4
       • .avi
       • .mkv
       • .mov

4. FPS Reduction Logic
   - Each video is passed to FFmpeg.
   - A video filter (vf) is applied:
       fps = target_fps
   - This reduces the number of frames per second while keeping
     the video duration unchanged.

5. Encoding Settings
   - Video codec: H.264 (libx264)
   - Quality: CRF 18 (high quality, near-lossless)
   - Preset: slow (better compression efficiency)
   - Audio stream: copied without re-encoding

6. Output Generation
   - Output file name includes the target FPS.
     Example:
       input.mp4 → input_fps5.mp4
   - All processed videos are saved in the output folder.

7. Batch Processing
   - Each video is processed one by one.
   - Script stops if FFmpeg encounters an error.

--------------------------------------------------------
USAGE
--------------------------------------------------------

python fps_downgrade.py

(No command-line arguments required.
Paths and FPS are defined inside the script.)

--------------------------------------------------------
OUTPUT
--------------------------------------------------------

• Videos with reduced FPS saved in the output folder.
• Original videos remain unchanged.

--------------------------------------------------------
NOTES
--------------------------------------------------------

• Lower FPS reduces file size and inference load.
• Useful for:
    - Traffic counting
    - Surveillance analytics
    - Model testing on low frame rates
• Audio quality is preserved.
• Target FPS can be adjusted easily.

========================================================
""")


help()

for file in os.listdir(input_folder):
    if file.lower().endswith((".mp4", ".avi", ".mkv", ".mov")):
        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, os.path.splitext(file)[0] + f"_fps{target_fps}.mp4")

        cmd = [
            ffmpeg_path,  # 👈 pip-installed ffmpeg binary
            "-y",
            "-i",
            input_path,
            "-vf",
            f"fps={target_fps}",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "slow",
            "-c:a",
            "copy",
            output_path,
        ]

        subprocess.run(cmd, check=True)

print("All videos processed successfully!")
