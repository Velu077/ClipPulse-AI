import os
import subprocess
import whisper

def download_youtube_video(url, output_path="input_video.mp4"):
    print(f"Downloading video from {url}...")
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
        "-o", output_path,
        url
    ]
    subprocess.run(cmd, check=True)
    print("Download complete!")
    return output_path

def transcribe_video(video_path):
    print("Loading Whisper model (base)...")
    model = whisper.load_model("base")
    print("Transcribing audio (this may take a minute)...")
    result = model.transcribe(video_path, word_timestamps=True)
    return result

def crop_to_vertical_short(input_path, output_path, start_time, duration):
    print(f"Cropping clip from {start_time}s to {start_time + duration}s into 9:16 vertical...")
    # FFmpeg crop filter: centers the video horizontally and crops to 9:16 ratio (1080x1920)
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_time),
        "-t", str(duration),
        "-i", input_path,
        "-vf", "crop=ih*(9/16):ih:(iw-ow)/2:0,scale=1080:1920",
        "-c:v", "libx264", "-crf", "23", "-preset", "fast",
        "-c:a", "aac",
        output_path
    ]
    subprocess.run(ffmpeg_cmd, check=True)
    print(f"Successfully generated vertical short: {output_path}")

if __name__ == "__main__":
    # Test with a short public YouTube URL
    youtube_url = input("Enter YouTube Video URL: ").strip()
    
    # 1. Download
    video_file = download_youtube_video(youtube_url)
    
    # 2. Transcribe
    transcript = transcribe_video(video_file)
    print(f"\nTranscription Preview:\n{transcript['text'][:300]}...\n")
    
    # 3. Cut a sample highlight (e.g., from second 10 to second 40)
    os.makedirs("outputs", exist_ok=True)
    output_clip = "outputs/sample_short.mp4"
    crop_to_vertical_short(video_file, output_clip, start_time=10, duration=30)