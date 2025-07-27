#!/usr/bin/env python3
"""media_backdrop_processor.py
Generate short backdrop clips for Emby/Jellyfin libraries.

Features
========
* Adjustable **clip length** (`--length`, default 5 s).
* Selectable **resolution** 720p, 1080p, 1440p, or 2160p (`--resolution`, default 720p).
  The filter only downsizes → no up‑scaling of small sources.
* Tunable **compression** via CRF (`--crf`, default 28) and preset
  (`--preset`, default *veryfast*).
* **Audio control** (`--no-audio`) to generate silent clips (default includes audio).
* **Expert mode** (`--ffmpeg-extra`) to add custom FFmpeg parameters.
* **Overwrite** mode (`--force`) to re‑generate even if the backdrop exists
  or a *.failed* placeholder is present.
* Optional **per‑folder delay** (`--delay`, default 0 s).
* Daemon mode with re‑scan interval just as before.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
LOG_FILE = SCRIPT_DIR / "media_processor.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BACKDROP_TIMEOUT_DEFAULT = 300  # seconds per FFmpeg call
PLACEHOLDER_SUFFIX = ".failed"  # mark failed attempts
VIDEO_EXTS = {".mkv", ".mp4", ".avi", ".mov", ".wmv", ".flv", ".webm", ".m4v"}

# ---------------------------------------------------------------------------
# Processor class
# ---------------------------------------------------------------------------


class MediaBackdropProcessor:
    """Create backdrop clips for movie & TV folders."""

    def __init__(
        self,
        movies_path: str,
        tv_path: str,
        *,
        timeout: int,
        clip_len: int,
        resolution: int,
        crf: int,
        preset: str,
        delay: float,
        force: bool,
        include_audio: bool,
        ffmpeg_extra: str,
    ) -> None:
        self.movies_path = Path(movies_path)
        self.tv_path = Path(tv_path)
        self.timeout = timeout
        self.clip_len = clip_len
        self.resolution = resolution
        self.crf = crf
        self.preset = preset
        self.delay = delay
        self.force = force
        self.include_audio = include_audio
        self.ffmpeg_extra = ffmpeg_extra

        self.width, self.height = {
            720: (1280, 720),
            1080: (1920, 1080),
            1440: (2560, 1440),
            2160: (3840, 2160),
        }.get(resolution, (1280, 720))

    @staticmethod
    def _touch_placeholder(target: Path) -> None:
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            (target.with_suffix(target.suffix + PLACEHOLDER_SUFFIX)).touch(exist_ok=True)
        except Exception as exc:
            logger.error("Could not create placeholder for %s: %s", target, exc)

    @staticmethod
    def _video_duration(video: Path) -> float:
        try:
            cmd = [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                str(video),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return float(json.loads(result.stdout)["format"]["duration"])
        except Exception as exc:
            logger.warning("Duration error for %s: %s", video, exc)
            return 0.0

    @staticmethod
    def _video_dimensions(video: Path) -> tuple[int, int]:
        try:
            cmd = [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_streams",
                "-select_streams",
                "v:0",
                str(video),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            streams = json.loads(result.stdout)["streams"]
            if streams:
                return int(streams[0]["width"]), int(streams[0]["height"])
            return 0, 0
        except Exception as exc:
            logger.warning("Dimensions error for %s: %s", video, exc)
            return 0, 0

    def _extract_clip(self, src: Path, dst: Path, video_index: int = 1) -> bool:
        total = self._video_duration(src)
        #if total < self.clip_len * 1.6:
        #    logger.info("Video too short: %s", src)
        #    return False
    
        # Dynamic safe ranges based on video index (1-5, cycling back to 1 after 5)
        index = ((video_index - 1) % 5) + 1  # Ensure we cycle through 1-5
        
        # Define 5 segments across the video duration
        # Skip first 5% and last 15% to avoid credits/intro issues
        usable_duration = total * 0.8  # Use 80% of video (skip 5% start, 15% end)
        segment_size = usable_duration / 5
        start_offset = total * 0.05  # 5% into the video
        
        # Calculate segment boundaries
        segment_start = start_offset + (index - 1) * segment_size
        segment_end = start_offset + index * segment_size
        
        # Ensure we have enough room for the clip within this segment
        if segment_end - segment_start < self.clip_len:
            logger.info("Segment too small for clip in %s (index %d)", src, video_index)
            return False
        
        # Calculate the safe range within this segment (leaving room for the clip)
        start_safe = segment_start
        end_safe = segment_end - self.clip_len
        
        # If the segment is still too small after accounting for clip length, use fallback
        if end_safe <= start_safe:
            # Fallback to original safe range if segment is too small
            start_safe, end_safe = total * 0.1, total * 0.5
            if end_safe - start_safe < self.clip_len:
                logger.info("Insufficient safe range in %s", src)
                return False
            logger.debug("Using fallback range for %s (index %d)", src.name, video_index)
        else:
            logger.debug("Using segment %d (%.1f%%-%.1f%%) for %s", 
                        index, 
                        (segment_start/total)*100, 
                        (segment_end/total)*100, 
                        src.name)
    
        # Get input video dimensions and adjust output resolution to avoid upscaling
        input_width, input_height = self._video_dimensions(src)
        if input_width > 0 and input_height > 0:
            # Use the smaller of target resolution or input resolution to avoid upscaling
            output_width = min(self.width, input_width)
            output_height = min(self.height, input_height)
            
            # Maintain aspect ratio - scale down proportionally if needed
            input_aspect = input_width / input_height
            target_aspect = output_width / output_height
            
            if input_aspect > target_aspect:
                # Input is wider, fit to width
                output_height = int(output_width / input_aspect)
            else:
                # Input is taller, fit to height  
                output_width = int(output_height * input_aspect)
                
            logger.debug("Input: %dx%d, Target: %dx%d, Output: %dx%d", 
                        input_width, input_height, self.width, self.height, output_width, output_height)
        else:
            # Fallback to target resolution if we can't detect input dimensions
            output_width, output_height = self.width, self.height
    
        start = random.uniform(start_safe, end_safe)
        dst.parent.mkdir(parents=True, exist_ok=True)
    
        # Use simple scale filter without padding to avoid black borders
        scale_filter = f"scale={output_width}:{output_height}"
    
        ff_cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start}",
            "-i",
            str(src),
            "-t",
            str(self.clip_len),
            "-c:v",
            "libx264",
            "-vf",
            scale_filter,
            "-preset",
            self.preset,
            "-crf",
            str(self.crf),
            "-avoid_negative_ts",
            "make_zero",
        ]
    
        # Handle audio options
        if self.include_audio:
            ff_cmd.extend(["-c:a", "aac"])
        else:
            ff_cmd.extend(["-an"])  # Remove audio stream
    
        # Add custom FFmpeg parameters if provided
        if self.ffmpeg_extra:
            try:
                # Parse the extra parameters safely
                extra_params = shlex.split(self.ffmpeg_extra)
                ff_cmd.extend(extra_params)
                logger.debug("Added custom FFmpeg parameters: %s", extra_params)
            except ValueError as exc:
                logger.warning("Could not parse custom FFmpeg parameters '%s': %s", self.ffmpeg_extra, exc)
    
        # Add output file at the end
        ff_cmd.append(str(dst))
    
        try:
            subprocess.run(ff_cmd, capture_output=True, text=True, timeout=self.timeout, check=True)
            logger.info("Backdrop created: %s (from segment %d, %.1f%%-%.1f%%, start=%.1fs)", 
                       dst, index, (segment_start/total)*100, (segment_end/total)*100, start)
            return True
        except subprocess.TimeoutExpired:
            logger.error("FFmpeg timed out (%ss) for %s", self.timeout, src)
        except subprocess.CalledProcessError as exc:
            logger.error("FFmpeg error for %s: %s", src, exc.stderr or exc)
        except Exception as exc:
            logger.error("Unexpected error for %s: %s", src, exc)
    
        self._touch_placeholder(dst)
        return False

    def _generate_backdrop_image(self, folder: Path, video_path: Path) -> None:
        """Generate a backdrop image from the created video clip."""
        # Check for existing backdrop images
        backdrop_extensions = [".jpg", ".jpeg", ".png"]
        existing_backdrops = []
        
        for ext in backdrop_extensions:
            backdrop_path = folder / f"backdrop{ext}"
            if backdrop_path.exists():
                existing_backdrops.append(backdrop_path)
        
        # Skip if backdrop already exists and not in force mode
        if existing_backdrops: # and not self.force:
            logger.debug("Backdrop image already exists for %s", folder.name)
            return
        
        if self.force:
            for backdrop in existing_backdrops:
                # TODO Extra flag to remove existing backdrops
                # backdrop.unlink(missing_ok=True)
                logger.debug("Removed existing backdrop: %s", backdrop)
        
        backdrop_dst = folder / "backdrop.jpg"
        
        # Extract multiple frames and pick the best one
        # Use 3 different time points to increase chances of getting a good frame
        frame_times = [
            min(2.0, self.clip_len * 0.3),   # 30% into clip or 2 seconds
            min(3.0, self.clip_len * 0.5),   # 50% into clip or 3 seconds  
            min(4.0, self.clip_len * 0.7),   # 70% into clip or 4 seconds
        ]
        
        # Try each frame time and use the first successful extraction
        for i, frame_time in enumerate(frame_times):
            temp_output = backdrop_dst.with_suffix(f".temp{i}.jpg")
            
            ff_cmd = [
                "ffmpeg",
                "-y",
                "-ss", str(frame_time),
                "-i", str(video_path),
                "-vframes", "1",
                "-vf", "select=eq(pict_type\\,I)",  # Prefer I-frames (keyframes) for better quality
                "-q:v", "1",  # Highest quality JPEG (was 2)
                "-pix_fmt", "yuvj420p",  # Better color space for JPEG
                str(temp_output)
            ]
            
            try:
                subprocess.run(ff_cmd, capture_output=True, text=True, timeout=30, check=True)
                # If successful, move temp file to final location and clean up
                temp_output.rename(backdrop_dst)
                logger.info("Backdrop image created: %s (from frame at %.1fs)", backdrop_dst, frame_time)
                
                # Clean up any remaining temp files
                for j in range(len(frame_times)):
                    temp_file = backdrop_dst.with_suffix(f".temp{j}.jpg")
                    temp_file.unlink(missing_ok=True)
                return
                
            except subprocess.CalledProcessError as exc:
                logger.debug("Frame extraction failed at %.1fs for %s: %s", frame_time, folder.name, exc.stderr or exc)
                temp_output.unlink(missing_ok=True)  # Clean up failed temp file
                continue
            except subprocess.TimeoutExpired:
                logger.debug("Frame extraction timed out at %.1fs for %s", frame_time, folder.name)
                temp_output.unlink(missing_ok=True)
                continue
            except Exception as exc:
                logger.debug("Unexpected error extracting frame at %.1fs for %s: %s", frame_time, folder.name, exc)
                temp_output.unlink(missing_ok=True)
                continue
        
        # If all frame extractions failed, fall back to simple extraction
        logger.warning("All I-frame extractions failed for %s, trying simple extraction", folder.name)
        
        ff_cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(frame_times[1]),  # Use middle time point
            "-i", str(video_path),
            "-vframes", "1",
            "-q:v", "1",  # Highest quality
            "-pix_fmt", "yuvj420p",
            str(backdrop_dst)
        ]
        
        try:
            subprocess.run(ff_cmd, capture_output=True, text=True, timeout=30, check=True)
            logger.info("Backdrop image created (fallback): %s", backdrop_dst)
        except subprocess.TimeoutExpired:
            logger.error("FFmpeg timed out while creating backdrop image for %s", folder.name)
        except subprocess.CalledProcessError as exc:
            logger.error("FFmpeg error creating backdrop image for %s: %s", folder.name, exc.stderr or exc)
        except Exception as exc:
            logger.error("Unexpected error creating backdrop image for %s: %s", folder.name, exc)
    
    @staticmethod
    def _find_videos(folder: Path) -> List[Path]:
        return sorted(p for p in folder.rglob("*") if p.suffix.lower() in VIDEO_EXTS)

    def _find_first_ep(self, show: Path) -> Optional[Path]:
        seasons = [d for d in show.iterdir() if d.is_dir() and "season" in d.name.lower()]
        season1 = next((d for d in seasons if "01" in d.name or "season 1" in d.name.lower()), None) or (
            seasons[0] if seasons else None
        )
        for folder in ([season1] if season1 else [show]):
            vids = self._find_videos(folder)
            if not vids:
                continue
            for v in vids:
                l = v.name.lower()
                if "e01" in l or "episode 1" in l:
                    return v
            return vids[0]
        return None

    def _process(self, folder: Path, is_tv: bool) -> None:
        label = "TV" if is_tv else "Movie"
        logger.info("Processing %s: %s", label, folder.name)

        backdrops_dir = folder / "backdrops"
        video_num = 1
        while True:
            dst = backdrops_dir / f"video{video_num}.mp4"
            placeholder = dst.with_suffix(dst.suffix + PLACEHOLDER_SUFFIX)
            if not dst.exists() and not placeholder.exists():
                break
            video_num += 1
        
        if self.force:
            dst = backdrops_dir / "video1.mp4"
            placeholder = dst.with_suffix(dst.suffix + PLACEHOLDER_SUFFIX)
            
        # Check for existing will fail if force=false due to automated video number increase. Add new parameter to skip if video_num 1 exists?
        if not self.force and (dst.exists() or placeholder.exists()):
            logger.debug("Backdrop already present for %s", folder.name)
            return

        if self.force:
            if dst.exists():
                dst.unlink(missing_ok=True)
            if placeholder.exists():
                placeholder.unlink(missing_ok=True)

        src: Optional[Path]
        if is_tv:
            src = self._find_first_ep(folder)
        else:
            vids = self._find_videos(folder)
            src = max(vids, key=lambda p: p.stat().st_size) if vids else None

        if not src:
            logger.warning("No video found in %s", folder)
            self._touch_placeholder(dst)
            return

        video_created = self._extract_clip(src, dst, video_num)
        
        if video_created:
            self._generate_backdrop_image(folder, dst)

        if self.delay:
            time.sleep(self.delay)

    def run_once(self) -> None:
        if not self.movies_path.exists() and not self.tv_path.exists():
            logger.error("Both movies and TV paths are missing – nothing to do.")
            return

        if self.movies_path.exists():
            for f in (d for d in self.movies_path.iterdir() if d.is_dir()):
                try:
                    self._process(f, is_tv=False)
                except Exception as exc:
                    logger.error("Error processing movie %s: %s", f.name, exc)

        if self.tv_path.exists():
            for f in (d for d in self.tv_path.iterdir() if d.is_dir()):
                try:
                    self._process(f, is_tv=True)
                except Exception as exc:
                    logger.error("Error processing TV show %s: %s", f.name, exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate backdrop clips for Emby/Jellyfin.")
    p.add_argument("--movies", required=True, help="Path to the movies directory")
    p.add_argument("--tv", required=True, help="Path to the TV‑shows directory")

    p.add_argument("--daemon", action="store_true", help="Run continuously and rescan on an interval")
    p.add_argument("--interval", type=int, default=3600, help="Seconds between scans when --daemon is set")

    p.add_argument("--length", type=int, default=5, help="Clip length in seconds (default 5)")
    p.add_argument("--resolution", type=int, choices=[720, 1080, 1440, 2160], default=720, help="Output resolution (default 720)")
    p.add_argument("--crf", type=int, default=28, help="x264 CRF value (default 28 → smaller file)")
    p.add_argument("--preset", default="veryfast", help="x264 preset (default 'veryfast')")

    p.add_argument("--no-audio", action="store_true", help="Generate clips without audio (default includes audio)")
    p.add_argument("--ffmpeg-extra", type=str, default="", help="Expert mode: additional FFmpeg parameters (e.g., '--ffmpeg-extra \"-movflags +faststart -pix_fmt yuv420p\"')")

    p.add_argument("--timeout", type=int, default=BACKDROP_TIMEOUT_DEFAULT, help="FFmpeg timeout in seconds")
    p.add_argument("--delay", type=float, default=0, help="Seconds to wait after each folder")
    p.add_argument("--force", action="store_true", help="Overwrite existing backdrops and ignore placeholders")

    return p.parse_args()


def main() -> None:
    args = parse_cli()

    processor = MediaBackdropProcessor(
        args.movies,
        args.tv,
        timeout=args.timeout,
        clip_len=args.length,
        resolution=args.resolution,
        crf=args.crf,
        preset=args.preset,
        delay=args.delay,
        force=args.force,
        include_audio=not args.no_audio,  # Default True, False when --no-audio is set
        ffmpeg_extra=args.ffmpeg_extra,
    )

    audio_status = "with audio" if not args.no_audio else "without audio"
    extra_info = f" + custom params: {args.ffmpeg_extra}" if args.ffmpeg_extra else ""

    if args.daemon:
        logger.info(
            "Daemon mode – interval:%ds len:%ds res:%dp crf:%d preset:%s %s%s",
            args.interval,
            args.length,
            args.resolution,
            args.crf,
            args.preset,
            audio_status,
            extra_info,
        )
        try:
            while True:
                processor.run_once()
                logger.info("Sleeping …")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            logger.info("Daemon stopped by user")
    else:
        logger.info("Single run mode – %s%s", audio_status, extra_info)
        processor.run_once()


if __name__ == "__main__":
    main()