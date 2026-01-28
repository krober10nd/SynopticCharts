"""
Video generation module for creating MP4 animations from frame sequences.

This module provides the VideoGenerator class for encoding synoptic chart frames
into MP4 videos using ffmpeg. Supports various quality settings and encoding
presets for different use cases.

Example:
    >>> from synoptic_charts import VideoGenerator
    >>> from pathlib import Path
    >>> 
    >>> # Create video from frames
    >>> video_gen = VideoGenerator(
    ...     frame_dir=Path("frames"),
    ...     output_path=Path("forecast.mp4"),
    ...     fps=10
    ... )
    >>> video_gen.create_video()
"""

import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

from .exceptions import VideoCreationError

logger = logging.getLogger(__name__)

# Try to import ffmpeg-python
try:
    import ffmpeg
    FFMPEG_PYTHON_AVAILABLE = True
except ImportError:
    FFMPEG_PYTHON_AVAILABLE = False
    logger.debug("ffmpeg-python not available")


class VideoGenerator:
    """
    Generate MP4 videos from synoptic chart frame sequences.
    
    Uses ffmpeg to encode frames into a video with configurable quality
    and encoding settings. Requires ffmpeg to be installed on the system.
    
    Attributes:
        frame_dir: Directory containing frame images
        output_path: Path for output video file
        fps: Frames per second
        codec: Video codec (default: libx264)
        crf: Constant rate factor for quality (lower = better, 18-28 typical)
        preset: Encoding preset (ultrafast, fast, medium, slow, veryslow)
        
    Example:
        >>> video_gen = VideoGenerator(
        ...     frame_dir=Path("frames"),
        ...     output_path=Path("forecast.mp4"),
        ...     fps=10,
        ...     crf=23
        ... )
        >>> video_gen.create_video()
    """
    
    def __init__(
        self,
        frame_dir: Path,
        output_path: Path,
        fps: int = 10,
        codec: str = "libx264",
        crf: int = 23,
        preset: str = "medium"
    ):
        """
        Initialize video generator.
        
        Args:
            frame_dir: Directory containing frame images
            output_path: Path for output video file
            fps: Frames per second (default: 10)
            codec: Video codec (default: "libx264")
            crf: Constant rate factor, 0-51, lower=better quality (default: 23)
            preset: Encoding speed preset (default: "medium")
        """
        self.frame_dir = Path(frame_dir)
        self.output_path = Path(output_path)
        self.fps = fps
        self.codec = codec
        self.crf = crf
        self.preset = preset
        
        logger.info(
            f"Initialized VideoGenerator: fps={fps}, codec={codec}, "
            f"crf={crf}, preset={preset}"
        )
    
    def create_video(
        self,
        frame_pattern: str = "frame_%03d.png",
        overwrite: bool = False
    ) -> str:
        """
        Create video from frame sequence.
        
        Handles both contiguous and non-contiguous frame numbering by reindexing
        frames to a contiguous sequence when necessary.
        
        Args:
            frame_pattern: Printf-style pattern for frame filenames
                          (default: "frame_%03d.png")
            overwrite: Overwrite existing output file (default: False)
            
        Returns:
            Path to created video file
            
        Raises:
            VideoCreationError: If ffmpeg is unavailable or encoding fails
            FileNotFoundError: If frame directory doesn't exist
            FileExistsError: If output exists and overwrite=False
            
        Example:
            >>> video_path = video_gen.create_video(overwrite=True)
            >>> print(f"Video created: {video_path}")
        """
        logger.info(f"Creating video: {self.output_path}")
        
        # Check if ffmpeg is available
        if not shutil.which("ffmpeg"):
            raise VideoCreationError(
                "ffmpeg not found. Please install ffmpeg:\n"
                "  macOS: brew install ffmpeg\n"
                "  Ubuntu/Debian: apt-get install ffmpeg\n"
                "  Windows: Download from https://ffmpeg.org/download.html"
            )
        
        # Validate frame directory
        if not self.frame_dir.exists():
            raise FileNotFoundError(f"Frame directory not found: {self.frame_dir}")
        
        # Convert printf-style pattern to glob pattern for discovery
        # e.g., "frame_%03d.png" -> "frame_*.png"
        glob_pattern = self._convert_pattern_to_glob(frame_pattern)
        
        # Discover frames using derived glob pattern
        frame_files = sorted(self.frame_dir.glob(glob_pattern))
        
        if not frame_files:
            raise VideoCreationError(
                f"No frames found in {self.frame_dir} matching pattern '{glob_pattern}'"
            )
        
        logger.info(f"Found {len(frame_files)} frames matching pattern '{glob_pattern}'")
        
        # Check output file
        if self.output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Output file already exists: {self.output_path}. "
                "Use overwrite=True to replace."
            )
        
        # Create output directory if needed
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use create_video_from_frames() to handle non-contiguous frame numbering
        # This method creates a temporary directory with reindexed symlinks,
        # ensuring ffmpeg receives a contiguous sequence (000, 001, 002, ...)
        # even when batch outputs are non-contiguous (000, 003, 006, ...)
        logger.debug("Using create_video_from_frames() to handle frame reindexing")
        
        start_time = time.time()
        
        try:
            result = VideoGenerator.create_video_from_frames(
                frame_paths=frame_files,
                output_path=self.output_path,
                fps=self.fps,
                codec=self.codec,
                crf=self.crf,
                preset=self.preset,
            )
        except VideoCreationError:
            raise
        except Exception as e:
            raise VideoCreationError("Video encoding failed") from e
        
        # Calculate encoding time and file size
        encoding_time = time.time() - start_time
        file_size = self.output_path.stat().st_size / (1024 * 1024)  # MB
        
        logger.info(
            f"Video created successfully: {self.output_path} "
            f"({file_size:.1f} MB, encoded in {encoding_time:.1f}s)"
        )
        
        return result
    
    def _convert_pattern_to_glob(self, pattern: str) -> str:
        """
        Convert printf-style pattern to glob pattern.
        
        Args:
            pattern: Printf-style pattern (e.g., "frame_%03d.png")
            
        Returns:
            Glob pattern (e.g., "frame_*.png")
            
        Example:
            >>> self._convert_pattern_to_glob("frame_%03d.png")
            'frame_*.png'
            >>> self._convert_pattern_to_glob("img_%d.jpg")
            'img_*.jpg'
        """
        import re
        # Replace printf-style format specifiers with glob wildcard
        # Handles: %d, %03d, %04d, etc.
        glob_pattern = re.sub(r'%0?\d*d', '*', pattern)
        logger.debug(f"Converted pattern '{pattern}' to glob '{glob_pattern}'")
        return glob_pattern
    
    def _encode_with_ffmpeg_python(
        self,
        input_pattern: Path,
        overwrite: bool
    ) -> None:
        """
        Encode video using ffmpeg-python library.
        
        Args:
            input_pattern: Path with printf-style pattern for frames
            overwrite: Whether to overwrite existing output
        """
        # Build ffmpeg command
        stream = ffmpeg.input(
            str(input_pattern),
            framerate=self.fps,
            pattern_type='sequence'
        )
        
        stream = ffmpeg.output(
            stream,
            str(self.output_path),
            vcodec=self.codec,
            crf=self.crf,
            preset=self.preset,
            pix_fmt='yuv420p'  # For compatibility
        )
        
        # Run ffmpeg
        try:
            if overwrite:
                ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
            else:
                ffmpeg.run(stream, capture_stdout=True, capture_stderr=True)
        except ffmpeg.Error as e:
            stderr = e.stderr.decode() if e.stderr else "Unknown error"
            raise VideoCreationError(f"ffmpeg encoding failed: {stderr}")
    
    def _encode_with_subprocess(
        self,
        input_pattern: Path,
        overwrite: bool
    ) -> None:
        """
        Encode video using subprocess (fallback).
        
        Args:
            input_pattern: Path with printf-style pattern for frames
            overwrite: Whether to overwrite existing output
        """
        import subprocess
        
        # Build ffmpeg command
        cmd = [
            "ffmpeg",
            "-framerate", str(self.fps),
            "-i", str(input_pattern),
            "-c:v", self.codec,
            "-crf", str(self.crf),
            "-preset", self.preset,
            "-pix_fmt", "yuv420p",
        ]
        
        if overwrite:
            cmd.append("-y")
        
        cmd.append(str(self.output_path))
        
        # Run ffmpeg
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            logger.debug(f"ffmpeg output: {result.stderr}")
        except subprocess.CalledProcessError as e:
            raise VideoCreationError(f"ffmpeg encoding failed: {e.stderr}")
    
    @staticmethod
    def create_video_from_frames(
        frame_paths: List[Path],
        output_path: Path,
        fps: int = 10,
        codec: str = "libx264",
        crf: int = 23,
        preset: str = "medium"
    ) -> str:
        """
        Create video from list of frame paths (static method).
        
        Creates a temporary directory with symlinks to frames in sequential
        order, then encodes the video. Useful when frames aren't sequentially
        named.
        
        Args:
            frame_paths: List of paths to frame images (in desired order)
            output_path: Path for output video file
            fps: Frames per second (default: 10)
            codec: Video codec (default: "libx264")
            crf: Quality setting (default: 23)
            preset: Encoding preset (default: "medium")
            
        Returns:
            Path to created video file
            
        Example:
            >>> frames = [Path("frame_a.png"), Path("frame_b.png")]
            >>> VideoGenerator.create_video_from_frames(
            ...     frames,
            ...     Path("output.mp4"),
            ...     fps=10
            ... )
        """
        logger.info(f"Creating video from {len(frame_paths)} frames")
        
        import subprocess

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create temporary directory with sequential symlinks so ffmpeg can read
        # a contiguous sequence (frame_000.png, frame_001.png, ...).
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            logger.debug(f"Using temporary directory: {temp_path}")

            for i, frame_path in enumerate(frame_paths):
                frame_path = Path(frame_path)
                if not frame_path.exists():
                    raise FileNotFoundError(f"Frame not found: {frame_path}")

                link_name = temp_path / f"frame_{i:03d}.png"
                link_name.symlink_to(frame_path.resolve())

            input_pattern = temp_path / "frame_%03d.png"

            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                str(input_pattern),
                # H.264/yuv420p requires even frame dimensions. Our matplotlib
                # output can be odd-sized depending on DPI/layout, so pad by
                # at most 1px to the next even number.
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2:color=black",
                "-c:v",
                codec,
                "-crf",
                str(crf),
                "-preset",
                preset,
                "-pix_fmt",
                "yuv420p",
                str(output_path),
            ]

            try:
                subprocess.run(cmd, capture_output=True, text=True, check=True)
            except subprocess.CalledProcessError as e:
                stderr = (e.stderr or "").strip() or "Unknown ffmpeg error"
                raise VideoCreationError(f"ffmpeg encoding failed: {stderr}") from e

        return str(output_path)


def create_video_from_batch(
    batch_result: Dict[str, Any],
    output_path: Path,
    fps: int = 10,
    codec: str = "libx264",
    crf: int = 23,
    preset: str = "medium"
) -> str:
    """
    Create video from BatchChartGenerator result.
    
    Convenience function that extracts successful frames from a batch result
    and creates a video with proper frame ordering.
    
    Args:
        batch_result: Dictionary from BatchChartGenerator.generate_frames()
                     or generate_forecast_sequence()
        output_path: Path for output video file
        fps: Frames per second (default: 10)
        codec: Video codec (default: "libx264")
        crf: Quality setting (default: 23)
        preset: Encoding preset (default: "medium")
        
    Returns:
        Path to created video file
        
    Raises:
        VideoCreationError: If no successful frames or encoding fails
        
    Example:
        >>> from synoptic_charts import BatchChartGenerator, create_video_from_batch
        >>> 
        >>> batch = BatchChartGenerator("GFS", datetime(2024, 1, 15, 0))
        >>> result = batch.generate_forecast_sequence(0, 48, 6)
        >>> 
        >>> create_video_from_batch(
        ...     result,
        ...     Path("forecast.mp4"),
        ...     fps=10
        ... )
    """
    logger.info("Creating video from batch result")
    
    # Extract successful frames
    successful_frames = batch_result.get("successful_frames", [])
    
    if not successful_frames:
        raise VideoCreationError("No successful frames in batch result")
    
    logger.info(f"Processing {len(successful_frames)} frames")
    
    # Sort frames by lead time (extract from filename)
    def extract_lead_time(frame_path: str) -> int:
        """Extract lead time from frame filename."""
        try:
            filename = Path(frame_path).stem  # frame_024
            lead_time_str = filename.split("_")[-1]  # 024
            return int(lead_time_str)
        except (ValueError, IndexError):
            return 0
    
    sorted_frames = sorted(successful_frames, key=extract_lead_time)
    frame_paths = [Path(f) for f in sorted_frames]
    
    # Create video
    return VideoGenerator.create_video_from_frames(
        frame_paths=frame_paths,
        output_path=output_path,
        fps=fps,
        codec=codec,
        crf=crf,
        preset=preset
    )
