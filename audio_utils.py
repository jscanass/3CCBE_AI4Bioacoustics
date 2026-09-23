import wave
from pathlib import Path

import numpy as np
import wavio

WINDOW_DURATION = 15360 / 441000
SAMPLERATE = 441000


def load_audio(
    path: str | Path,
    start_time: float = 0,
    end_time: float | None = None,
) -> tuple[np.ndarray, int]:
    """Load audio from wav file

    Can specify load starting time and ending time, which is particularly
    helpful to quickly retrieve a small audio clip without loading the full
    file.

    If `start_time` is not provided audio will be loaded from the start
    of the recording. If `end_time` is not provided loading will stop at
    the end of the recording.

    If `start_time` or `end_time` fall outside the recording's duration the
    returned wav will be padded with 0's.

    Parameters
    ----------
    path: str
        Path to audio file in filesystem
    start_time: float
        Time from which to start loading audio, in seconds. Time is
        relative to the start of the recording and computed based on the
        recording's samplerate. If the recording is time expanded, adjust the
        `start_time` accordingly. Defaults to 0.
    end_time: Optional[float]
        Time at which to end loading audio, in seconds. Time is
        relative to the start of the recording and computed based on the
        recording's samplerate. If the recording is time expanded, adjust the
        `end_time` accordingly.

    Returns
    -------
    wav: numpy.ndarray
        2D-Array with the loaded audio data. The array is two dimensional
        with shape [nframes, channels].
    sr: int
        Samplerate of the audio file

    """
    if isinstance(path, Path):
        path = str(path)

    with wave.open(path, "rb") as wav_file:
        # Read wav file parameters
        params = wav_file.getparams()

        samplerate = params.framerate
        length = params.nframes
        channels = params.nchannels

        start = int(np.floor(start_time * samplerate))

        if end_time is None:
            end = length
        else:
            end = int(np.floor(end_time * samplerate))

        if (start >= length) or (end <= 0):
            return np.zeros([end - start, channels]), samplerate

        if start > 0:
            wav_file.setpos(start)  # type: ignore

        extra_start = min(start, 0)
        start = max(start, 0)

        extra_end = max(end, length) - length
        end = min(end, length)

        data = wav_file.readframes(end - start)  # type: ignore

        # Turn into numpy array
        wav: np.ndarray = wavio._wav2array(
            channels,
            params.sampwidth,
            data,
        )

        # Normalize to [-1, 1]
        wav = wav / np.iinfo(wav.dtype).max

        # Pad with 0s if start_time of end_time extends over the recording
        # interval
        if (extra_start < 0) or (extra_end > 0):
            wav = np.pad(
                wav,
                [  # type: ignore
                    [-extra_start, extra_end],
                    [0, 0],
                ],
            )

        return wav, samplerate
