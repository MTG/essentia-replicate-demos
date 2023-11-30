# Prediction interface for Cog ⚙️
# Reference: https://github.com/replicate/cog/blob/main/docs/python.md

import tempfile

from cog import BasePredictor, Input, Path
from essentia.standard import (
    MonoLoader,
    RhythmExtractor2013,
    PercivalBpmEstimator,
    Resample,
    TempoCNN,
)
import numpy as np
import youtube_dl


class Predictor(BasePredictor):
    def setup(self):
        """Instantiate Essentia algorithms for predictions"""

        self.sample_rate = 44100

        # Instantiate algorithms.
        self.loader = MonoLoader(sampleRate=44100)
        self.resample = Resample(outputSampleRate=11025, quality=4)
        self.algos = {
            "degara": RhythmExtractor2013(method="degara"),
            "multifeature": RhythmExtractor2013(method="multifeature"),
            "percival": PercivalBpmEstimator(),
            "deepsquare-k16": TempoCNN(graphFilename="/models/deepsquare-k16-3.pb"),
        }

    def predict(
        self,
        audio: Path = Input(
            description="Audio file to process",
            default=None,
        ),
        url: str = Input(
            description="YouTube URL to process (overrides audio input)",
            default=None,
        ),
        algo_type: str = Input(
            description="Analysis algorithm: degara, multifeature, percival, deepsquare-k16",
            default="degara",
            choices=["degara", "multifeature", "percival", "deepsquare-k16"],
        ),
    ) -> Path:
        """Run analysis."""

        assert audio or url, "Specify either an audio filename or a YouTube URL"

        # If there is a YouTube url use that.
        if url:
            if audio:
                print(
                    "Warning: Both `url` and `audio` inputs were specified. "
                    "The `url` will be process. To process the `audio` input clear the `url` input field."
                )
            audio, title = self._download(url)
        else:
            title = audio.name

        print("loading audio...")
        self.loader.configure(sampleRate=self.sample_rate, filename=str(audio))
        waveform = self.loader()

        result = self._run_algos(waveform, algo_type, title)

        out_path = Path(tempfile.mkdtemp()) / "out.md"
        with open(out_path, "w") as f:
            f.write(result)

        print("done!")
        return out_path

    def _download(self, url, ext="wav"):
        """Download a YouTube URL in the specified format to a temporal directory"""

        tmp_dir = Path(tempfile.mktemp())
        ydl_opts = {
            # The download is quite slow, use the most compressed format that doesn't affect
            # the sense of the prediction (too much):
            #
            # Code  Container  Audio Codec  Audio Bitrate     Channels    Still offered?
            # 250   WebM       Opus (VBR)   ~70 Kbps          Stereo (2)  Yes
            # 251   WebM       Opus         (VBR) <=160 Kbps  Stereo (2)  Yes
            # 40    MP4        AAC (LC)     128 Kbps          Stereo (2)  Yes, YT Music
            #
            # Download speeds:
            # 250 -> ~19s, 251 -> 30s, 40 -> ~35s
            # but 250 changes the predictions too munch. Using 251 as a compromise.
            #
            # From https://gist.github.com/AgentOak/34d47c65b1d28829bb17c24c04a0096f
            "format": "251",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": ext,
                }
            ],
            # render audio @16kHz to prevent resampling latter on
            "postprocessor_args": ["-ar", f"{self.sample_rate}"],
            "outtmpl": str(tmp_dir / "audio.%(ext)s"),
        }

        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url)

            if "title" in info:
                title = info["title"]
            else:
                title = ""  # is it possible that the title metadata is unavailable? Continue anyway

        paths = [p for p in tmp_dir.glob(f"audio.{ext}")]
        assert (
            len(paths) == 1
        ), "Something unexpected happened. Should be only one match!"

        return paths[0], title

    def _run_algos(self, waveform: np.ndarray, algo_type: str, title: str):
        print("running analysis algorithm...")

        if algo_type in ["degara", "multifeature"]:
            bpm = self.algos[algo_type](waveform)[0]
        elif algo_type == "percival":
            bpm = self.algos[algo_type](waveform)
        elif algo_type == "deepsquare-k16":
            waveform = self.resample(waveform)
            bpm = self.algos[algo_type](waveform)[0]

        return f"Estimated BPM: {bpm}"
