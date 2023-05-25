# Prediction interface for Cog ⚙️
# Reference: https://github.com/replicate/cog/blob/main/docs/python.md

import tempfile
from cmath import polar

import pandas as pd
import numpy as np
import seaborn as sns
import youtube_dl
from cog import BasePredictor, Input, Path
from essentia.standard import (
    MonoLoader,
    TensorflowPredictVGGish,
    TensorflowPredictMusiCNN,
    TensorflowPredict2D,
)
from matplotlib import pyplot as plt


MODELS_HOME = Path("/models")


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory and create the Essentia network for predictions"""

        self.musicnn_graph = str(MODELS_HOME / "msd-musicnn-1.pb")
        self.vggish_graph = str(MODELS_HOME / "audioset-vggish-3.pb")

        self.sample_rate = 16000

        self.loader = MonoLoader()
        self.embeddings = {
            "msd-musicnn": TensorflowPredictMusiCNN(
                graphFilename=self.musicnn_graph,
                output="model/dense/BiasAdd",
                patchHopSize=187,
            ),
            "audioset-vggish": TensorflowPredictVGGish(
                graphFilename=self.vggish_graph,
                output="model/vggish/embeddings",
                patchHopSize=96,
            ),
        }

        self.input = "flatten_in_input"
        self.output = "dense_out"
        # Algorithms for specific models.
        self.classifiers = {}

        datasets = ("emomusic", "deam", "muse")
        # datasets = ("deam", "muse")
        for dataset in datasets:
            for embedding in self.embeddings.keys():
                classifier_name = f"{dataset}-{embedding}"
                graph_filename = str(MODELS_HOME / f"{classifier_name}-1.pb")
                self.classifiers[classifier_name] = TensorflowPredict2D(
                    graphFilename=graph_filename,
                    input=self.input,
                    output=self.output,
                )

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
        embedding_type: str = Input(
            description="Embedding type to use: vggish, or musicnn",
            default="msd-musicnn",
            choices=["msd-musicnn", "audioset-vggish"],
        ),
        dataset: str = Input(
            description="Arousal/Valence training dataset",
            default="emomusic",
            choices=["emomusic", "deam", "muse"],
        ),
    ) -> Path:
        """Run a single prediction on the model"""

        assert audio or url, "Specify either an audio filename or a YouTube url"

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
        self.loader.configure(
            sampleRate=self.sample_rate,
            filename=str(audio),
            resampleQuality=4,
        )
        waveform = self.loader()

        embeddings = self.embeddings[embedding_type](waveform)

        classifier_name = f"{dataset}-{embedding_type}"
        results = self.classifiers[classifier_name](embeddings)
        results = np.mean(results.squeeze(), axis=0)

        # Manual normalization (1, 9) -> (-1, 1)
        results = (results - 5) / 4

        valence = results[0]
        arousal = results[1]

        sns.set_style("darkgrid")
        g = sns.lmplot(
            data=pd.DataFrame({"arousal": [arousal], "valence": [valence]}),
            x="valence",
            y="arousal",
            markers="x",
            scatter_kws={"s": 100},
        )

        g.set(ylim=(-1, 1))
        g.set(xlim=(-1, 1))
        plt.plot([-1, 1], [0, 0], linewidth=1.5, color="grey")
        plt.plot([0, 0], [-1, 1], linewidth=1.5, color="grey")
        plt.subplots_adjust(top=0.95, bottom=0.1, left=0.15)
        plt.title(title)

        out_path = Path(tempfile.mkdtemp()) / "out.png"
        plt.savefig(out_path)

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
