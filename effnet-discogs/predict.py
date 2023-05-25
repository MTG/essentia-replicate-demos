# Prediction interface for Cog ⚙️
# Reference: https://github.com/replicate/cog/blob/main/docs/python.md

from pathlib import Path
import tempfile
from itertools import chain
from textwrap import wrap

from cog import BasePredictor, Input, Path
from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs, TensorflowPredict2D
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas
import youtube_dl

from labels import labels


def process_labels(label):
    genre, style = label.split("---")
    return f"{style}\n({genre})"


labels = list(map(process_labels, labels))


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory and create the Essentia network for predictions"""

        self.embedding_model_file = "/models/discogs-effnet-bs64-1.pb"
        self.classification_model_file = "/models/genre_discogs400-discogs-effnet-1.pb"
        self.output = "activations"
        self.sample_rate = 16000

        self.loader = MonoLoader()
        self.tensorflowPredictEffnetDiscogs = TensorflowPredictEffnetDiscogs(
            graphFilename=self.embedding_model_file,
            output="PartitionedCall:1",
        )
        self.classification_model = TensorflowPredict2D(
            graphFilename=self.classification_model_file,
            input="serving_default_model_Placeholder",
            output="PartitionedCall:0",
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
        top_n: int = Input(description="Top n music styles to show", default=10),
        output_format: str = Input(
            description="Output either a bar chart visualization or a JSON blob",
            default="Visualization",
            choices=["Visualization", "JSON"],
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
        self.loader.configure(sampleRate=self.sample_rate, filename=str(audio))
        waveform = self.loader()

        print("running the model...")
        embeddings = self.tensorflowPredictEffnetDiscogs(waveform)
        activations = self.classification_model(embeddings)
        activations_mean = np.mean(activations, axis=0)

        if output_format == "JSON":
            return dict(zip(labels, activations_mean))

        print("plotting...")
        top_n_idx = np.argsort(activations_mean)[::-1][:top_n]

        result = {
            "label": list(
                chain(*[[labels[idx]] * activations.shape[0] for idx in top_n_idx])
            ),
            "activation": list(chain(*[activations[:, idx] for idx in top_n_idx])),
        }
        result = pandas.DataFrame.from_dict(result)

        # Wrap title to lines of approximately 50 chars.
        title = wrap(title, width=50)

        # Allow a maximum of 2 lines of title.
        if len(title) > 2:
            title = title[:2]
            title[-1] += "..."

        title = "\n".join(title)

        g = sns.catplot(
            data=result,
            kind="bar",
            y="label",
            x="activation",
            color="#abc9ea",
            alpha=0.8,
            height=6,
        )
        g.set(xlabel=None)
        g.set(ylabel=None)
        g.set(title=title)
        g.set(xlim=(0, 1))

        # Add some margin so that the title is not cut.
        g.fig.subplots_adjust(top=0.90)

        out_path = Path(tempfile.mkdtemp()) / "out.png"
        plt.savefig(out_path)

        # Clean-up.
        if url:
            audio.unlink()

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
