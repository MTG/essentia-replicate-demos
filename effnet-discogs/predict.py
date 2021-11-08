# Prediction interface for Cog ⚙️
# Reference: https://github.com/replicate/cog/blob/main/docs/python.md

from pathlib import Path
import tempfile
from itertools import chain

import cog
from essentia.streaming import (
    MonoLoader,
    FrameCutter,
    VectorRealToTensor,
    TensorToPool,
    TensorflowInputMusiCNN,
    TensorflowPredict,
    PoolToTensor,
    TensorToVectorReal,
)
from essentia import Pool, run
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas
import youtube_dl

from labels import labels


# Remove the broader genre label for readability
labels = [l.split("---")[-1] for l in labels]


class Predictor(cog.Predictor):
    def setup(self):
        """Load the model into memory and create the Essentia network for predictions"""

        self.model = "models/210402-122138_dev-multi-node.pb"  # Only supported variant for now. It's a development version!
        self.input = "melspectrogram"
        self.output = "activations"
        self.sample_rate = 16000
        self.patch_size = 128
        self.batch_size = 32
        self.patch_hop_size = 65  # so that the activation rate is 1 Hz
        self.nbands = 96
        self.frame_size = 512
        self.hop_size = 256

        # Most Essentia TensorFlow models have a dedicated wrapper algorithm to simplify inference.
        # While this is a WIP for effnet-discogs, this is am opportunity to learn a bit of
        # Essentia's streaming mode, which prevents copying intermediate outputs to Python making
        # it more efficient among other benefits.
        self.pool = Pool()
        self.loader = MonoLoader(sampleRate=self.sample_rate)
        self.frameCutter = FrameCutter(
            frameSize=self.frame_size,
            hopSize=self.hop_size,
            silentFrames="keep",
        )
        self.melSpectrogram = TensorflowInputMusiCNN()
        self.vectorRealToTensor = VectorRealToTensor(
            shape=[self.batch_size, 1, self.patch_size, self.nbands],
            patchHopSize=self.patch_hop_size,
        )
        self.tensorToPool = TensorToPool(namespace=self.input)
        self.tensorflowPredict = TensorflowPredict(
            graphFilename=self.model,
            inputs=[self.input],
            outputs=[self.output],
        )
        self.poolToTensor = PoolToTensor(namespace=self.output)
        self.tensorToVectorReal = TensorToVectorReal()

        self.loader.audio >> self.frameCutter.signal
        self.frameCutter.frame >> self.melSpectrogram.frame
        self.melSpectrogram.bands >> self.vectorRealToTensor.frame
        self.vectorRealToTensor.tensor >> self.tensorToPool.tensor
        self.tensorToPool.pool >> self.tensorflowPredict.poolIn
        self.tensorflowPredict.poolOut >> self.poolToTensor.pool
        self.poolToTensor.tensor >> self.tensorToVectorReal.tensor
        self.tensorToVectorReal.frame >> (self.pool, self.output)

    @cog.input(
        "filename",
        type=cog.Path,
        default=None,
        help="Input audio path",
    )
    @cog.input(
        "url",
        type=str,
        default=None,
        help="YouTube URL to process",
    )
    @cog.input(
        "top_n",
        type=int,
        default=10,
        help="Top n music styles to show",
    )
    @cog.input(
        "output_format",
        type=str,
        default="Visualization",
        options=["Visualization", "JSON"],
        help="Output either a bar chart visualization or a JSON blob",
    )
    def predict(self, filename, url, top_n, output_format):
        """Run a single prediction on the model"""

        assert filename or url, "A filename or a YouTube URL should be specified"

        # If there is a YouTube url use that.
        if url:
            filename = self._download(url)

        print("loading and resampling audio...")

        self.loader.configure(sampleRate=self.sample_rate, filename=str(filename))
        run(self.loader)

        print("computing activations...")
        activations = self.pool[self.output]
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

        out_path = Path(tempfile.mkdtemp()) / "out.png"
        plt.savefig(out_path)

        # Clean-up.
        if url:
            filename.unlink()

        print("done!")
        return out_path

    def _download(self, url, ext="mp3"):
        """Download a YouTube URL in the specified format to a temporal directory"""

        tmp_dir = Path(tempfile.mktemp())
        ydl_opts = {
            "format": "bestaudio/best",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": ext,
                    "preferredquality": "192",
                }
            ],
            "outtmpl": str(tmp_dir / "%(id)s.%(ext)s"),
        }

        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        url_id = url.split("=")[-1]
        paths = [p for p in tmp_dir.glob(f"*{url_id}.{ext}")]

        assert (
            len(paths) == 1
        ), "Something unexpected happened. Should be only one match!"

        return paths[0]
