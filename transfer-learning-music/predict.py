# Prediction interface for Cog ⚙️
# Reference: https://github.com/replicate/cog/blob/main/docs/python.md

from pathlib import Path
import os
import tempfile

import cog
import youtube_dl
from essentia.streaming import (
    MonoLoader,
    FrameCutter,
    VectorRealToTensor,
    TensorToPool,
    TensorflowInputMusiCNN,
    TensorflowInputVGGish,
    TensorflowPredict,
    PoolToTensor,
    TensorToVectorReal,
)
from essentia import Pool, run
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas


MODELS_HOME = "/models"

# TODO Factor out the list of models to a separate file.
models = [
    {
        'name': 'genre_dortmund',
        'labels': ["alternative", "blues", "electronic", "folkcountry", "funksoulrnb", "jazz", "pop", "raphiphop", "rock"]
    },
    {
        'name': 'genre_rosamerica',
        'labels': ["classic", "dance", "hip hop", "jazz", "pop", "rhythm and blues", "rock", "speech"]
    },
    {
        'name': 'genre_tzanetakis',
        'labels': ["blues", "classic", "country", "disco", "hip hop", "jazz", "metal", "pop", "reggae", "rock"]
    },
    {
        'name': 'genre_electronic',
        'labels': ["ambient", "dnb", "house", "techno", "trance"]
    },
    {
        'name': 'mood_acoustic',
        'labels': ["Acoustic", "Not acoustic"]
    },
    {
        'name': 'mood_electronic',
        'labels': ["Electronic", "Not electronic"]
    },
    {
        'name': 'mood_aggressive',
        'labels': ["Aggressive", "Not aggressive"]
    },
    {
        'name': 'mood_relaxed',
        'labels': ["Not relaxed", "Relaxed"]
    },
    {
        'name': 'mood_happy',
        'labels': ["Happy", "Not happy"],
    },
    {
        'name': 'mood_sad',
        'labels': ["Not sad", "Sad"]
    },
    {
        'name': 'mood_party',
        'labels': ["Not party", "Party"]
    },
    # TODO Add missing models.
]


class Predictor(cog.Predictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.sample_rate = 16000
        self.frame_size = 512
        self.hop_size = 256

        self.input = "melspectrogram"
        self.output = "activations"

    @cog.input(
        "audio",
        type=cog.Path,
        default=None,
        help="Audio file to process",
    )
    @cog.input(
        "url",
        type=str,
        default=None,
        help="YouTube URL to process",
    )
    @cog.input(
        "model_type",
        type=str,
        default="musicnn-msd-2",
        options=["musicnn-msd-2", "musicnn-mtt-2", "vggish-audioset-1"],
        help="Model type (embeddings)",
    )
    @cog.input(
        "output_format",
        type=str,
        default="Visualization",
        options=["Visualization", "JSON"],
        help="Output either a bar chart visualization or a JSON blob",
    )
    def predict(self, audio, url, model_type, output_format):
        """Run a single prediction by all models of the selected type"""

        assert audio or url, "A filename or a YouTube URL should be specified"

        # If there is a YouTube url use that.
        if url:
            audio = self._download(url)

        # Configure a processing chain based on the selected model type.
        pool = Pool()
        loader = MonoLoader(filename=str(audio), sampleRate=self.sample_rate)

        patch_hop_size = 0  # No overlap for efficiency
        batch_size = 256

        if model_type in ["musicnn-msd-2", "musicnn-mtt-2"]:
            frame_size = 512
            hop_size = 256
            patch_size = 187
            nbands = 96
            melSpectrogram = TensorflowInputMusiCNN()
        elif model_type in ["vggish-audioset-1"]:
            frame_size = 400
            hop_size = 160
            patch_size = 96
            nbands = 64
            melSpectrogram = TensorflowInputVGGish()

        frameCutter = FrameCutter(
            frameSize=frame_size,
            hopSize=hop_size,
            silentFrames="keep",
        )
        vectorRealToTensor = VectorRealToTensor(
            shape=[batch_size, 1, patch_size, nbands],
            patchHopSize=patch_hop_size,
        )
        tensorToPool = TensorToPool(namespace='model/Placeholder')

        # Algorithms for specific models.
        tensorflowPredict = {}
        poolToTensor = {}
        tensorToVectorReal = {}

        for model in models:
            modelFilename = '/models/%s-%s.pb' % (model['name'], model_type)
            tensorflowPredict[model['name']] = \
                TensorflowPredict(graphFilename=modelFilename,
                                  inputs=['model/Placeholder'],
                                  outputs=['model/Sigmoid'])
            poolToTensor[model['name']] = PoolToTensor(namespace='model/Sigmoid')
            tensorToVectorReal[model['name']] = TensorToVectorReal()

        loader.audio >> frameCutter.signal
        frameCutter.frame >> melSpectrogram.frame
        melSpectrogram.bands >> vectorRealToTensor.frame
        vectorRealToTensor.tensor >> tensorToPool.tensor

        # TODO Ugly. Loop through a simple list of model names instead.
        for model in models:
            tensorToPool.pool >> tensorflowPredict[model['name']].poolIn
            tensorflowPredict[model['name']].poolOut >> poolToTensor[model['name']].pool
            poolToTensor[model['name']].tensor >> tensorToVectorReal[model['name']].tensor
            tensorToVectorReal[model['name']].frame >> (pool, 'activations.%s' % model['name'])

        print("running the inference network...")
        run(loader)

        header = "| class | activation |"
        bar = "|---|---|"
        table = header + "\n" + bar + "\n"
        for model in models:
            average = np.mean(pool["activations.%s" % model["name"]], axis=0)
            for i, label in enumerate(model["labels"]):
                # Do not plot negative labels
                if label.startswith("Not"):
                    continue
                table += f"{label} | {average[i]:.2f}\n"
        return table

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
            ydl.download([url])

        paths = [p for p in tmp_dir.glob(f"audio.{ext}")]
        assert (
            len(paths) == 1
        ), "Something unexpected happened. Should be only one match!"

        return paths[0]
