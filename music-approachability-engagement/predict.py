# Prediction interface for Cog ⚙️
# Reference: https://github.com/replicate/cog/blob/main/docs/python.md

import tempfile

from cog import BasePredictor, Input, Path
from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs, TensorflowPredict
from essentia import Pool
import numpy as np
import youtube_dl

from models import models

MODELS_HOME = Path("/models")


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory and create the Essentia network for predictions"""

        self.model = str(MODELS_HOME / "discogs-effnet-bs64-1.pb")
        self.input = "model/Placeholder"
        self.output = "model/Softmax"
        self.sample_rate = 16000

        self.pool = Pool()
        self.loader = MonoLoader()
        self.tensorflowPredictEffnetDiscogs = TensorflowPredictEffnetDiscogs(
            graphFilename=self.model,
            output="PartitionedCall:1",
        )

        # Algorithms for specific models.
        self.classifiers = {}
        model_type = "mlp-effnet_b0_3M"
        for model in models:
            modelFilename = str(MODELS_HOME / f"{model['name']}-{model_type}.pb")
            self.classifiers[model["name"]] = TensorflowPredict(
                graphFilename=modelFilename,
                inputs=[self.input],
                outputs=[self.output],
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

        # resize embedding in a tensor
        embeddings = np.expand_dims(embeddings, (1, 2))
        self.pool.set(self.input, embeddings)

        title = "# %s\n" % title
        header = "| model | class | activation |\n"
        bar = "|---|---|---|\n"
        table = title + header + bar

        # predict with each model
        print("running classification heads...")
        for model in models:
            results = self.classifiers[model["name"]](self.pool)[self.output]
            average = np.mean(results.squeeze(), axis=0)

            labels = []
            activations = []

            top_class = np.argmax(average)
            for i, label in enumerate(model["labels"]):
                labels.append(label)
                if i == top_class:
                    activations.append(f"**{average[i]:.2f}**")
                else:
                    activations.append(f"{average[i]:.2f}")

            labels = "<br>".join(labels)
            activations = "<br>".join(activations)

            table += f"{model['name']} | {labels} | {activations}\n"
            if model != models[-1]:
                table += "||<hr>|<hr>|\n"  # separator for readability

        out_path = Path(tempfile.mkdtemp()) / "out.md"
        with open(out_path, "w") as f:
            f.write(table)

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
