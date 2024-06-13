# Prediction interface for Cog ⚙️
# Reference: https://github.com/replicate/cog/blob/main/docs/python.md

import tempfile
import json

from cog import BasePredictor, Input, Path
import youtube_dl
from essentia.standard import (
    MonoLoader,
    TensorflowPredict2D,
    TensorflowPredictMusiCNN,
    TensorflowPredictVGGish,
    TensorflowPredictEffnetDiscogs,
)
import numpy as np

from models import models

MODELS_HOME = "/models"

embedding_models = {
    "effnet-discogs": TensorflowPredictEffnetDiscogs,
    "musicnn-msd": TensorflowPredictMusiCNN,
    "vggish-audioset": TensorflowPredictVGGish,
}


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.sample_rate = 16000

    def predict(
        self,
        audio: Path = Input(
            description="Audio file to process",
            default=None,
        ),
        url: str = Input(
            description="YouTube URL to process (overrides audio input)",
            default="",
        ),
        model_type: str = Input(
            description="Model type (embeddings)",
            default="effnet-discogs",
            choices=[
                "effnet-discogs",
                "musicnn-msd",
                "musicnn-mtt",
                "vggish-audioset",
            ],
        ),
    ) -> Path:
        """Run a single prediction by all models of the selected type"""

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

        # Configure a processing chain based on the selected model type.
        audio = MonoLoader(filename=str(audio), sampleRate=self.sample_rate)()

        model_name = "/models/" + models[model_type]["name"]
        downstream_models = models[model_type]["downstream_models"]
        embedding_layer = models[model_type]["embedding_layer"]

        print(model_name)

        model = embedding_models[model_type](
            graphFilename=model_name,
            output=embedding_layer,
        )
        embeddings = model(audio)

        results = {}
        for downstream_model_name, downstream_model_path in downstream_models.items():
            metadata_file = "/models/" + downstream_model_path.replace(".pb", ".json")
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            classes = metadata["classes"]
            input_name = metadata["schema"]["inputs"][0]["name"]

            for output in metadata["schema"]["outputs"]:
                if output["output_purpose"] == "predictions":
                    output_name = output["name"]

            activations = TensorflowPredict2D(
                graphFilename="/models/" + downstream_model_path,
                input=input_name,
                output=output_name,
            )(embeddings)
            activations_mean = np.mean(activations, axis=0)

            top_class = np.argmax(activations_mean)
            prob = activations_mean[top_class]
            label = classes[top_class]

            results[downstream_model_name] = {
                "prob": prob,
                "label": label,
            }

        title = "# %s\n" % title
        header = "| model | top class | activation |\n"
        bar = "|---|---|---|\n"
        table = title + header + bar

        rows = []
        for ds_model_name, values in results.items():
            rows.append(
                f"| {ds_model_name} | {values['label']} | {values['prob']:.2f} |\n"
            )

        table += "".join(rows)

        out_path = Path(tempfile.mkdtemp()) / "out.md"
        with open(out_path, "w") as f:
            f.write(table)
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
