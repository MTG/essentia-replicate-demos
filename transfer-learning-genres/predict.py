# Prediction interface for Cog ⚙️
# Reference: https://github.com/replicate/cog/blob/main/docs/python.md

import os
from dataclasses import dataclass
from typing import Dict, List
import tempfile
from pathlib import Path

import cog

from essentia.standard import MonoLoader, TensorflowPredictMusiCNN, TensorflowPredictVGGish
import numpy as np


MODELS_HOME = "/models"


@dataclass
class Model:
    name: str
    labels: List[str]
    model_files: Dict[str, str]



class Predictor(cog.Predictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""

    def predict_single_model(self, audio: str, model: Model):
        sr = 16000
        audio = MonoLoader(filename=str(audio), sampleRate=sr)()

        all_results = {}
        for name, file in model.model_files.items():
            if 'musicnn' in name.lower():
                classifier = TensorflowPredictMusiCNN(graphFilename=os.path.join(MODELS_HOME, file))
            elif 'vggish' in name.lower():
                classifier = TensorflowPredictVGGish(graphFilename=os.path.join(MODELS_HOME, file))
            else:
                classifier = None
            if not classifier:
                raise Exception("Unknown classifier")

            results = classifier(audio)
            averaged_predictions = np.mean(results, axis=0)
            all_results[name] = averaged_predictions

        result_order = list(all_results.keys())
        header = "|  | " + " | ".join(result_order)
        bar = "|---|" + "|".join(["---"] * len(all_results.keys()))
        table = header + "\n" + bar + "\n"
        for i, label in enumerate(model.labels):
            line = f"{label} | "
            for classifier in result_order:
                line += f" {all_results[classifier][i]:.2f} | "
            table += line + "\n"
        
        return table

    @cog.input("audio", type=cog.Path, help="Audio file to process")
    def predict(self, audio):

        models = [
            Model(name="Genre Dortmund", labels=["alternative", "blues", "electronic", "folkcountry", "funksoulrnb", "jazz", "pop", "raphiphop", "rock"],
                model_files={'MusiCNN MSD': 'genre_dortmund-musicnn-msd-2.pb', 
                             'MusiCNN MTT': 'genre_dortmund-musicnn-mtt-2.pb', 
                             'VGGish': 'genre_dortmund-vggish-audioset-1.pb'}),
            Model(name="Genre Rosamerica", labels=["classic", "dance", "hip hop", "jazz", "pop", "rhythm and blues", "rock", "speech"],
                    model_files={'MusiCNN MSD': 'genre_rosamerica-musicnn-msd-2.pb', 
                                'MusiCNN MTT': 'genre_rosamerica-musicnn-mtt-2.pb', 
                                'VGGish': 'genre_rosamerica-vggish-audioset-1.pb'}),
            Model(name="Genre Tzanetakis", labels=["blues", "classic", "country", "disco", "hip hop", "jazz", "metal", "pop", "reggae", "rock"],
                model_files={'MusiCNN MSD': 'genre_tzanetakis-musicnn-msd-2.pb', 
                                'MusiCNN MTT': 'genre_tzanetakis-musicnn-mtt-2.pb', 
                                'VGGish': 'genre_tzanetakis-vggish-audioset-1.pb'}),
            Model(name="Genre Electronic", labels=["ambient", "dnb", "house", "techno", "trance"],
                model_files={'MusiCNN MSD': 'genre_electronic-musicnn-msd-2.pb', 
                                'MusiCNN MTT': 'genre_electronic-musicnn-mtt-2.pb', 
                                'VGGish': 'genre_electronic-vggish-audioset-1.pb'})
        ]

        results = ""
        for model in models:
            results += f"# {model.name}\n\n"
            model_table = self.predict_single_model(audio, model)
            results += model_table + "\n\n"

        out_path = Path(tempfile.mkdtemp()) / "out.md"
        with open(out_path, "w") as f:
            f.write(results)
        return out_path
