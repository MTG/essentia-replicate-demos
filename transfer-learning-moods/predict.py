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
            Model(name="Mood Acoustic", labels=["Acoustic", "Not acoustic"],
                model_files={'MusiCNN MSD': 'mood_acoustic-musicnn-msd-2.pb', 
                             'MusiCNN MTT': 'mood_acoustic-musicnn-mtt-2.pb', 
                             'VGGish': 'mood_acoustic-vggish-audioset-1.pb'}),
            Model(name="Mood Electronic", labels=["Electronic", "Not electronic"],
                    model_files={'MusiCNN MSD': 'mood_electronic-musicnn-msd-2.pb', 
                                'MusiCNN MTT': 'mood_electronic-musicnn-mtt-2.pb', 
                                'VGGish': 'mood_electronic-vggish-audioset-1.pb'}),
            Model(name="Mood Aggressive", labels=["Aggressive", "Not aggressive"],
                model_files={'MusiCNN MSD': 'mood_aggressive-musicnn-msd-2.pb', 
                                'MusiCNN MTT': 'mood_aggressive-musicnn-mtt-2.pb', 
                                'VGGish': 'mood_aggressive-vggish-audioset-1.pb'}),
            Model(name="Mood Relaxed", labels=["Not relaxed", "Relaxed"],
                model_files={'MusiCNN MSD': 'mood_relaxed-musicnn-msd-2.pb', 
                                'MusiCNN MTT': 'mood_relaxed-musicnn-mtt-2.pb', 
                                'VGGish': 'mood_relaxed-vggish-audioset-1.pb'}),
            Model(name="Mood Happy", labels=["Happy", "Not happy"],
                model_files={'MusiCNN MSD': 'mood_happy-musicnn-msd-2.pb', 
                                'MusiCNN MTT': 'mood_happy-musicnn-mtt-2.pb', 
                                'VGGish': 'mood_happy-vggish-audioset-1.pb'}),
            Model(name="Mood Sad", labels=["Not sad", "Sad"],
                model_files={'MusiCNN MSD': 'mood_sad-musicnn-msd-2.pb',
                                'MusiCNN MTT': 'mood_sad-musicnn-mtt-2.pb',
                              'VGGish': 'mood_sad-vggish-audioset-1.pb'}),
            Model(name="Mood Party", labels=["Not party", "Party"],
                model_files={'MusiCNN MSD': 'mood_party-musicnn-msd-2.pb',
                                'MusiCNN MTT': 'mood_party-musicnn-mtt-2.pb',
                              'VGGish': 'mood_party-vggish-audioset-1.pb'})
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
