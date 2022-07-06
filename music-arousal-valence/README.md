# Regression of musical arousal and valence values

This demo runs a series of transfer learning regression models trained to predict musical arousal and valence values.
These classifiers were trained on a mixture of public and in-house MTG datasets.

## Source models
- [MusiCNN](https://arxiv.org/abs/1909.06654). A musically motivated CNN with two variants trained on the Million Song Dataset and the MagnaTagATune. 
- [VGGish](https://ieeexplore.ieee.org/abstract/document/7952132). A large VGG variant trained on a preliminary version of the AudioSet Dataset.

## Transfer learning classifiers 
Our models consist of single-hidden-layer MLPs trained on the considered embeddings.

## License
These models are part of [Essentia Models](https://essentia.upf.edu/models.html) made by [MTG-UPF](https://www.upf.edu/web/mtg/) and are publicly available under [CC by-nc-sa](https://creativecommons.org/licenses/by-nc-sa/4.0/) and commercial license.
