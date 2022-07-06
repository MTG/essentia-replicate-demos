# Transfer learning approachability and engagement

This demo runs transfer learning models to estimate music approachability and engagement using effnet-discogs embeddings. We include 3 different models per task to provide the different outcome format: multi-class, binary and regression.

These classifiers were trained on in-house MTG datasets.

## Source models
[effnet-discogs](https://essentia.upf.edu/models/feature-extractors/discogs-effnet/) is an [EfficientNet](https://arxiv.org/abs/1905.11946) architecture trained to predict music styles for 400 of the most popular Discogs music styles.

## Transfer Learning models
Our [models](https://essentia.upf.edu/models/classification-heads/) consist of single-hidden-layer MLPs trained on the considered embeddings.

## License
These models are part of [Essentia Models](https://essentia.upf.edu/models.html) made by [MTG-UPF](https://www.upf.edu/web/mtg/) and are publicly available under (CC by-nc-sa)[https://creativecommons.org/licenses/by-nc-sa/4.0/] and commercial license.
