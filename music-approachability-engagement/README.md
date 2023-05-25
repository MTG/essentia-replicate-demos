# Classification of music approachability and engagement

This demo runs transfer learning models to estimate music approachability and engagement using *effnet-discogs* embeddings. We include three model types, providing different outcome formats: `two classes`, `three classes` and `regression` with continuous values:

* **two classes**:  low, and high approachability and engagement.
* **three classes**: low, mid, and high approachability and engagement.
* **regression**: continuous values of approachability and engagement from 0 (low) to 1 (high).


These classifiers were trained on in-house MTG datasets.

## Source models
*effnet-discogs* is an EfficientNet architecture trained to predict music styles for 400 of the most popular Discogs music styles.

## Transfer learning models
Our models consist of single-hidden-layer MLPs trained on the considered embeddings.

## License
These models are part of [Essentia Models](https://essentia.upf.edu/models.html) made by [MTG-UPF](https://www.upf.edu/web/mtg/) and are publicly available under [CC by-nc-sa](https://creativecommons.org/licenses/by-nc-sa/4.0/) and commercial license.
