# effnet-discogs: An EfficientNet for music style classification by 400 styles from the Discogs taxonomy

effnet-discogs is an [EfficientNet](https://arxiv.org/abs/1905.11946) architecture trained to predict music styles for 400 of the most popular [Discogs music styles](https://blog.discogs.com/en/genres-and-styles/). The output plot also shows the Discogs [genre](https://blog.discogs.com/en/genres-and-styles/) the predicted style belongs to.

This model was trained in more than two million music recordings from an in-house dataset annotated by Discogs metadata and is part of an ongoing research.

The architecture consists of an EfficientNet on its B0 configuration with an additional penultimate dense layer plus batch normalization to facilitate using the model as an embedding extractor.

This demo outputs the `top_n` music style activations, summarized as their mean and standard deviation through time.

## License

This model is part of [Essentia Models](https://essentia.upf.edu/models.html) made by [MTG-UPF](https://www.upf.edu/web/mtg/).
