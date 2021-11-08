# effnet-discogs

effnet-discogs is an [EfficientNet](https://arxiv.org/abs/1905.11946) architecture trained to predict music styles for 400 of the most popular [Discogs music styles](https://blog.discogs.com/en/genres-and-styles/).

This model was trained in more than two million music recordings from an in-house dataset annotated by Discogs metadata.

The architecture consists in an EfficientNet on its B0 configuration with an additional penultimate dense layer plus batch normalization to facilitate using the model as an embedding extractor.

This demo outputs the the `top_n` music style activations summarized as their mean and standard deviation through time.
