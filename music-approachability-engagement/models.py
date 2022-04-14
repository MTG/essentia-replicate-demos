# define the model labels for approachability and engagement per model_type
models = {
    "effnet-discogs-test-2class":{
        "approachability": {
            "name": "approachability-2class",
            "labels": [
                "not approachable",
                "approachable",
            ],
        },
        "engagement": {
            "name": "engagement-2class",
            "labels": [
                "not engaging",
                "engaging",
            ],
        },
    },
    "effnet-discogs-test-3class":{
        "approachability": {
            "name": "approachability-3class",
            "labels": [
                "low",
                "mid",
                "high",
            ],
        },
        "engagement": {
            "name": "engagement-3class",
            "labels": [
                "low",
                "mid",
                "high",
            ],
        },
    },
    "effnet-discogs-test-regression":{
        "approachability": {
            "name": "approachability-regression",
            "labels": [
                "mean",
                "std",
            ],
        },
        "engagement": {
            "name": "engagement-regression",
            "labels": [
                "mean",
                "std",
            ],
        },
    }
}
