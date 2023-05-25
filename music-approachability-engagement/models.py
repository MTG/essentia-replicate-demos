# define the model labels for approachability and engagement per model_type
models = {
    "2 classes":{
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
    "3 classes":{
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
    "regression":{
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
