models_effnet = {
    "genre_discogs400": "genre_discogs400-discogs-effnet-1.pb",
    "mtg_jamendo_genre": "mtg_jamendo_genre-discogs-effnet-1.pb",
    "mood_acoustic": "mood_acoustic-discogs-effnet-1.pb",
    "mood_aggressive": "mood_aggressive-discogs-effnet-1.pb",
    "mood_electronic": "mood_electronic-discogs-effnet-1.pb",
    "mood_happy": "mood_happy-discogs-effnet-1.pb",
    "mood_party": "mood_party-discogs-effnet-1.pb",
    "mood_relaxed": "mood_relaxed-discogs-effnet-1.pb",
    "mood_sad": "mood_sad-discogs-effnet-1.pb",
    "mtg_jamendo_moodtheme": "mtg_jamendo_moodtheme-discogs-effnet-1.pb",
    "approachability_regression": "approachability_regression-discogs-effnet-1.pb",
    "danceability": "danceability-discogs-effnet-1.pb",
    "engagement_regression": "engagement_regression-discogs-effnet-1.pb",
    "gender": "gender-discogs-effnet-1.pb",
    "mtt_autotagging": "mtt-discogs-effnet-1.pb",
    "mtg_jamendo_instrument": "mtg_jamendo_instrument-discogs-effnet-1.pb",
    "timbre": "timbre-discogs-effnet-1.pb",
    "tonal_atonal": "tonal_atonal-discogs-effnet-1.pb",
    "voice_instrumental": "voice_instrumental-discogs-effnet-1.pb",
}


models_musicnn = {
    "gender": "gender-msd-musicnn-1.pb",
    "mood_sad": "mood_sad-msd-musicnn-1.pb",
    "voice_instrumental": "voice_instrumental-msd-musicnn-1.pb",
    "msd": "msd-msd-musicnn-1.pb",
    "tonal_atonal": "tonal_atonal-msd-musicnn-1.pb",
    "mood_relaxed": "mood_relaxed-msd-musicnn-1.pb",
    "emomusic": "emomusic-msd-musicnn-1.pb",
    "moods_mirex": "moods_mirex-msd-musicnn-1.pb",
    "fs_loop_ds": "fs_loop_ds-msd-musicnn-1.pb",
    "mood_acoustic": "mood_acoustic-msd-musicnn-1.pb",
    "mood_party": "mood_party-msd-musicnn-1.pb",
    "danceability": "danceability-msd-musicnn-1.pb",
    "mood_happy": "mood_happy-msd-musicnn-1.pb",
    "deam": "deam-msd-musicnn-1.pb",
    "muse": "muse-msd-musicnn-1.pb",
    "mood_electronic": "mood_electronic-msd-musicnn-1.pb",
    "mood_aggressive": "mood_aggressive-msd-musicnn-1.pb",
}

models_vggish = {
    "gender": "gender-audioset-vggish-1.pb",
    "mood_sad": "mood_sad-audioset-vggish-1.pb",
    "voice_instrumental": "voice_instrumental-audioset-vggish-1.pb",
    "tonal_atonal": "tonal_atonal-audioset-vggish-1.pb",
    "mood_relaxed": "mood_relaxed-audioset-vggish-1.pb",
    "emomusic": "emomusic-audioset-vggish-1.pb",
    "moods_mirex": "moods_mirex-audioset-vggish-1.pb",
    "mood_acoustic": "mood_acoustic-audioset-vggish-1.pb",
    "mood_party": "mood_party-audioset-vggish-1.pb",
    "danceability": "danceability-audioset-vggish-1.pb",
    "mood_happy": "mood_happy-audioset-vggish-1.pb",
    "deam": "deam-audioset-vggish-1.pb",
    "muse": "muse-audioset-vggish-1.pb",
    "mood_electronic": "mood_electronic-audioset-vggish-1.pb",
    "mood_aggressive": "mood_aggressive-audioset-vggish-1.pb",
}


models = {
    "effnet-discogs": {
        "name": "discogs-effnet-bs64-1.pb",
        "downstream_models": models_effnet,
        "embedding_layer": "PartitionedCall:1",
    },
    "musicnn-msd": {
        "name": "msd-musicnn-1.pb",
        "downstream_models": models_musicnn,
        "embedding_layer": "model/dense/BiasAdd",
    },
    "vggish-audioset": {
        "name": "audioset-vggish-3.pb",
        "downstream_models": models_vggish,
        "embedding_layer": "model/vggish/embeddings",
    },
}
