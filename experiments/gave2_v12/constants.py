from __future__ import annotations

EXPECTED_HEIGHT = 1024
EXPECTED_WIDTH = 1536
EXPECTED_TRAINING_CASES = 50
EXPECTED_VALIDATION_CASES = 50
CHANNEL_NAMES = ("artery", "vessel", "vein")
BIOMARKER_KEYS = (
    "CRAE",
    "CRVE",
    "AVR",
    "artery_density",
    "vein_density",
    "artery_fractal_dimension",
    "vein_fractal_dimension",
)

# The public leaderboard is consistent with these Task 1/2 component weights.
LIVE_CLASSIFICATION_WEIGHT = 0.40
LIVE_DICE_WEIGHT = 0.20
LIVE_TOPOLOGY_WEIGHT = 0.40

MINIMA_SOURCE_URL = "https://github.com/LSXI7/MINIMA.git"
MINIMA_LOFTR_URL = (
    "https://github.com/LSXI7/storage/releases/download/MINIMA/minima_loftr.ckpt"
)
MINIMA_LOFTR_SHA256 = "810d19773ff898ba04a68c99a3eff9c112210bf884214bd76aec885e83b0e257"
