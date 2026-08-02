from __future__ import annotations

import unittest

from experiments.gave2_v12.folds import validate_manifest


class FoldTests(unittest.TestCase):
    def test_three_fold_manifest_has_no_leakage(self):
        ids = [f"g_{index:03d}" for index in range(1, 51)]
        chunks = (ids[:17], ids[17:34], ids[34:])
        manifest = {
            "n_folds": 3,
            "folds": [
                {
                    "training": [case_id for case_id in ids if case_id not in set(validation)],
                    "validation": list(validation),
                }
                for validation in chunks
            ],
        }
        validate_manifest(manifest, ids)

    def test_duplicate_validation_case_is_rejected(self):
        ids = [f"g_{index:03d}" for index in range(1, 51)]
        manifest = {
            "n_folds": 3,
            "folds": [
                {"training": ids[17:], "validation": ids[:17]},
                {"training": ids[:17] + ids[34:], "validation": ids[17:34]},
                {
                    "training": [case_id for case_id in ids if case_id not in set([ids[0], *ids[35:]])],
                    "validation": [ids[0], *ids[35:]],
                },
            ],
        }
        with self.assertRaises(ValueError):
            validate_manifest(manifest, ids)


if __name__ == "__main__":
    unittest.main()
