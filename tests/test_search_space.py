"""Unit tests verifying exact NAS search space counts against the original notebook.

Expected notebook parity:
  - Train (L4 token budget <= 5,776): 111 valid configs (231 raw)
  - Infer (8 listed patch sizes):     3,948 valid configs (7,392 raw)
  - Infer (7 train patch sizes):      3,780 valid configs (6,468 raw)
"""

import unittest

from rfdetr_reproduction.nas.search_space import (
    build_infer_search_space_7ps,
    build_infer_search_space_8ps,
    build_representative_infer_configs,
    build_train_search_space,
)


class TestSearchSpaceCounts(unittest.TestCase):
    def test_train_search_space_exact_count(self):
        raw_train, valid_train, _ = build_train_search_space()
        self.assertEqual(len(raw_train), 231)
        self.assertEqual(len(valid_train), 111)

    def test_infer_8ps_search_space_exact_count(self):
        raw_8ps, valid_8ps, _ = build_infer_search_space_8ps()
        self.assertEqual(len(raw_8ps), 7392)
        self.assertEqual(len(valid_8ps), 3948)

    def test_infer_7ps_search_space_exact_count(self):
        raw_7ps, valid_7ps, _ = build_infer_search_space_7ps()
        self.assertEqual(len(raw_7ps), 6468)
        self.assertEqual(len(valid_7ps), 3780)

    def test_representative_infer_subset_non_empty(self):
        rep = build_representative_infer_configs()
        self.assertGreater(len(rep), 0)
        for cfg in rep:
            self.assertIn("config_id", cfg)
            self.assertIn("grid_size", cfg)
            self.assertIn("num_tokens", cfg)


if __name__ == "__main__":
    unittest.main()
