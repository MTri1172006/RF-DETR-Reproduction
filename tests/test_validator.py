"""Unit tests for shared NAS Search Space Validator."""

import unittest

from rfdetr_reproduction.nas.validator import validate_nas_config


class TestNASValidator(unittest.TestCase):
    def test_valid_train_config(self):
        cfg = {
            "resolution": 512,
            "patch_size": 16,
            "num_windows": 2,
            "decoder_layers": 6,
            "num_queries": 300,
        }
        ok, reason = validate_nas_config(cfg, mode="train", max_tokens=5776)
        self.assertTrue(ok)
        self.assertEqual(reason, "ok")

    def test_reject_non_divisible_patch_size(self):
        cfg = {
            "resolution": 512,
            "patch_size": 14,
            "num_windows": 2,
            "decoder_layers": 6,
            "num_queries": 300,
        }
        ok, reason = validate_nas_config(cfg, mode="inference")
        self.assertFalse(ok)
        self.assertEqual(reason, "resolution_not_divisible_by_patch_size")

    def test_reject_token_grid_not_divisible_by_windows(self):
        cfg = {
            "resolution": 320,
            "patch_size": 32,
            "num_windows": 4,  # grid_size = 10, 10 % 4 != 0
            "decoder_layers": 6,
            "num_queries": 300,
        }
        ok, reason = validate_nas_config(cfg, mode="train")
        self.assertFalse(ok)
        self.assertEqual(reason, "token_grid_not_divisible_by_num_windows")

    def test_reject_exceeds_token_budget(self):
        cfg = {
            "resolution": 960,
            "patch_size": 8,  # 120^2 = 14,400 > 5,776
            "num_windows": 4,
            "decoder_layers": 6,
            "num_queries": 300,
        }
        ok, reason = validate_nas_config(cfg, mode="train", max_tokens=5776)
        self.assertFalse(ok)
        self.assertEqual(reason, "exceeds_token_budget(>5776)")

    def test_reject_train_mode_wrong_decoder_or_queries(self):
        cfg_bad_dl = {
            "resolution": 384,
            "patch_size": 16,
            "num_windows": 2,
            "decoder_layers": 2,
            "num_queries": 300,
        }
        ok, reason = validate_nas_config(cfg_bad_dl, mode="train")
        self.assertFalse(ok)
        self.assertEqual(reason, "train_decoder_layers_must_be_6")

        cfg_bad_nq = {
            "resolution": 384,
            "patch_size": 16,
            "num_windows": 2,
            "decoder_layers": 6,
            "num_queries": 100,
        }
        ok, reason = validate_nas_config(cfg_bad_nq, mode="train")
        self.assertFalse(ok)
        self.assertEqual(reason, "train_queries_must_be_300")

    def test_valid_inference_zero_decoder_layers(self):
        cfg = {
            "resolution": 448,
            "patch_size": 14,
            "num_windows": 4,
            "decoder_layers": 0,
            "num_queries": 50,
        }
        ok, reason = validate_nas_config(cfg, mode="inference")
        self.assertTrue(ok)
        self.assertEqual(reason, "ok")


if __name__ == "__main__":
    unittest.main()
