"""Unit tests for AdaptiveNASSampler (uniform, pruned_anchor, sandwich)."""

import unittest

from configs.nas_config import NAS_CONFIG
from rfdetr_reproduction.nas.sampling import AdaptiveNASSampler
from rfdetr_reproduction.nas.search_space import build_train_search_space


class TestAdaptiveNASSampler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.valid_train_configs, _ = build_train_search_space()

    def test_uniform_samples_across_full_space(self):
        sampler = AdaptiveNASSampler.from_config(
            self.valid_train_configs, NAS_CONFIG, strategy="uniform", seed=42
        )
        seen_patch_sizes = {sampler.sample(epoch=1, micro_step_idx=i)["patch_size"] for i in range(500)}
        seen_windows = {sampler.sample(epoch=1, micro_step_idx=i)["num_windows"] for i in range(500)}
        self.assertIn(32, seen_patch_sizes)
        self.assertIn(4, seen_windows)

    def test_pruned_anchor_drops_p32_and_w4_and_respects_two_phases(self):
        sampler = AdaptiveNASSampler.from_config(
            self.valid_train_configs, NAS_CONFIG, strategy="pruned_anchor", seed=42
        )
        # Phase 1 (Ep 1..8): only pruned patch_sizes (12,14,16) and num_windows (1,2)
        for step in range(400):
            cfg = sampler.sample(epoch=4, micro_step_idx=step)
            self.assertIn(cfg["patch_size"], (12, 14, 16))
            self.assertIn(cfg["num_windows"], (1, 2))
            self.assertNotEqual(cfg["patch_size"], 32)
            self.assertNotEqual(cfg["num_windows"], 4)

        # Phase 2 (Ep 9..20): ~68% core Pareto (res in 512..704, p16) and ~32% anchor (res in 320..448)
        n_samples = 2000
        n_core = 0
        for step in range(n_samples):
            cfg = sampler.sample(epoch=15, micro_step_idx=step)
            self.assertIn(cfg["patch_size"], (12, 14, 16))
            self.assertIn(cfg["num_windows"], (1, 2))
            if cfg["resolution"] in (512, 576, 640, 704) and cfg["patch_size"] == 16:
                n_core += 1
            else:
                self.assertIn(cfg["resolution"], (320, 384, 448))
        core_ratio = n_core / n_samples
        self.assertGreater(core_ratio, 0.62)
        self.assertLess(core_ratio, 0.74)

    def test_sandwich_micro_batch_slots(self):
        sampler = AdaptiveNASSampler.from_config(
            self.valid_train_configs, NAS_CONFIG, strategy="sandwich", seed=42
        )
        for ep in (3, 15):
            for rep in range(50):
                base = rep * 4
                c0 = sampler.sample(epoch=ep, micro_step_idx=base + 0)
                c1 = sampler.sample(epoch=ep, micro_step_idx=base + 1)
                c2 = sampler.sample(epoch=ep, micro_step_idx=base + 2)
                c3 = sampler.sample(epoch=ep, micro_step_idx=base + 3)

                # Slot 0: large Pareto anchor (res >= 640, p16, w in {1,2})
                self.assertGreaterEqual(c0["resolution"], 640)
                self.assertEqual(c0["patch_size"], 16)
                self.assertIn(c0["num_windows"], (1, 2))

                # Slot 1: small anchor (res <= 384, p16, w in {1,2})
                self.assertLessEqual(c1["resolution"], 384)
                self.assertEqual(c1["patch_size"], 16)
                self.assertIn(c1["num_windows"], (1, 2))

                # Slots 2-3: pruned in Phase 1, core Pareto in Phase 2
                for c_mid in (c2, c3):
                    self.assertIn(c_mid["patch_size"], (12, 14, 16))
                    self.assertIn(c_mid["num_windows"], (1, 2))
                    if ep > 8:
                        self.assertIn(c_mid["resolution"], (512, 576, 640, 704))
                        self.assertEqual(c_mid["patch_size"], 16)

    def test_invalid_strategy_raises(self):
        with self.assertRaises(ValueError):
            AdaptiveNASSampler(self.valid_train_configs, strategy="unknown_mode")


if __name__ == "__main__":
    unittest.main()
