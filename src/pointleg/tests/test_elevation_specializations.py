"""Check the supported elevation task entries after retiring single directions."""

from pathlib import Path
import unittest

from mjlab.tasks.registry import list_tasks, load_env_cfg
import pointleg
from pointleg.evaluate_elevation import evaluation_cases
from pointleg.terrains.elevation import ELEVATION_TASK_TERRAINS
from pointleg.transfer import prepare_transfer


class ElevationEntryTests(unittest.TestCase):
    def test_only_paired_and_mixed_entries_are_available(self):
        expected = {"PointLeg-Stairs-v0", "PointLeg-Slope-v0", "PointLeg-Elevation-v0"}
        self.assertEqual(set(ELEVATION_TASK_TERRAINS), expected)
        self.assertTrue(expected <= set(list_tasks()))
        self.assertTrue(expected <= set(pointleg.TASK_SPECS))
        self.assertEqual(len(evaluation_cases("PointLeg-Stairs-v0")), 10)
        self.assertEqual(len(evaluation_cases("PointLeg-Slope-v0")), 10)
        self.assertEqual(len(evaluation_cases("PointLeg-Elevation-v0")), 21)

    def test_retired_tasks_cannot_load_evaluate_or_transfer(self):
        config = Path(pointleg.__file__).parent / "config"
        for suffix, module in (("StairsUp", "stairs_up"), ("StairsDown", "stairs_down"),
                               ("SlopeUp", "slope_up"), ("SlopeDown", "slope_down")):
            task = "PointLeg-" + suffix + "-v0"
            with self.subTest(task=task):
                self.assertNotIn(task, list_tasks())
                self.assertNotIn(task, pointleg.TASK_SPECS)
                self.assertFalse((config / (module + ".py")).exists())
                with self.assertRaises(KeyError):
                    load_env_cfg(task)
                with self.assertRaises(ValueError):
                    evaluation_cases(task)
                with self.assertRaises(ValueError):
                    prepare_transfer(task, "unused.pt")


if __name__ == "__main__":
    unittest.main()
