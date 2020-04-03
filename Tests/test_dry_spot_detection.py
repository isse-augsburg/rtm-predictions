import shutil
import unittest

import Resources.testing as resources
from Utils.dry_spot_detection_leoben import dry_spot_analysis, create_triangle_mesh


class TestDrySpotDetectionLeoben(unittest.TestCase):
    def setUp(self):
        self.p = resources.test_pipeline_dir / 'dry_spots'
        self.output = resources.test_out_dir / 'output_dry_spots'
        self.silent = True

    def test_dry_spots(self):
        first_occurrences = [203, 69, 31, 108, 37, 171, 185, 117, 208, 71, 108]
        a = list(self.p.glob('**/*.erfh5'))
        for entry in a:
            output_dir = self.output / entry.parent.stem
            index = int(entry.parent.stem)
            output_dir.mkdir(parents=True, exist_ok=True)
            Xi, Yi, triang, xi, yi = create_triangle_mesh(entry)
            spot_list_s, spot_list_e, deltas_prob = dry_spot_analysis(entry, triang, Xi, Yi, xi, yi, silent=self.silent,
                                                                      output_dir_imgs=output_dir,
                                                                      save_flowfront_img=False)
            if len(spot_list_s) == 0:
                if not self.silent:
                    print(f'Wrong index: should be {first_occurrences[index]}, is {spot_list_s}, '
                          f'found following deltas: {deltas_prob}; which is wanted behaviour here.')
                continue
            first_occurr = spot_list_s[0]
            self.assertAlmostEqual(first_occurr, first_occurrences[index], delta=4)

    def tearDown(self):
        shutil.rmtree(self.output)
