import unittest

import Tests.resources_for_testing as resources
from Utils.dry_spot_detection_leoben import dry_spot_analysis


class TestDrySpotDetectionLeoben(unittest.TestCase):
    def setUp(self):
        self.p = resources.test_pipeline_dir / 'dry_spots'
        self.output = resources.test_out_dir / 'output_dry_spots'

    @unittest.skip("Not working in this branch")
    def test_dry_spots(self):
        # first_occurrences = [203, 69, 31, 108, 37, 81, 185, 117, 208, 71, 108]
        a = list(self.p.glob('**/*.erfh5'))
        a = [a[2]]
        for entry in a:
            output_dir = self.output / entry.parent.stem
            # index = int(entry.parent.stem)
            print(entry, output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            spot_list_s, spot_list_e = dry_spot_analysis(entry, output_dir)
            # first_occurr = spot_list_s[0]
            # self.assertEqual(first_occurr, first_occurrences[index])

        # for i, entry in enumerate(self.p.iterdir()):
        #     output_dir = self.output / str(i)
        #     print(entry / f'2019-07-23_15-38-08_{i}_RESULT.erfh5')
        #     output_dir.mkdir(parents=True, exist_ok=True)
        #     dry_spot_analysis(entry, output_dir)

    # def tearDown(self):
    #     shutil.rmtree(self.output)
