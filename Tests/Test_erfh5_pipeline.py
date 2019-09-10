import pickle
import unittest
from pathlib import Path

from Pipeline import erfh5_pipeline as pipeline, data_loaders_IMG as dli, \
    data_gather as dg


class TestERFH5Pipeline(unittest.TestCase):
    def setUp(self):
        self.paths = [r'X:\s\t\stiebesi\data\RTM\Leoben\output\with_shapes\2019-07-11_15-14-48_100p']
        self.cached_files = ['X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p\\0'
                             '/2019-07-11_15-14-48_0_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p\\1'
                             '/2019-07-11_15-14-48_1_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\10/2019-07-11_15-14-48_10_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\11/2019-07-11_15-14-48_11_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\12/2019-07-11_15-14-48_12_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\13/2019-07-11_15-14-48_13_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\14/2019-07-11_15-14-48_14_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\15/2019-07-11_15-14-48_15_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\16/2019-07-11_15-14-48_16_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\17/2019-07-11_15-14-48_17_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\18/2019-07-11_15-14-48_18_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\19/2019-07-11_15-14-48_19_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p\\2'
                             '/2019-07-11_15-14-48_2_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\20/2019-07-11_15-14-48_20_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\21/2019-07-11_15-14-48_21_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\22/2019-07-11_15-14-48_22_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\23/2019-07-11_15-14-48_23_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\24/2019-07-11_15-14-48_24_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\25/2019-07-11_15-14-48_25_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\26/2019-07-11_15-14-48_26_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\27/2019-07-11_15-14-48_27_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\28/2019-07-11_15-14-48_28_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\29/2019-07-11_15-14-48_29_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p\\3'
                             '/2019-07-11_15-14-48_3_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\30/2019-07-11_15-14-48_30_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\31/2019-07-11_15-14-48_31_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\32/2019-07-11_15-14-48_32_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\33/2019-07-11_15-14-48_33_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\34/2019-07-11_15-14-48_34_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\35/2019-07-11_15-14-48_35_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\36/2019-07-11_15-14-48_36_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\37/2019-07-11_15-14-48_37_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\38/2019-07-11_15-14-48_38_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\39/2019-07-11_15-14-48_39_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p\\4'
                             '/2019-07-11_15-14-48_4_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\40/2019-07-11_15-14-48_40_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\41/2019-07-11_15-14-48_41_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\42/2019-07-11_15-14-48_42_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\43/2019-07-11_15-14-48_43_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\44/2019-07-11_15-14-48_44_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\45/2019-07-11_15-14-48_45_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\46/2019-07-11_15-14-48_46_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\47/2019-07-11_15-14-48_47_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\48/2019-07-11_15-14-48_48_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\49/2019-07-11_15-14-48_49_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p\\5'
                             '/2019-07-11_15-14-48_5_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\50/2019-07-11_15-14-48_50_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\51/2019-07-11_15-14-48_51_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\52/2019-07-11_15-14-48_52_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\53/2019-07-11_15-14-48_53_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\54/2019-07-11_15-14-48_54_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\55/2019-07-11_15-14-48_55_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\56/2019-07-11_15-14-48_56_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\57/2019-07-11_15-14-48_57_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\58/2019-07-11_15-14-48_58_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\59/2019-07-11_15-14-48_59_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p\\6'
                             '/2019-07-11_15-14-48_6_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\60/2019-07-11_15-14-48_60_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\61/2019-07-11_15-14-48_61_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\62/2019-07-11_15-14-48_62_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\63/2019-07-11_15-14-48_63_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\64/2019-07-11_15-14-48_64_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\65/2019-07-11_15-14-48_65_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\66/2019-07-11_15-14-48_66_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\67/2019-07-11_15-14-48_67_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\68/2019-07-11_15-14-48_68_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\69/2019-07-11_15-14-48_69_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p\\7'
                             '/2019-07-11_15-14-48_7_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\70/2019-07-11_15-14-48_70_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\71/2019-07-11_15-14-48_71_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\72/2019-07-11_15-14-48_72_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\73/2019-07-11_15-14-48_73_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\74/2019-07-11_15-14-48_74_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\75/2019-07-11_15-14-48_75_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\76/2019-07-11_15-14-48_76_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\77/2019-07-11_15-14-48_77_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\78/2019-07-11_15-14-48_78_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\79/2019-07-11_15-14-48_79_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p\\8'
                             '/2019-07-11_15-14-48_8_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\80/2019-07-11_15-14-48_80_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\81/2019-07-11_15-14-48_81_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\82/2019-07-11_15-14-48_82_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\83/2019-07-11_15-14-48_83_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\84/2019-07-11_15-14-48_84_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\85/2019-07-11_15-14-48_85_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\86/2019-07-11_15-14-48_86_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\87/2019-07-11_15-14-48_87_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\88/2019-07-11_15-14-48_88_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\89/2019-07-11_15-14-48_89_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p\\9'
                             '/2019-07-11_15-14-48_9_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\90/2019-07-11_15-14-48_90_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\91/2019-07-11_15-14-48_91_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\92/2019-07-11_15-14-48_92_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\93/2019-07-11_15-14-48_93_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\94/2019-07-11_15-14-48_94_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\95/2019-07-11_15-14-48_95_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\96/2019-07-11_15-14-48_96_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\97/2019-07-11_15-14-48_97_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\98/2019-07-11_15-14-48_98_RESULT.erfh5',
                             'X:\\s\\t\\stiebesi\\data\\RTM\\Leoben\\output\\with_shapes\\2019-07-11_15-14-48_100p'
                             '\\99/2019-07-11_15-14-48_99_RESULT.erfh5']

    def test_caching(self):
        self.num_validation_samples = 400
        self.num_test_samples = 400
        self.caching_path = Path(r'X:\s\t\stiebesi\code\tests\erfh5_pipeline\caching')
        self.generator = pipeline.ERFH5DataGenerator(data_paths=self.paths,
                                                     num_validation_samples=self.num_validation_samples,
                                                     num_test_samples=self.num_test_samples,
                                                     max_queue_length=8096,
                                                     data_processing_function=dli.get_sensordata_and_flowfront,
                                                     data_gather_function=dg.get_filelist_within_folder,
                                                     num_workers=1,
                                                     cache_path=self.caching_path)
        p = Path(self.caching_path)
        filelists = p / 'filelists'
        with open(filelists / '2019-07-11_15-14-48_100p', 'rb') as f:
            fl = pickle.load(f)
        self.assertEqual(fl, self.cached_files)

    def tearDown(self):
        all_files = self.caching_path.glob('**/*')
        [x.unlink() for x in all_files if x.is_file()]
        all_files = self.caching_path.glob('**/*')
        [x.rmdir() for x in all_files if x.is_dir()]


class Test_dataset_split(unittest.TestCase):
    def setUp(self):
        self.paths = [r'X:\s\t\stiebesi\data\RTM\Leoben\output\with_shapes\2019-07-11_15-14-48_100p']
        num_samples = 100  # <- Number of files * Number of frames
        self.num_validation_samples = 400
        self.num_test_samples = 400
        self.generator = pipeline.ERFH5DataGenerator(data_paths=self.paths,
                                                     num_validation_samples=self.num_validation_samples,
                                                     num_test_samples=self.num_test_samples,
                                                     max_queue_length=8096,
                                                     data_processing_function=dli.get_sensordata_and_flowfront,
                                                     data_gather_function=dg.get_filelist_within_folder,
                                                     num_workers=10)

    def test_test_set(self):
        self.assertEqual(self.num_test_samples, len(self.generator.get_test_samples()))

    def test_validation_set(self):
        self.assertEqual(self.num_validation_samples, len(self.generator.get_validation_samples()))


if __name__ == '__main__':
    unittest.main()
