from pathlib import Path

test_src_dir = Path(r'X:\s\t\stiebesi\code\tests')

test_training_out_dir = test_src_dir / 'training'
test_training_src_dir = test_src_dir / 'training_data_source'

test_eval_dir = test_src_dir / Path('evaluation/2019-09-02_19-40-56')

test_pipeline_dir = test_src_dir / Path('erfh5_pipeline/source')
test_caching_dir = test_src_dir / Path('erfh5_pipeline/caching')

data_loader_img_file = test_src_dir.joinpath(r'data_loader_IMG\2019-06-05_15-30-52_0_RESULT.erfh5')