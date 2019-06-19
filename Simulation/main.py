import datetime
import time
import os
from pathlib import Path

from Simulation import analizer
from Simulation.SimCreator import SimCreator

if __name__== "__main__":
    if os.environ['Write_Simulation'] == '1':
        n_batches = 1
        count = 1
        overall_count = n_batches * count
        perturbation_factors = \
        {
        "General_Sigma": .001,
        "Shapes":
            {
                "Rectangles":
                    {
                        "Num": 1,
                        "Fiber_Content":
                            [.7, .8]
                    },
                "Circles":
                    {
                        "Num": 1,
                        "Fiber_Content": [.7, .8]
                    },
                "Runners":
                    {
                        "Num": 0,
                        "Fiber_Content": [-.7, -.8]
                    }
            }
        }
        initial_timestamp = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        print(initial_timestamp)
        t000 = time.time()
        for batch_num in range(n_batches):
            sc = SimCreator(perturbation_factors, initial_timestamp=initial_timestamp, n_in_batch=count,
                            batch_num=batch_num, run_on_cluster=False, overall_count=overall_count,
                            data_path=Path(r'Y:\data\RTM\Leoben'))
            t00 = time.time()
            print(f'Batch {batch_num + 1}/{n_batches}: creating {count * (batch_num + 1)}/{overall_count} simulations.')
            sc.run()
            print(f'Creation of {count} simulations took {(time.time() - t00)/60:.1f} minutes.')
        print(f'Whole creation of {overall_count} simulations took {(time.time() - t000) / 3600:.2f} hours.')
    # Execute slurm ...

    # Run an analysis over the new simulations
    elif os.environ['Analysis'] == '1':
        path = r'Y:\data\RTM\Lautern\output\with_shapes\2019-06-07_13-48-28_3500p'
        print_options = ['all', 'fails_only', 'success_only']
        analizer.analize_subdirs(path, print_options='none', suffix='erfh5')