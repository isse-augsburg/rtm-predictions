import time
import os
import analizer
from SimCreator import SimCreator, zip_folder

if __name__== "__main__":
    if os.environ['Write_Simulation'] == '1':
        n_batches = 10
        count = 200
        perturbation_factors = \
        {
        "General_Sigma": .001,
        "Shapes":
            {
                "Rectangles":
                    {
                        "Num": 1,
                        "Fiber_Content":
                            [-.3, .3]
                    },
                "Circles":
                    {
                        "Num": 1,
                        "Fiber_Content": [-.3, 0]
                    }
            }
        }
        for batch_num in range(n_batches):
            sc = SimCreator(perturbation_factors, count=count, batch_num=batch_num, run_on_cluster=False)
            t00 = time.time()
            print(f'Batch {batch_num + 1}/{n_batches}: creating {count}/{count * n_batches} simulations.')
            sc.run()
            print(f'Whole creation of {count} simulations took {(time.time() - t00)/60:.1f} minutes.')

    # Execute slurm ...

    # Run an analysis over the new simulations
    elif os.environ['Analysis'] == '1':
        path = r'Y:\data\RTM\Lautern\output\with_shapes'
        print_options = ['all', 'fails_only', 'success_only']
        analizer.analize_subdirs(path, print_options='all')