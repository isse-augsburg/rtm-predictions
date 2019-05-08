import time
import os
import analizer
from SimCreator import SimCreator, zip_folder

if __name__== "__main__":
    if os.environ['Write_Simulation'] == '1':
        count = 10
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

        sc = SimCreator(perturbation_factors, count)
        t00 = time.time()
        print(f'Creating {count} simulations.')
        sc.run()
        print(f'Whole creation of {count} simulations took {(time.time() - t00)/60:.1f} minutes.')
    # Execute slurm ...

    # Run an analysis over the new simulations
    elif os.environ['Analysis'] == '1':
        path = r'Y:\data\RTM\Lautern\output\with_shapes'
        print_options = ['all', 'fails_only', 'success_only']
        analizer.analize_subdirs(path, print_options='all')