import time
import os
import analizer
from SimCreator import SimCreator, zip_folder

if __name__== "__main__":
    if os.environ['Write_Simulation'] == '1':
        count = 1
        perturbation_factors = \
        {
        "General_Sigma": .001,
        "Shapes":
            {
                "Rectangles":
                    {
                        "Num": 10,
                        "Fiber_Content":
                            [-.3, .3]
                    },
                "Circles":
                    {
                        "Num": 10,
                        "Fiber_Content": [-.3, 0]
                    }
            }
        }

        sc = SimCreator(perturbation_factors, count)
        t00 = time.time()
        print(f'Creating {count} simulations.')
        sc.run()
        print(f'Whole creation of {count} simulations took {time.time() - t00:.0f} seconds.')
    # Execute slurm ...

    # Run an analysis over the new simulations
    elif os.environ['Analysis'] == '1':
        path = r'Y:\data\RTM\Lautern\1_solved_simulations\output\k1_sigK1_8e-12_K2_8e-12\2019-04-15_15-09-43_10p'
        print_options = ['all', 'fails_only', 'success_only']
        analizer.analize_subdirs(path, print_options='all')