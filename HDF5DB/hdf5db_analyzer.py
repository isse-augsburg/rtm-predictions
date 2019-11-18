import matplotlib.pyplot as plt


class H5DBAnalyzer:
    def __init__(self, myhdf5db):
        self.hdf5db = myhdf5db

    def get_sum_single_states(self):
        state_sum = 0
        single_state_set = []
        for i in self.hdf5db.hdf5_object_list:
            if hasattr(i, "single_states"):
                state_sum += i.single_states
                single_state_set.append(i.single_states)
        return state_sum

    def plot_single_states_distribution(self):
        single_state_set = []
        for i in self.hdf5db.hdf5_object_list:
            if hasattr(i, "single_states"):
                single_state_set.append(i.single_states)
        plt.hist(single_state_set,
                 [50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500, 525, 550,
                  575, 600, 625, 650, 675, 700, 725, 750, 775, 800, 825, 850, 875, 900, 925, 950])
        plt.show()
