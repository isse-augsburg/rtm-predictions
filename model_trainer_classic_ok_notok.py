import logging
import pickle
import socket
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from matplotlib import rcParams
# from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from HDF5DB.hdf5db_toolbox import HDF5DBToolbox


class DataLoader:
    def __init__(self):
        self.hdf5db = HDF5DBToolbox()
        self.hdf5db.load(
            "/cfs/share/cache/HDF5DB_Cache",
            "Unrestricted404"
        )

        self.pressure_path = "post/multistate/TIMESERIES1/multientityresults/SENSOR/PRESSURE/ZONE1_set1/erfblock/res"
        self.result_path = []
        self.avg_lvl = []
        for obj in self.hdf5db.hdf5_object_list:
            if obj.avg_level == 1:
                self.avg_lvl.append(1)
            else:
                self.avg_lvl.append(0)
            self.result_path.append(obj.path_result)

    def select(self, variable, comparisonOperator, value):
        self.hdf5db.select(variable, comparisonOperator, value)
        for obj in self.hdf5db.hdf5_object_list:
            self.avg_lvl.append(obj.avg_level)
            self.result_path.append(obj.path_result)

    def get_dataset(self):
        dataset = []
        max_shape = [0, 0]
        # Different shapes for pressure found, find max-shape
        for path in self.result_path:
            r = h5py.File(path, "r")
            if self.pressure_path in r:
                pressure = np.asarray(r[self.pressure_path][()][:, :, 0], object)
                pressure_shape = pressure.shape
                if pressure_shape[0] > max_shape[0]:
                    max_shape[0] = pressure_shape[0]
                if pressure_shape[1] > max_shape[1]:
                    max_shape[1] = pressure_shape[1]

        # Fill dataset with pressuredata from RESULT.erfh5
        # Reshape to max-shape, fill remaining with 0
        for path in self.result_path:
            r = h5py.File(path, "r")
            if self.pressure_path in r:
                pressure = np.zeros(max_shape)
                temp = r[self.pressure_path][()][:, :, 0]
                pressure[:temp.shape[0], :temp.shape[1]] = temp
                pressure = pressure.reshape(1, max_shape[0] * max_shape[1])
                dataset.append(pressure)

        dataset = np.asarray(dataset).reshape((404, max_shape[0] * max_shape[1]))
        return dataset

    def get_resultset(self):
        return np.asarray(self.avg_lvl)


class Classic:
    def __init__(self, save_path):
        self.initial_timestamp = str(
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        self.save_path = save_path

    def run_xgboost_training(self):
        save_path = self.save_path / self.initial_timestamp
        save_path.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            filename=save_path / Path("output.log"),
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        logger = logging.getLogger(__name__)
        logger.info("Generating Datasets")
        dataloader = DataLoader()
        resultset = dataloader.get_resultset()
        dataset = dataloader.get_dataset()
        x_train, x_test, y_train, y_test = train_test_split(dataset, resultset,
                                                            test_size=0.3,
                                                            random_state=100)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                          test_size=0.2,
                                                          random_state=100)
        logger.info("The Training Will Start Shortly")
        # clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100,
        #                                   max_depth=3, min_samples_leaf=5)
        tree = xgb.XGBRegressor(silent=False, objective='reg:logistic', colsample_bytree=0.3, learning_rate=0.06,
                                max_depth=3, n_estimators=10000, subsample=0.8, gamma=1)
        tree.fit(x_train, y_train, eval_set=[(x_val, y_val)], early_stopping_rounds=30)

        # Predict
        y_pred = tree.predict(x_test)
        rcParams['figure.figsize'] = 900, 600
        xgb.plot_tree(tree, num_trees=0)
        plt.show()

        # For sklearn decisiontree
        # rcParams['figure.figsize'] = 80, 50
        # plt.figure()
        # plot_tree(tree, filled=True)
        # plt.show()
        accuracy = accuracy_score(y_test, y_pred.round())
        pickle.dump(xgb, open("xgb_model.dat", "wb"))
        print("Accuracy is " + str(accuracy * 100))
        logger.info("Accuracy is " + str(accuracy * 100))
        logging.shutdown()


if __name__ == "__main__":

    if socket.gethostname() == "swt-dgx1":
        """ _cache_path = None
        _data_root = Path("/cfs/home/s/t/stiebesi/data/RTM/Leoben/
        output/with_shapes")
        _batch_size = 320
        _eval_freq = 50

        if getpass.getuser() == "stiebesi":
            _save_path = Path("/cfs/share/cache/output_simon")
        elif getpass.getuser() == "schroeni":
            _save_path = Path("/cfs/share/cache/output_niklas")
            # cache_path = "/run/user/1001/gvfs/smb-share:
            server=137.250.170.56,share=share/cache"
        else:
            _save_path = Path("/cfs/share/cache/output")
        _epochs = 10
        _num_workers = 18
        _num_validation_samples = 2000
        _num_test_samples = 2000 """

        print("TODO Fix paths for DGX")

    else:
        _cache_path = Path(
            '/cfs/share/cache')

        _data_root = Path(
            '/cfs/home/s/t/stiebesi/data/RTM/Leoben/output/with_shapes')
        _batch_size = 4
        _eval_freq = 50
        _save_path = Path('/home/hartmade/Train_Out')
        _num_workers = 4
        _num_validation_samples = 50
        _num_test_samples = 50

    st = Classic(save_path=_save_path)
    st.run_xgboost_training()
