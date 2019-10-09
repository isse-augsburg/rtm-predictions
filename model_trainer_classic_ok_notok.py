import logging
import math
import pickle
import socket
from datetime import datetime
from pathlib import Path

import h5py
# import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from matplotlib import rcParams
# from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.cluster import KMeans
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

from HDF5DB.hdf5db_toolbox import HDF5DBToolbox


class DataLoader:
    def __init__(self):
        self.hdf5db = HDF5DBToolbox()
        self.hdf5db.load(
            "/cfs/share/cache/HDF5DB_Cache",
            "Unrestricted"
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

    def select(self, variable, comparisonoperator, value):
        self.hdf5db.select(variable, comparisonoperator, value)
        self.result_path = []
        self.avg_lvl = []
        for obj in self.hdf5db.hdf5_object_list:
            self.avg_lvl.append(obj.avg_level)
            self.result_path.append(obj.path_result)

    # 1. version of get_dataset, map to max-shape
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
                # Column shaping
                data = r[self.pressure_path][()][:, :, 0]
                temp = np.zeros((data.shape[0], max_shape[1]))
                temp[:, data.shape[0]:] = data[:, -1]
                temp[:data.shape[0], :data.shape[1]] = data
                # Line shaping
                pressure = np.zeros(max_shape)
                pressure[temp.shape[0]:, :] = temp[-1, :]
                pressure[:temp.shape[0], :temp.shape[1]] = temp
                pressure = pressure.reshape(pressure.shape[0] * pressure.shape[1])
                dataset.append(pressure)
        return np.asarray(dataset)

    # 2. version of get_dataset, map to given shape, begin filling from right lower edge of original data
    def get_dataset_subsampled(self, sensorgrid_size=1140, time_steps=400):
        dataset = []

        # Fill dataset with pressure-data from RESULT.erfh5
        # Reshape to given sensorgrid-size and time-steps
        # Fill the remaining part created by splitting with the last n values of the original data set
        for path in self.result_path:
            r = h5py.File(path, "r")
            if self.pressure_path in r:
                # Column shaping
                data = r[self.pressure_path][()][:, :, 0]
                sampler = math.ceil(data.shape[1] / (sensorgrid_size - 1))
                temp = data[:, data.shape[1] - sensorgrid_size:]
                temp2 = data[:, 0::sampler]
                temp[:, :np.asarray(temp2).shape[1]] = temp2
                # Line shaping
                sampler = math.ceil(temp.shape[0] / (time_steps - 1))
                pressure = temp[temp.shape[0] - time_steps:, :]
                temp = temp[0::sampler]
                pressure[:temp.shape[0], :] = temp
                pressure = pressure.reshape(pressure.shape[0] * pressure.shape[1])
                dataset.append(pressure)
        return np.asarray(dataset)

    # 3. version of get_dataset, map to given shape, fill remaining columns with last column and
    # remaining rows with last row
    def get_dataset_subsampled_v2(self, sensorgrid_size=1140, time_steps=400):
        dataset = []

        # Fill dataset with pressure-data from RESULT.erfh5
        # Reshape to given sensorgrid-size and time-steps
        # Fill the remaining with the last elements of the dimension
        for path in self.result_path:
            r = h5py.File(path, "r")
            if self.pressure_path in r:
                # Column shaping
                data = r[self.pressure_path][()][:, :, 0]
                indices = np.round(np.arange(0, data.shape[1] - 1, data.shape[1] / sensorgrid_size))
                temp = np.zeros((data.shape[0], sensorgrid_size))
                temp[:, data.shape[0]:] = data[:, -1]
                temp2 = data[:, indices.astype(int)]
                temp[:, :np.asarray(temp2).shape[1]] = temp2
                # Line shaping
                indices = np.flip(np.round(np.arange(data.shape[0] - 1, 0, -temp.shape[0] / time_steps)))
                pressure = np.zeros((time_steps, sensorgrid_size))
                pressure[temp.shape[0]:, :] = temp[-1, :]
                temp = temp[indices.astype(int), :]
                pressure[:temp.shape[0], :temp.shape[1]] = temp
                pressure = pressure.reshape(pressure.shape[0] * pressure.shape[1])
                dataset.append(pressure)
        return np.asarray(dataset)

    def get_resultset(self):
        return np.asarray(self.avg_lvl)


class Classic:
    def __init__(self, save_path, sensorgrid_size=0, time_steps=0):
        self.initial_timestamp = str(
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        self.save_path = save_path / self.initial_timestamp
        self.save_path.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            filename=self.save_path / Path("output.log"),
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        dataloader = DataLoader()
        self.resultset = dataloader.get_resultset()
        if sensorgrid_size > 0 and time_steps > 0:
            self.dataset = dataloader.get_dataset_subsampled_v2(sensorgrid_size, time_steps)
        else:
            self.dataset = dataloader.get_dataset()
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.dataset, self.resultset,
                                                                                test_size=0.3,
                                                                                random_state=100)

    def run_xgboost_training(self):
        logger = logging.getLogger(__name__)
        print("Running XGBoost")
        logger.info("Running XGBoost")
        self.x_train, x_val, self.y_train, y_val = train_test_split(self.x_train, self.y_train,
                                                                    test_size=0.2,
                                                                    random_state=100)
        # clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100,
        #                                   max_depth=3, min_samples_leaf=5)
        tree = xgb.XGBRegressor(silent=False, objective='reg:logistic', colsample_bytree=0.3, learning_rate=0.06,
                                max_depth=3, n_estimators=10000, subsample=0.8, gamma=1)
        tree.fit(self.x_train, self.y_train, eval_set=[(x_val, y_val)], early_stopping_rounds=30)

        # Predict
        y_pred = tree.predict(self.x_test)
        rcParams['figure.figsize'] = 900, 600
        xgb.plot_tree(tree, num_trees=0)
        # plt.show()

        # For sklearn decisiontree
        # rcParams['figure.figsize'] = 80, 50
        # plt.figure()
        # plot_tree(tree, filled=True)
        # plt.show()

        accuracy = accuracy_score(self.y_test, y_pred.round())
        print(confusion_matrix(self.y_test, y_pred.round()))
        logger.info(confusion_matrix(self.y_test, y_pred.round()))
        print(classification_report(self.y_test, y_pred.round()))
        logger.info(classification_report(self.y_test, y_pred.round()))
        print("Accuracy is " + str(accuracy * 100))
        logger.info("Accuracy is " + str(accuracy * 100))
        pickle.dump(tree, open(self.save_path / "xgb_model.dat", "wb"))
        logger.info("XGB-Model saved")

    def svc_param_selection(self, x, y, nfolds):
        cs = [0.001, 0.01, 0.1, 1, 10]
        gammas = [0.001, 0.01, 0.1, 1]
        param_grid = {'C': cs, 'gamma': gammas}
        grid_search = GridSearchCV(SVC(kernel='linear', probability=True, class_weight='balanced'), param_grid,
                                   cv=nfolds)
        grid_search.fit(x, y)
        print(grid_search.best_params_)
        return grid_search

    def run_svm_training(self):
        n_estimators = 10
        logger = logging.getLogger(__name__)
        print("Running sklearn.svm")
        logger.info("Running sklearn.svm")

        # svclassifier = SVC(self.svc_param_selection(self.dataset, self.resultset, 10))
        # svclassifier = SVC(kernel='linear')
        svclassifier = OneVsRestClassifier(BaggingClassifier(SVC(kernel='linear', probability=True,
                                                                 class_weight='balanced'),
                                                             max_samples=1.0 / n_estimators,
                                                             n_estimators=n_estimators))
        svclassifier.fit(self.x_train, self.y_train)

        # Predict
        y_pred = svclassifier.predict(self.x_test)
        accuracy = accuracy_score(self.y_test, y_pred.round())

        print(confusion_matrix(self.y_test, y_pred.round()))
        logger.info(confusion_matrix(self.y_test, y_pred.round()))
        print(classification_report(self.y_test, y_pred.round()))
        logger.info(classification_report(self.y_test, y_pred.round()))
        print("Accuracy is " + str(accuracy * 100))
        logger.info("Accuracy is " + str(accuracy * 100))
        pickle.dump(svclassifier, open(self.save_path / "svm_model.dat", "wb"))
        logger.info("SVC-Model saved")

    def run_kmeans_training(self):
        logger = logging.getLogger(__name__)
        print("Running sklearn.kmeans")
        logger.info("Running sklearn.kmeans")

        # svclassifier = SVC(self.svc_param_selection(dataset, resultset, 10))
        kmeans = KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
                        n_clusters=2, n_init=10, n_jobs=2, precompute_distances='auto',
                        random_state=None, tol=0.0001, verbose=0)
        kmeans.fit(self.x_train)

        # Predict
        y_pred = kmeans.predict(self.x_test)
        accuracy = accuracy_score(self.y_test, y_pred.round())

        print(confusion_matrix(self.y_test, y_pred.round()))
        logger.info(confusion_matrix(self.y_test, y_pred.round()))
        print(classification_report(self.y_test, y_pred.round()))
        logger.info(classification_report(self.y_test, y_pred.round()))
        print("Accuracy is " + str(accuracy * 100))
        logger.info("Accuracy is " + str(accuracy * 100))
        pickle.dump(kmeans, open(self.save_path / "kmeans_model.dat", "wb"))
        logger.info("KMeans-Model saved")


if __name__ == "__main__":
    if socket.gethostname() == "swt-dgx1":
        save_path = Path('/cfs') / Path('share') / Path('cache') / Path('HDF5DB_Cache') / Path('Train_Out')
    else:
        _save_path = Path('/home/hartmade/Train_Out')

    st = Classic(_save_path, 1140, 400)
    st.run_kmeans_training()
    st.run_svm_training()
    st.run_xgboost_training()
    logging.shutdown()
