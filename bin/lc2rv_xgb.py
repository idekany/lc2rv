import os
import numpy as np
import importlib
from time import time
from joblib import Parallel, delayed, load, dump

import lc2rv_utils as ut
from lc2rv_params import *

from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.metrics import mean_squared_error
from skopt.utils import use_named_args
from skopt import gp_minimize
from statsmodels.stats.weightstats import DescrStatsW
from matplotlib import pyplot as plt

importlib.reload(ut)


def main():
    model = None

    # ======================================================================================================================
    # READ AND FILTER TRAINING DATA:

    if fit_hparams or train_model or train_cv_model:
        periods, _, _, df, feature_names, df_orig = \
            ut.load_dataset(os.path.join(rootdir, inputfile_train),
                            plothist=True,
                            usecols=usecols,
                            input_feature_names=input_feature_names,
                            subset_expr=subset_expr,
                            histfig=os.path.join(rootdir, 'hist'))

        # Create list of star identifiers:
        id_list = list(df['id'].astype(str))

        # Read the 'synthetic' light curves:
        lc_list, lc_phases = \
            ut.read_light_curves(id_list, os.path.join(rootdir, lc_subdir_train), lc_file_suffix_train, nuse=3,
                                 maxphase=1.0,
                                 scale=False)

        # Read the radial velocity data:
        rv_list, rve_list, rv_phases_list = \
            ut.read_rv_curves(id_list, os.path.join(rootdir, rv_subdir_train), rv_file_suffix_train, maxphase=1.0)

        # Create plot with all mean-subtracted, phase-aligned light curves:
        ut.plot_all_lc_grid(
            lc_phases, lc_list, shift=0, indx_highlight=indx_highlight,
            fname=os.path.join(rootdir, figname_lc),
            figformat="png", invert_yaxis=True)

        # Create plot with all mean-subtracted, phase-aligned RV curves:
        ut.plot_all_rv(
            rv_phases_list, rv_list, fname=os.path.join(rootdir, figname_rv),
            indx_highlight=indx_highlight, figformat="png")

        # Create the design matrix X and the vector of target values and weights y and yw:
        X, y, yw, ids, groups, id2group, group2id = \
            ut.get_data_matrix(id_list, periods, lc_list, rv_list, rve_list, rv_phases_list)

    # ======================================================================================================================
    # HYPERPARAMETER OPTIMIZATION:

    # ----------------------------------------
    # SET UP HYPERPARAMETER OPTIMIZATION:

    if fit_hparams:
        # Instantiate XGBOOST model:
        model = XGBRegressor(use_label_encoder=False, n_jobs=1, random_state=random_seed)

        # Define cross-validation splitter:
        cv = GroupKFold(n_splits=n_folds)
        cv.random_state = random_seed

        # function for Bayesian hyperparameter optimizaton of XGBOOST
        @use_named_args(search_space)
        def evaluate_model(**params):
            model.set_params(**params)
            # compute CV scores per fold
            scores = cross_val_score(model, X, y, groups=groups, cv=cv, n_jobs=-1,
                                     scoring=scoring, fit_params={"sample_weight": yw})
            # calculate the mean of the scores
            mean_score = np.mean(scores)
            # convert score to be minimized as the objective
            return np.sqrt(-1.0 * mean_score)

        warnings.filterwarnings('ignore', message='The objective has been evaluated at this point before.')

        # ----------------------------------------
        # PERFORM HYPER-PARAMETER OPTIMIZATION:

        tic = time()
        if verbose:
            print("Performing hyper-parameter optimization...")
        result = gp_minimize(evaluate_model, search_space, verbose=verbose,
                             n_calls=n_calls, n_initial_points=n_init,
                             random_state=random_seed,
                             n_restarts_optimizer=20, acq_optimizer='sampling')
        toc = time()

        if verbose:
            print("Hyper-parameter optimization completed.")
            print("skopt running time: {0:.2f}s".format(toc - tic))
            print("Best CV score: {0:.3f}".format(1 - result.fun))

        print("Best hyper-parameters: {0:s}".format(str(result.x)))

        # Set the optimal hyper-parameters for the model:
        model.set_params(learning_rate=result.x[0],
                         subsample=result.x[1],
                         n_estimators=result.x[2],
                         max_depth=result.x[3])

    # ======================================================================================================================
    # Cross-validate a model with weighted metrics and save the predictions made on the validation data.

    if train_cv_model:
        if model is None:
            # In this case, fit_hparams was set to False and thus we do not have a model instance yet.
            # Instantiate a model and set user-specified hyper-parameters:
            model = XGBRegressor(use_label_encoder=False, n_jobs=1, random_state=random_seed)
            model.set_params(learning_rate=learning_rate, subsample=subsample,
                             n_estimators=n_estimators, max_depth=max_depth)

        # Define cross-validation splitter:
        cv = GroupKFold(n_splits=n_folds)
        cv.random_state = random_seed
        cv_folds = list(cv.split(X, y, groups=groups))

        # Perform parallel training and cross-validation on data folds:
        cv_output = \
            np.array(Parallel(n_jobs=-1)(
                delayed(ut.fit_validate_model)
                (model, X, y, ids, train_index, val_index, sample_weight=yw) for train_index, val_index in cv_folds
            ), dtype=object)

        ids_v = np.concatenate(cv_output[:, 0]).astype(str)
        periods_v = np.concatenate(cv_output[:, 1]).astype(float)
        phases_v = np.concatenate(cv_output[:, 2]).astype(float)
        y_v = np.concatenate(cv_output[:, 3]).astype(float)
        yhat_v = np.concatenate(cv_output[:, 4]).astype(float)
        yw_v = np.concatenate(cv_output[:, 5]).astype(float)
        mean_r2_v = np.mean(cv_output[:, 6])

        wrmse_v = mean_squared_error(y_v, yhat_v, sample_weight=yw_v, squared=False)
        print("Weighted mean squared error (val.) = {:0.3f}".format(wrmse_v))

        # r2_v_all = r2_score(y_v, yhat_v, sample_weight=yw_v)
        print("Weighted R2 score (val.) = {0:0.3f}".format(mean_r2_v))

        # Compute weighted means and standard deviations
        bins = np.linspace(0, 1, 34)
        binned_phases = bins[1:] - bins[1] / 2.0

        bin_indices = np.digitize(X[:, -1], bins)
        bin_ids = np.unique(bin_indices)
        wstd_list = []
        for bin_id in bin_ids:
            mask = (bin_indices == bin_id)
            weighted_stats = DescrStatsW(y[mask].flatten(), weights=yw[mask], ddof=1)
            wmean = weighted_stats.mean
            wstd = weighted_stats.std
            wstd_list.append(wstd)

        bin_indices = np.digitize(phases_v, bins)
        bin_ids = np.unique(bin_indices)
        wstd_v_list = []
        for bin_id in bin_ids:
            mask = (bin_indices == bin_id)
            weighted_stats = DescrStatsW(y_v[mask] - yhat_v[mask], weights=yw_v[mask], ddof=1)
            wmean_v = weighted_stats.mean
            wstd_v = weighted_stats.std
            wstd_v_list.append(wstd_v)

        # train_out_array = np.rec.fromarrays((ids_t, y_t, yhat_t, yw_t))
        val_out_array = np.rec.fromarrays((ids_v, periods_v, phases_v, y_v, yhat_v, yw_v))

        # np.savetxt("train_pred.dat", train_out_array, fmt='%s %f %f %f', header="id y yhat weight")
        np.savetxt(os.path.join(rootdir, val_pred_out), val_out_array,
                   fmt='%s %f %f %f %f %f', header="id period phase y yhat weight")

        # Plot thehistogram of residuals:
        ut.plot_residual_hist(y_v, yhat_v, filename=os.path.join(rootdir, 'residual_hist'), figformat='png')

        binned_data = [(binned_phases, np.zeros(len(binned_phases)), wstd_list),
                       (binned_phases, np.zeros(len(binned_phases)), wstd_v_list)]

        # Plot CV error vs. phase:
        ut.plot_residual(
            y_v, yhat_v, phases_v,
            binned_data=binned_data, colors=['r', 'b'],
            xlabel="phase",
            filename=os.path.join(rootdir, "residual_vs_phase"))

        # Plot CV error vs period:
        ut.plot_residual(
            y_v, yhat_v, periods_v,
            xlabel="period",
            filename=os.path.join(rootdir, "residual_vs_period"))

    # ======================================================================================================================
    # TRAIN MODEL ON ALL INPUT DATA:

    if train_model:

        if model is None:
            # In this case, fit_hparams and train_cv_model were set to False
            #   and thus we do not have a model instance yet.
            # Instantiate a model and set user-specified hyper-parameters:
            model = XGBRegressor(use_label_encoder=False, n_jobs=-1, random_state=random_seed)
            model.set_params(learning_rate=learning_rate, subsample=subsample,
                             n_estimators=n_estimators, max_depth=max_depth)

        # Refit the best model with all data:
        model.fit(X, y, sample_weight=yw)

        dump(model, os.path.join(rootdir, output_model_file), compress=True, protocol=4)

    # ======================================================================================================================
    # DEPLOY A TRAINED MODEL TO MAKE PREDICTIONS:

    if predict:

        if model is None:
            # Load previously trained model:
            # with open(os.path.join(rootdir, input_model_file), 'r') as f:
            model = load(os.path.join(rootdir, input_model_file))

        identifiers, periods = np.genfromtxt(os.path.join(rootdir, inputfile_pred), dtype='S20, f8',
                                             unpack=True, comments='#')
        if periods.shape == ():
            periods = periods.reshape(1, )
            identifiers = identifiers.reshape(1, )
        identifiers = identifiers.astype(str)

        for i_obj, (identifier, period) in enumerate(zip(identifiers, periods)):

            magnitudes, _ = \
                ut.read_light_curves(identifier, os.path.join(rootdir, lc_subdir_pred), lc_file_suffix_pred,
                                     nuse=3, maxphase=1.0, verbose=False)

            if test:
                # In this case, true RV values and their errors are also provided for each input RV phase:

                rv, rve, rv_phases = \
                    ut.read_rv_curves(identifier, os.path.join(rootdir, rv_subdir_pred), rv_file_suffix_pred,
                                      maxphase=1.0, verbose=False)

            else:
                # In this case, only RV phases are provided:
                rv_phases = \
                    np.genfromtxt(os.path.join(rootdir, rv_subdir_pred, identifier + rv_file_suffix_pred),
                                  usecols=(0,), unpack=True, comments='#')
                rv = None

            n_rv = len(rv_phases)

            # Assemble the input design matrix:
            X = np.tile(np.append(magnitudes, period), (n_rv, 1))
            X = np.hstack((X, rv_phases.reshape(-1, 1)))

            # Make predictions:
            yhat = model.predict(X).flatten()

            # Print the phases and the corresponding predicted pulsational RVs for this object:
            for ph, yh in zip(rv_phases, yhat):
                print(identifier, ph, yh)

            if rv is not None:
                # Create a plot comparing the true and predcited RVs for this object:

                fig = plt.figure(figsize=(5, 4))
                plt.plot(rv_phases, yhat, 'bo')
                plt.plot(rv_phases, rv, 'ko')
                plt.xlabel('phase')
                plt.ylabel('RV [km/s]')
                plt.savefig(os.path.join(rootdir, identifier + "_rv_pred." + figformat), format=figformat)
                plt.tight_layout()
                plt.close(fig)


if __name__ == '__main__':
    main()
