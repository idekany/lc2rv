import pandas as pd
import numpy as np
import sys
import os
import joblib
from sklearn.datasets import make_regression
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score


def load_dataset(filename, trim_quantiles: list = None, qlo=0.25, qhi=0.75, plothist=False,
                 histfig: str = 'hist', figformat='png', n_poly: int = 1, usecols: list = None,
                 input_feature_names=None, output_feature_indices: list = None,
                 y_col: str = None, yerr_col: str = None, subset_expr: str = None,
                 dropna_cols: list = None, comment='#', pca_model: object = None):
    """
    Loads, trims, and exports dataset to numpy arrays.

    :param filename: string
    The name of file to read from.

    :param trim_quantiles: list
    If provided, data beyond the lower and upper quantiles (qlo, qhi) of the listed column names will be trimmed.

    :param qlo: float
    The lower quantile below which to trim the data.

    :param qhi: float
    The upper quantile above which to trim the data.

    :param plothist: bolean
    Whether to plot histograms of the data in the list of columns provided by usecols.

    :param histfig: string
    Name of the figure file of the plotted histograms.

    :param figformat: string
    Format of the plotted histogram figure file.

    :param n_poly: positive integer
    The order of the polynomial basis of the output features. If larger than 1, polynomial features of the provided
    order will be created from the feature list given by input_feature_names.

    :param usecols: list
    List of column names to be read in from the input file.
    Passed as the usecols parameter in the argument of pandas.read_csv.

    :param input_feature_names: list
    List of the column names to be used as covariate features.

    :param output_feature_indices: list of integers
    List of the indices of the features to be used. Useful if one wants to select specific features when n_poly>1.

    :param y_col: string
    Name of the column to be considered as the target variable.

    :param yerr_col:
    Name of the column to be considered as the uncertainty on the target variable.

    :param subset_expr: string
    Expression to be used for specifying threshold rejections to be applied on one or more variables listed in usecols.
    Passed to the argument of pandas.dataframe.query().

    :param dropna_cols: list
    List of the column names for which all rows containing NaN values should be omitted.

    :param comment: string
    If a row in the input file starts with this character string, it will be trated as a comment line.
    with the exception of the first row which must start with a "#" and contain the column names.

    :param pca_model: object (scikit-learn PCA transformer object)
    PCA-transformation to be applied on the input data, must include the standardization step.
    Applied on the features in input_feature_names after the threshold rejections and
    before the polynomial transformation.

    :return:
    X: ndarray: The final data matrix, shape: (n_samples, n_features).
    y: ndarray: Values of the target variable, shape: (n_samples, )
    yw: ndarray: Uncertainties of the target variable, shape: (n_samples, )
    df: pandas dataframe object: all input data after quantile trimming threshold rejections
    feature_names: list: The names of the features in X.
    df_orig: The original data frame as read from the input file.
    """

    if input_feature_names is None:
        input_feature_names = ['x']

    with open(filename) as f:
        header = f.readline()
    cols = header.strip('#').split()
    df = pd.read_csv(filename, names=cols, header=None, sep='\s+', usecols=usecols, comment=comment)
    if dropna_cols is not None:
        df.dropna(inplace=True, subset=dropna_cols)
    ndata = len(df)
    print(df.head())
    print("----------\n{} lines read from {}\n".format(ndata, filename))

    df_orig = df

    # plot histogram for each column in original dataset
    if plothist:
        fig, ax = plt.subplots(figsize=(20, 10))
        # df.hist('ColumnName', ax=ax)
        # fig.savefig('example.png')
        _ = pd.DataFrame.hist(df, bins=int(np.ceil(np.cbrt(ndata) * 2)), figsize=(20, 10), grid=False, color='red',
                              ax=ax)
        plt.savefig(histfig + '.' + figformat, format=figformat)

    # Apply threshold rejections:
    if subset_expr is not None:
        df = df.query(subset_expr)

        ndata = len(df)
        print("{} lines after threshold rejections\n".format(ndata))

        # plot histogram for each column in original dataset
        if plothist:
            fig, ax = plt.subplots(figsize=(20, 10))
            # df.hist('ColumnName', ax=ax)
            # fig.savefig('example.png')
            _ = pd.DataFrame.hist(df, bins=int(np.ceil(np.cbrt(ndata) * 2)), figsize=(20, 10), grid=False, color='red',
                                  ax=ax)
            plt.savefig(histfig + '_sel.' + figformat, format=figformat)

    # omit data beyond specific quantiles [qlo, qhi]
    if trim_quantiles is not None:

        dfq = df[trim_quantiles]
        quantiles = pd.DataFrame.quantile(dfq, q=[qlo, qhi], axis=0, numeric_only=True, interpolation='linear')
        print("Values at [{},{}] quantiles to be applied for data trimming:".format(qlo, qhi))
        print(quantiles.sum)
        # df_t = df[( df > df.quantile(qlo) ) & ( df < df.quantile(qhi) )]
        mask = (dfq > dfq.quantile(qlo)) & (dfq < dfq.quantile(qhi))
        # print(mask)
        mask = mask.all(axis=1)
        # print(mask.shape)
        df = pd.DataFrame.dropna(df[mask])
        ndata = len(df)
        print("\n{} lines remained after quantile rejection.\n".format(ndata))
        # plot histogram for each column in trimmed dataset
        if plothist:
            fig, ax = plt.subplots(figsize=(20, 10))
            _ = pd.DataFrame.hist(df, bins=int(np.ceil(np.cbrt(ndata) * 2)), figsize=(20, 10), grid=False,
                                  color='green', ax=ax)
            fig.savefig(histfig + "_trimmed." + figformat, format=figformat)

    # extract input features:
    X = df.loc[:, input_feature_names].to_numpy()

    # Apply PCA transformation on X using a previously trained model:
    if pca_model is not None:
        pca_model = joblib.load(pca_model)
        X = pca_model.transform(X)
        input_feature_names = ["E{}".format(i + 1) for i in range(X.shape[1])]

    # Extract column of target variable:
    if y_col is not None:
        # print(df_t[y_col])
        y = df[y_col].to_numpy()
    else:
        y = np.array([])

    # Define weights:
    if yerr_col is not None:
        df['weight'] = 1.0 / df[yerr_col] ** 2
        yw = df['weight'].to_numpy()
    else:
        yw = np.ones_like(y)

    # create polynomial features:
    if n_poly > 1:
        trans = PolynomialFeatures(degree=n_poly, include_bias=False)
        X = trans.fit_transform(X)
        input_feature_names = trans.get_feature_names(input_feature_names)

    if output_feature_indices is not None:
        feature_names = np.array(input_feature_names)[output_feature_indices]
        X = X[:, output_feature_indices]
    else:
        feature_names = input_feature_names

    return X, y, yw, df, feature_names, df_orig


def read_light_curves(names, subdir, file_suffix, nuse=1, maxphase=1.0, scale=False, verbose=True):
    """
    Read photometric light curve(s) for one or more objects.
    :param names: str or array-like
        The name (identifier) of the object or an array-like with the list of objects.
    :param subdir: str
        The name of the subdirectory where the light curve data files are stored.
    :param file_suffix: str
        Suffix of the files storing the light curve of each object in the scheme of <subdir>/<name><file_suffix>
    :param nuse: int
        Read only every `nuse`-th data point.
    :param maxphase: float
        Read the data with phases up to `maxphase`.
    :param scale: bool
        Whether to scale the light curves to the [0,1] range.
    :param verbose: bool
        Turns verbosity on/off.
    :return: (numpy.ndarray, numpy.ndarray) OR (list, list)
        Returns the magnitudes and corresponding phases if `names` is a single name string,
        returns the lists of magnitudes and corresponding phases if `names` is array-like.
    """

    if verbose:
        print("Reading light curves...", file=sys.stderr)

    if type(names) not in (list, tuple, np.ndarray):
        names = [names]

    tseries_list = []
    phases = None

    if scale:
        # Scale the time series to the [0,1] range
        scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
    else:
        scaler = None

    for ii, name in enumerate(names):

        if verbose:
            print('Reading light curve of {}\r'.format(name), end="", file=sys.stderr)

        phases, timeseries = np.genfromtxt(os.path.join(subdir, name + file_suffix),
                                           unpack=True, comments='#')
        timeseries = timeseries[phases < maxphase]
        phases = phases[phases < maxphase]

        if scaler:
            scaler.fit(timeseries.reshape(-1, 1))
            timeseries = (scaler.transform(timeseries.reshape(-1, 1))).flatten()

        tseries_list.append(timeseries[nuse - 1::nuse])
        phases = phases[nuse - 1::nuse]

    if verbose:
        print("")

    if len(tseries_list) == 1:
        return tseries_list[0], phases
    else:
        return tseries_list, phases


def read_rv_curves(names, subdir, file_suffix, maxphase=1.0, phase_shift=None, verbose=True):
    """
    Read radial velocity curve(s) for one or more objects.
    :param names: str or array-like
        The name (identifier) of the object or an array-like with the list of objects.
    :param subdir: str
        The name of the subdirectory where the radial velocity data files are stored.
    :param file_suffix: str
        Suffix of the files storing the light curve of each object in the scheme of <subdir>/<name><file_suffix>
    :param maxphase: float
        Read the data with phases up to `maxphase`.
    :param phase_shift: float
        Whether to systematically shift the input phases by the same amount (default: None, i.e., no shift).
    :param verbose: bool
        Turns verbosity on/off.
    :return: (numpy.ndarray, numpy.ndarray, numpy.ndarray) OR (list, list, list)
        Returns the radial velocities, their errors, and corresponding phases if `names` is a single name string,
        returns the lists of radial velocities, their errors, and corresponding phases if `names` is array-like.
    """

    if verbose:
        print("Reading RV curves...", file=sys.stderr)

    if type(names) not in (list, tuple, np.ndarray):
        names = [names]

    rv_list = []
    rve_list = []
    phases_list = []

    for ii, name in enumerate(names):

        if verbose:
            print('Reading data for {}\r'.format(name), end="", file=sys.stderr)

        pp, rv, rve = np.genfromtxt(os.path.join(subdir, name + file_suffix),
                                    unpack=True, comments='#')
        phasemask = (pp < maxphase)
        pp = pp[phasemask]
        rv = rv[phasemask]
        rve = rve[phasemask]

        if phase_shift is not None:
            pp = get_phases(1.0, pp, shift=phase_shift, all_positive=True)
            inds = np.argsort(pp)
            pp = pp[inds]
            rv = rv[inds]
            rve = rve[inds]

        rv_list.append(rv)
        rve_list.append(rve)
        phases_list.append(pp)

    if verbose:
        print("")

    if len(rv_list) == 1:
        return rv_list[0], rve_list[0], phases_list[0]
    else:
        return rv_list, rve_list, phases_list


def plot_all_lc_grid(phases, lc_list, shift=0, fname=None, indx_highlight=None,
                     figformat="png", invert_yaxis=True):
    """
    Plot all input light curves as phase diagrams.
    """
    nmags = lc_list[0].shape[0]
    n_roll = int(np.round(shift * nmags))

    n_samples = len(lc_list)

    if indx_highlight is not None:
        assert (indx_highlight >= 0)
        assert (indx_highlight == int(indx_highlight))
        assert (indx_highlight < n_samples)

    fig = plt.figure(figsize=(5, 4))
    fig.subplots_adjust(bottom=0.13, top=0.94, hspace=0.3, left=0.15, right=0.98, wspace=0)

    for ii, lc in enumerate(lc_list):
        plt.plot(phases, np.roll(lc, n_roll), ls='-', color='grey', lw=0.3, alpha=0.3)

    if indx_highlight is not None:
        plt.plot(phases, np.roll(lc_list[indx_highlight], n_roll), 'ko')
    plt.xlabel('phase')
    plt.ylabel('mag')
    # plt.ylim(-1.1, 0.8)
    if invert_yaxis:
        plt.gca().invert_yaxis()
    plt.savefig(fname + "." + figformat, format=figformat)
    plt.close(fig)


def plot_all_rv(phases_list, rv_list, fname=None, indx_highlight=None,
                figformat="png"):
    """
    Plot all input radial velocity as phase diagrams.
    """

    n_samples = len(rv_list)

    if indx_highlight is not None:
        assert (indx_highlight >= 0)
        assert (indx_highlight == int(indx_highlight))
        assert (indx_highlight < n_samples)

    fig = plt.figure(figsize=(5, 4))
    fig.subplots_adjust(bottom=0.13, top=0.94, hspace=0.3, left=0.15, right=0.98, wspace=0)

    for ii, (pp, rv) in enumerate(zip(phases_list, rv_list)):
        plt.plot(pp, rv, '.', color='grey', lw=0.3, alpha=0.3)

    if indx_highlight is not None:
        plt.plot(phases_list[indx_highlight], rv_list[indx_highlight], 'ko')
    plt.xlabel('phase')
    plt.ylabel('RV')

    plt.savefig(fname + "." + figformat, format=figformat)
    plt.close(fig)


def get_phases(period, x, epoch=0.0, shift=0.0, all_positive=True):
    """
    Compute the phases of a monoperiodic time series.

    :param period: float

    :param x: 1-dimensional ndarray.
    The array of the time values of which we want to compute the phases.

    :param epoch: float, optional (default=0.0)
    Time value corresponding to zero phase.

    :param shift: float, optional (default=0.0)
    Phase shift wrt epoch.

    :param all_positive: boolean, optional (default=True)
    If True, the computed phases will be positive definite.

    :return:
    phases: 1-dimensional ndarray
    The computed phases with indices matching x.
    """

    phases = np.modf((x - epoch + shift * period) / period)[0]

    if all_positive:
        phases = all_phases_positive(phases)
    return phases


def all_phases_positive(pha):
    """
    Converts an array of phases to be positive definite.

    :param pha : 1-dimensional ndarray
    The phases to be modified.

    :return:
    pha: 1-dimensional ndarray
    Positive definite version of pha.
    """

    while not (pha >= 0).all():  # make sure that all elements are >=0
        pha[pha < 0] = pha[pha < 0] + 1.0
    return pha


def get_data_matrix(id_list, periods, lc_list, rv_list, rve_list, rv_phases_list):
    """
    Assemble the input 'design matrix' from the light curves, periods, and phases,
    that will be the input of the predictive model.
    :param id_list: array-like
        List of object identifiers in the training set.
    :param periods: array-like
        List of periods of the objects in the training set.
    :param lc_list: array-like
        List of the light curves of objects in the training set.
    :param rv_list: array-like
        List of radial velocity curves of the objects in the training set.
    :param rve_list: array-like
        List of radial velocity errors of the objects in the training set.
    :param rv_phases_list: array-like
        List of radial velocity phases of the objects in the training set.
    :return: (5 * numpy.ndarray, dict, dict)
        Arrays of the design matrix, target values, sample weights, identifiers, group labels,
        and disctionaries mapping the identifiers to group labels and back.
    """

    X_list = []
    ids = []
    groups = []
    id2group = {}
    group2id = {}

    for ii, ident in enumerate(id_list):
        lc = lc_list[ii]
        period = periods[ii]
        rv_phases = rv_phases_list[ii]
        n_rv = len(rv_phases)

        X = np.tile(np.append(lc, period), (n_rv, 1))
        X = np.hstack((X, rv_phases.reshape(-1, 1)))
        X_list.append(X)

        ids.append(np.repeat(ident, n_rv))
        groups.append(np.repeat(ii, n_rv))

        id2group[ident] = ii
        group2id[ii] = ident

    X = np.concatenate(X_list)
    y = np.concatenate(rv_list).reshape(-1, 1)
    yw = np.concatenate(rve_list).reshape(-1, 1)
    yw = 1.0 / yw ** 2
    ids = np.concatenate(ids)
    groups = np.concatenate(groups)

    return X, y, yw, ids, groups, id2group, group2id


def plot_residual_hist(y, yhat, filename='res_hist', figformat='png'):
    """
    Plot a histogram of the residuals.
    :param y: array-like
        Array of the true values.
    :param yhat: array-like
        Array of the predicted values.
    :param filename: str
        Filename for the output figure (without extension).
    :param figformat: str
        File format of the output figure (valid matplotlib file formats are accepted).
    :return:
    """
    residual = (y - yhat).flatten()

    fig, ax = plt.subplots(figsize=(8, 5))
    _ = plt.hist(residual, bins='scott', color='red')
    ax.set_xlabel("$V_{r}~[km/s]$")
    ax.set_ylabel("$N$")
    plt.savefig(filename + '_sel.' + figformat, format=figformat)


def plot_residual(y, yhat, phase,
                  binned_data: list = None, colors: list = None,
                  filename: str = 'residual', xlabel: str = None, figformat='png'):
    """
    Plot the residuals vs the phases.
    """
    residual = (y - yhat).flatten()

    # ndata = yhat.shape[0]

    fig, ax = plt.subplots(figsize=(8, 5))
    # df.hist('ColumnName', ax=ax)
    # fig.savefig('example.png')
    # _ = plt.hist(residual, bins=int(np.ceil(np.cbrt(ndata) * 2)), color='red')
    _ = plt.scatter(phase, residual, s=2, c='black', alpha=0.5, marker='.')
    if binned_data is not None:
        for ii, tup in enumerate(binned_data):
            if colors is not None:
                color = colors[ii]
            else:
                color = None
            _ = plt.errorbar(tup[0], tup[1], yerr=tup[2], capsize=1, fmt='none', ecolor=color, c=color)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    ax.set_ylabel("Validation error ($\Delta V_{r}~[km/s]$)")
    plt.savefig(filename + '_sel.' + figformat, format=figformat)


def fit_validate_model(model, X: np.array, y: np.array, ids, train_index, val_index, sample_weight: np.array = None):
    """
    Train and cross-validate a model by repeatedly fitting on part of the data and
    evaluating the model on the rest of the data.
    :param model: object
        A predictive model instance implementing a `fit` method with signature:
         fit(X: np.ndarray(n_samples, n_features), y: np.ndarray(n_samples), sample_weight: np.ndarray(n_samples))
        and a `predict` method with signature:
         predict(X: np.ndarray(n_samples, n_features))
        An sklearn model instance will work.
    :param X: np.ndarray
        Input data matrix (or 'design matrix') of shape (n_samples, n_features)
    :param y: np.ndarray
        Array of the true values of the response variable
    :param ids: np.ndarray
        Identifiers of the data sample.
    :param train_index: np.ndarray
        Index array of the training set of the data sample.
    :param val_index: np.ndarray
        Index array of the validation set of the data sample.
    :param sample_weight: np.ndarray
        Array of the sample weights passed to the model.fit method.
    :return: 6 * numpy.ndarray, float
        Arrays with the ids, periods, phases, true values, predicted values, and sample weights of the validation set,
        and the R2 score computed for the validation set.
    """
    X_t, X_v = X[train_index, :], X[val_index, :]
    y_t, y_v = y[train_index], y[val_index]
    ids_t, ids_v = ids[train_index], ids[val_index]

    if sample_weight is not None:
        yw_t, yw_v = sample_weight[train_index], sample_weight[val_index]
    else:
        yw_t = None
        yw_v = None

    model.fit(X_t, y_t, sample_weight=yw_t)

    # yhat_t = (model.predict(X_t)).flatten()
    yhat_v = (model.predict(X_v)).flatten()

    periods_v = X_v[:, -2].flatten()
    phases_v = X_v[:, -1].flatten()
    y_v = y_v.flatten()
    yw_v = yw_v.flatten()

    r2_v = r2_score(y_v, yhat_v, sample_weight=yw_v)

    return ids_v, periods_v, phases_v, y_v, yhat_v, yw_v, r2_v


def get_stratification_labels(data, n_folds):
    """
    Create an array of stratification labels from an array of continuous values to be used in a stratified cross-
    validation splitter.
    :param data: list or numpy.ndarray
        The input data array.
    :param n_folds: int
        The number of cross-validation folds to be used with the output labels.
    :return: labels, numpy.ndarray
        The array of integer stratification labels.
    """

    assert isinstance(data, np.ndarray or list), "data must be of type list or numpy.ndarray"
    if isinstance(data, list):
        data = np.array(data)

    ndata = len(data)
    isort = np.argsort(data)  # Indices of sorted data
    labels = np.empty(ndata)
    labels[isort] = np.arange(ndata)  # Compute data order
    labels = np.floor(labels / n_folds)  # compute data labels for StratifiedKFold
    if np.min(np.bincount(labels.astype(int))) < n_folds:  # If too few elements are with last label, ...
        labels[labels == np.max(labels)] = np.max(
            labels) - 1  # ... the then change that label to the one preceding it

    return labels
