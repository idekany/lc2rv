from skopt.space import Integer
from skopt.space import Real
import warnings
# from skopt.space import Categorical

# ----------------------------------------
# Execution mode:

# Whether to perform hyper-parameter optimization with `skopt` (True) or use a fixed set of hyper-parameters (False)
fit_hparams = False
# Whether to train and cross-validate the model using hyper-parameters either found by `skopt` or specified by user:
train_cv_model = True
# Whether to train a model on all input data using hyper-parameters either found by `skopt` or specified by user:
train_model = True
# Whether to use a trained model to make predictions:
predict = True
test = True

# ----------------------------------------
# General settings:

rootdir = "."            # all paths will be relative to this
figformat = 'png'
verbose = True
n_folds = 10                                        # number of CV folds
random_seed = 23

# ----------------------------------------
# I/O for training:

inputfile_train = "rrab_g_X_rv_gpr_param.dat"       # input file with object metadata
lc_subdir_train = "synlc_g"                         # subdirectory of the 'synthetic' light curves for training
lc_file_suffix_train = "_gpr_syn.dat"               # suffix of the synthetic light curves in the scheme of
                                                    #   <object_name><lc_file_suffix_train>
rv_subdir_train = "phasedrv"                        # subdirectory of the redial velocity data for training
rv_file_suffix_train = ".dat"                       # suffix of the radial velocity curves in the scheme of
                                                    #   <object_name><rv_file_suffix_train>

output_model_file = "bin/lc2rv_xgb_model.save"    # filename for saving the trained predictive model

figname_lc = "gaia_lc_all"
figname_rv = "gaia_rv_all"
val_pred_out = "val_pred.dat"                       # filename for siving the predictions for the validation data
                                                    # if `train_cv_model` is True
# Index of object to be highlighted in plots:
indx_highlight = 42

# ----------------------------------------
# I/O for prediction:
input_model_file = "bin/lc2rv_xgb_model.save"       # filename for loading the trained predictive model
inputfile_pred = "test.dat"                         # input file with the names and periods of the target objects
lc_subdir_pred = "synlc_g"                          # subdirectory of the 'synthetic' light curves for prediction
rv_subdir_pred = "phasedrv"                         # subdirectory of the redial velocity data for prediction
lc_file_suffix_pred = "_gpr_syn.dat"                # suffix of the light curves in the scheme of
                                                    #   <object_name><lc_file_suffix_pred>
rv_file_suffix_pred = ".dat"                        # suffix of the radial velocity curves in the scheme of
                                                    #   <object_name><rv_file_suffix_pred>

# ----------------------------------------
# Training dataset settings:

# The column names to be read from `inputfile_train`:
usecols = ['id', 'period', 'Nep', 'Nep_rv', 'totamp', 'totamp_rv', 'cost', 'cost_rv',
           'phcov', 'phcov_rv', 'snr', 'snr_rv', 'meanmag', 'remark']

# The names of the input features from `inputfile_train`:
input_feature_names = ['period']

# Data subset definition:
# subset_expr = 'phcov>0.9 and phcov_rv>0.8 and snr>50 and snr_rv>20 and snr_rv<300 and Nep_rv>10 and remark=="1"'
subset_expr = 'remark=="1" and snr_rv>30 and snr_rv<1000 and snr>30 and phcov>0.85 and phcov_rv>0.75 ' \
              'and cost<0.05 and cost_rv<9'

# ----------------------------------------
# Hyper-parameters settings:

# fixed settings to be used if `fit_hparams` is False:
learning_rate = 0.00716
subsample = 0.3977
n_estimators = 957
max_depth = 20

# hyper-parameter search space to be used by skopt if `fit_hparams` is True:
search_space = list()
search_space.append(Real(0.001, 0.3, 'log-uniform', name='learning_rate'))
search_space.append(Real(0.1, 0.8, 'uniform', name='subsample'))
search_space.append(Integer(100, 1500, 'uniform', name='n_estimators'))
search_space.append(Integer(2, 30, 'uniform', name='max_depth'))

# Settings for hyper-parameter optimization if `fit_hparams` is True
scoring = "neg_mean_squared_error"
n_init = 30                                             # number of initial points for GP optimizer of skopt
n_calls = 50                                            # number of samples to obtain for the GPR by skopt


warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
