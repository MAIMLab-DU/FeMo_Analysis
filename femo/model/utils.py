import numpy as np


def stratified_kfold(
        X_TPD_norm: np.ndarray,
        X_FPD_norm: np.ndarray,
        custom_ratio: int|None = None,
        num_folds: int = 5
):

    num_tpd = X_TPD_norm.shape[0]
    num_fpd = X_FPD_norm.shape[0]

    # Randomize the datasets
    np.random.seed(0)
    rand_num_TPD = np.random.permutation(X_TPD_norm.shape[0])
    X_TPD_norm_rand = X_TPD_norm[rand_num_TPD, :]

    rand_num_FPD = np.random.permutation(X_FPD_norm.shape[0])
    X_FPD_norm_rand = X_FPD_norm[rand_num_FPD, :]
    
    # Determine the number of data points per fold
    n_data_TPD_each_fold = X_TPD_norm.shape[0] // num_folds
    n_data_TPD_last_fold = X_TPD_norm.shape[0] - n_data_TPD_each_fold * (num_folds - 1)

    if custom_ratio is None:
        FPD_TPD_ratio = X_FPD_norm.shape[0] / X_TPD_norm.shape[0]
        n_data_FPD_each_fold = X_FPD_norm.shape[0] // num_folds
        n_data_FPD_last_fold = X_FPD_norm.shape[0] - n_data_FPD_each_fold * (num_folds - 1)
    else:
        FPD_TPD_ratio = custom_ratio
        n_data_FPD_each_fold = int(np.floor(FPD_TPD_ratio * n_data_TPD_each_fold))
        n_data_FPD_last_fold = int(np.floor(FPD_TPD_ratio * n_data_TPD_last_fold))

    # Initialize the K-fold containers
    X_K_fold = [None] * num_folds
    Y_K_fold = [None] * num_folds

    # Create the K-1 folds
    for i in range(num_folds - 1):
        start_TPD = i * n_data_TPD_each_fold
        end_TPD = (i + 1) * n_data_TPD_each_fold
        start_FPD = i * n_data_FPD_each_fold
        end_FPD = (i + 1) * n_data_FPD_each_fold

        X_K_fold[i] = np.vstack((X_TPD_norm_rand[start_TPD:end_TPD, :],
                                 X_FPD_norm_rand[start_FPD:end_FPD, :]))
        Y_K_fold[i] = np.concatenate((np.ones(end_TPD - start_TPD),
                                      np.zeros(end_FPD - start_FPD)))

    # Create the last fold
    start_TPD_last = (num_folds - 1) * n_data_TPD_each_fold
    end_TPD_last = start_TPD_last + n_data_TPD_last_fold
    start_FPD_last = (num_folds - 1) * n_data_FPD_each_fold
    end_FPD_last = start_FPD_last + n_data_FPD_last_fold

    X_K_fold[num_folds - 1] = np.vstack((X_TPD_norm_rand[start_TPD_last:end_TPD_last, :],
                                         X_FPD_norm_rand[start_FPD_last:end_FPD_last, :]))
    Y_K_fold[num_folds - 1] = np.concatenate((np.ones(end_TPD_last - start_TPD_last),
                                              np.zeros(end_FPD_last - start_FPD_last)))

    return {
        "X_K_fold": X_K_fold,
        "Y_K_fold": Y_K_fold,
        "num_folds": num_folds,
        "num_tpd": num_tpd,
        "num_fpd": num_fpd,
        "num_tpd_each_fold": n_data_TPD_each_fold,
        "num_tpd_last_fold": n_data_TPD_last_fold,
        "num_fpd_each_fold": n_data_FPD_each_fold,
        "num_fpd_last_fold": n_data_FPD_last_fold,
        "tpd_fpd_ratio": FPD_TPD_ratio,
        "rand_num_tpd": rand_num_TPD,
        "rand_num_fpd": rand_num_FPD
    }
