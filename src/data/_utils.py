import hashlib
import numpy as np


def gen_hash(input_string: str) -> str:
    # Use SHA-256 to generate a hash of the input string
    sha256 = hashlib.sha256()
    sha256.update(input_string.encode('utf-8'))  # Convert the input string to bytes and hash it
    full_hash = sha256.hexdigest()[-4:]  # Get the full hash as a hexadecimal string
    full_hash = int(''.join([str(ord(c)) for c in full_hash]))
    return full_hash


def divide_by_K_folds(X_TPD_norm, X_FPD_norm, data_div_option, K):
    # Division into stratified K-fold
    #   2: K-fold with original ratio of FPD and TPD in each fold,
    #   3: K-fold with custom ratio of FPD and TPD in each fold.
    #   However, each of these processes will create stratified division, i.e.
    #   each fold will have the same ratio of FPD and TPD.

    # Randomization of the data sets
    n_data_TPD = X_TPD_norm.shape[0]
    np.random.seed(0)  # sets random seed to default value. This is necessary for reproducibility.
    rand_num_TPD = np.random.permutation(n_data_TPD)
    X_TPD_norm_rand = X_TPD_norm[rand_num_TPD, :]

    n_data_FPD = X_FPD_norm.shape[0]
    np.random.seed(0)  # sets random seed to default value. This is necessary for reproducibility.
    rand_num_FPD = np.random.permutation(n_data_FPD)
    X_FPD_norm_rand = X_FPD_norm[rand_num_FPD, :]

    custom_FPD_TPD_ratio = 1  # FPD/TPD ratio in case of option 3

    # Determining the data in each folds
    n_data_TPD_each_fold = n_data_TPD // K
    n_data_TPD_last_fold = n_data_TPD - n_data_TPD_each_fold * (K - 1)  # Last fold will hold the rest of the data

    if data_div_option == 2:
        # -- Based on K-fold partition with original ratio of TPDs and FPDs ---
        FPD_TPD_ratio = n_data_FPD / n_data_TPD

        n_data_FPD_each_fold = n_data_FPD // K
        n_data_FPD_last_fold = n_data_FPD - n_data_FPD_each_fold * (K - 1)  # Last fold will hold the rest of the data

    elif data_div_option == 3:
        FPD_TPD_ratio = custom_FPD_TPD_ratio

        n_data_FPD_each_fold = np.floor(FPD_TPD_ratio * n_data_TPD_each_fold)  # Each fold will have this number of FPDs
        n_data_FPD_last_fold = np.floor(FPD_TPD_ratio * n_data_TPD_last_fold)  # Last fold will hold the rest of the data

    else:
        print('\nWrong data division option chosen.\n')
        return

    # Creating K-fold data
    X_K_fold = [None] * K  # List variable to hold the each data fold
    Y_K_fold = [None] * K

    for i in range(1, K):
        start_idx_TPD = (i - 1) * n_data_TPD_each_fold
        end_idx_TPD = i * n_data_TPD_each_fold

        start_idx_FPD = (i - 1) * n_data_FPD_each_fold
        end_idx_FPD = i * n_data_FPD_each_fold

        X_K_fold[i - 1] = np.vstack((X_TPD_norm_rand[start_idx_TPD:end_idx_TPD, :],
                                     X_FPD_norm_rand[start_idx_FPD:end_idx_FPD, :]))
        Y_K_fold[i - 1] = np.concatenate((np.ones(end_idx_TPD - start_idx_TPD),
                                          np.zeros(end_idx_FPD - start_idx_FPD)))
        
    #For Last fold
    start_idx_TPD_last = (K - 1) * n_data_TPD_each_fold
    end_idx_TPD_last = (K - 1) * n_data_TPD_each_fold + n_data_TPD_last_fold

    start_idx_FPD_last = (K - 1) * n_data_FPD_each_fold
    end_idx_FPD_last = (K - 1) * n_data_FPD_each_fold + n_data_FPD_last_fold

    X_K_fold[K - 1] = np.vstack((X_TPD_norm_rand[start_idx_TPD_last:end_idx_TPD_last, :],
                                 X_FPD_norm_rand[start_idx_FPD_last:end_idx_FPD_last, :]))
    Y_K_fold[K - 1] = np.concatenate((np.ones(end_idx_TPD_last - start_idx_TPD_last),
                                      np.zeros(end_idx_FPD_last - start_idx_FPD_last)))

    return X_K_fold, Y_K_fold, n_data_TPD_each_fold, n_data_TPD_last_fold, n_data_FPD_each_fold, n_data_FPD_last_fold, FPD_TPD_ratio, rand_num_TPD, rand_num_FPD
