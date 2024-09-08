import pywt
import numpy as np
from scipy.fft import fft
from scipy.integrate import simpson
from scipy.stats import skew, kurtosis
from scipy.signal import convolve, hilbert, spectrogram


def get_frequency_mode(data, sampling_freq, trim_length):
    """This function calculates the frequency spectrum of the input data, trims the spectrum,
    and identifies the main frequency mode.

    Args:
        data (numpy.ndarray): Input data array representing the signal.
        sampling_freq (int): Sampling frequency of the input data in Hz.
        trim_length (int): Length in seconds up to which the frequency spectrum needs to be trimmed.

    Returns:
        tuple: A tuple containing three elements:
            spectrum (numpy.ndarray): Trimmed frequency spectrum of the input data.
            f_vector (numpy.ndarray): Trimmed frequency vector corresponding to the spectrum.
            f_mode (float): Main frequency mode identified from the trimmed spectrum.
    """
    # -----------Generating the frequency vector of the spectrum---------------
    length = len(data)
    # Index upto which trimming needs to be done. L/Fs gives number of points/Hz in the frequency spectrum.
    # When multiplied by trim_length, it gives the index
    # upto which trimming is necessary 
    freq_trim_indx = int(np.ceil(trim_length * length / sampling_freq + 1))
    # There are L/2 points in the single-sided spectrum.
    # Each point will be Fs/L apart.
    freq_vector = sampling_freq * np.arange(int(length / 2) + 1) / length
    # frequency vector after trimming
    freq_vector_trimmed = freq_vector[freq_trim_indx:]


    # --------------------------Fourier transform------------------------------
    # Two-sided Fourier spectrum. Normalizing by L is generally performed during fft so that it is not neede for inverse fft
    FT_2_sided = np.abs(fft(data)) / length
    # Floor is used for the cases where L is an odd number
    FT_data_1_sided = FT_2_sided[:int(length / 2) + 1]
    # multiplication by 2 is used to maintain the conservation of energy
    FT_data_1_sided[1:-1] = 2 * FT_data_1_sided[1:-1]
    FT_data_1_sided_trimmed = FT_data_1_sided[freq_trim_indx:]

    # ---------------------------Main frequency mode---------------------------
    # corresponding index
    index = np.argmax(FT_data_1_sided_trimmed)
    # cell matrix with main frequency mode
    freq_mode_data = freq_vector_trimmed[index]

    # ---------------------------Finalizing the output variables---------------
    f_vector = freq_vector_trimmed
    spectrum = FT_data_1_sided_trimmed
    f_mode = freq_mode_data

    return spectrum, f_vector, f_mode


def get_band_energy(data, sampling_freq, min_freq, max_freq):
    """Calculates the energy within a specific frequency band of a given signal
    using the Fast Fourier Transform (FFT).

    Args:
        data (array_like): Signal data. Must be real.
        sampling_freq (int): Sampling frequency of the signal.
        min_freq (int): Lower bound of the frequency band (in Hz).
        max_freq (int): Upper bound of the frequency band (in Hz).

    Returns:
        float: Sum of the energies within the specified frequency band.
    """
    n_sample = len(data)

    # Y = fft(X) computes the discrete Fourier transform (DFT) of X using a fast Fourier transform (FFT) algorithm.
    FFTX = fft(data)
    # power = (abs(FFTX(1:floor(numsample/2+1))).^2. %Power: magnitude^2
    power = np.abs(FFTX[:int(n_sample / 2) + 1 ]) ** 2
    # Computing the corresponding frequency values
    freq_bin = np.linspace(0, sampling_freq/2, int(n_sample/2)+1)

    energy_range = np.sum(power[(min_freq <= freq_bin) & (freq_bin <= max_freq)])
    return energy_range


def get_inst_amplitude(data):
    # Apply the Hilbert transform to the input signal to obtain the analytic signal
    hx = hilbert(data)

    # Calculate the instantaneous amplitude as the absolute value of the analytic signal
    inst_amplitude = np.abs(hx)
    return inst_amplitude


def get_inst_frequency(data, sampling_freq):

    # Apply the Hilbert transform to the input signal to obtain the analytic signal 
    # analytic signal = which is a complex-valued signal representing the original signal's amplitude and phase information
    hx = hilbert(data)
    
    # Calculate the instantaneous phase of the analytic signal and unwrap it to avoid phase wrapping issues
    phase = np.unwrap(np.angle(hx))
    
    # Calculate the instantaneous frequency by taking the derivative of the phase and dividing by the sampling frequency of the sensor
    inst_freq = np.diff(phase) * sampling_freq / (2 * np.pi) # division by 2π is required to convert the phase difference (in radians) into frequency
  
    return inst_freq


def get_wavelet_coeff(data):
    # Name of the wavelet to be used for transformation.
    wavelet_name = 'haar'
    # Perform discrete wavelet transform
    wavelet_coeff, _ = pywt.dwt(data, wavelet_name)

    # 'wavelet_coeff' contains the wavelet coefficients, and the ignored output is typically the approximation coefficients. 
    # One can use 'wavelet_coeff' for further analysis or processing.
    return wavelet_coeff


def get_conv1D(convolved_signal, key):

    def calculate_entropy(x):
        # Monaf's implementation
        # Flatten the array and count occurrences of each unique value
        unique, counts = np.unique(x.flatten(), return_counts=True) 
        # Compute probabilities
        probs = counts / np.sum(counts)
        # Compute and return entropy
        return -np.sum(probs * np.log2(probs))
    
    def calculate_crest_factor(x):
        peak_value = np.max(np.abs(x))
        rms_value = np.sqrt(np.mean(x**2))
        # Crest factor
        return peak_value / rms_value 
    
    operations = {
        1: lambda x: np.mean(x),
        2: lambda x: np.std(x),
        3: lambda x: np.max(x),
        4: lambda x: np.min(x),
        5: lambda x: np.ptp(x),
        6: lambda x: np.max(np.abs(x)),
        7: lambda x: np.sum(x**2),
        # Zero crossing rate
        8: lambda x: len(np.where(np.diff(np.sign(x)))[0]) / len(x),
        9: lambda x: np.mean(np.gradient(x)),
        10: lambda x: np.max(x) - np.min(x),
        11: lambda x: skew(x),
        12: lambda x: kurtosis(x),
        13: lambda x: np.sqrt(np.mean(x**2)),
        14: lambda x: np.percentile(x, 75) - np.percentile(x, 25),
        15: lambda x: np.mean(np.abs(x - np.mean(x))),
        16: lambda x: np.mean(np.abs(x)),
        17: lambda x: np.sum(x[x > 0]),
        18: lambda x: np.sum(x[x < 0]),
        # Mean crossing rate
        19: lambda x: len(np.where(np.diff(np.sign(x - np.mean(x))))[0]) / len(x),
        20: lambda x: calculate_entropy(x),
        21: lambda x: calculate_crest_factor(x)
    }
    
    return operations.get(key, lambda x: None)(convolved_signal)


def get_convolved_signal(data):
    # Define the Gaussian filter kernel
    sigma = 1  # Standard deviation
    filter_length = 7  # Filter length (odd number)

    # Create a time vector for the filter kernel
    t = np.arange(-(filter_length-1)/2, (filter_length-1)/2 + 1)

    # Calculate the Gaussian filter values for the time vector
    F = np.exp(-t**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))

    # Perform 1D convolution using the convolve function
    convolved_signal = convolve(data, F, mode='same')
    return convolved_signal


def get_time_freq_mod_filter(S):
    # Set parameters
    windowSize = 256
    hopSize = 128
    timeOffset = 50
    frequencyOffset = 10

    # Compute the STFT of the signal  # Shape of f will be (n_freq,) ||| # Shape of t will be (n_time,)
    f, t, S_tf = spectrogram(S, window='hann', nperseg=windowSize, noverlap=windowSize - hopSize)

    # Define the reference template equation with time offset
    referenceTemplate = 2**(1/4) * np.exp(-np.pi * ((t[:, np.newaxis] - timeOffset)**2)) * np.exp(-1j * 2 * np.pi * f[np.newaxis, :] * frequencyOffset)
    # Perform TFMF with time offset
    correlation = np.sum(np.sum(np.dot(S_tf,np.conj(referenceTemplate)))) # This works. but it all gives the same value near to 0
    
    output = np.abs(correlation)
    return output


def get_morphology(signal, sampling_freq):

    time_interval = 1 / sampling_freq
    
    # Calculate absolute area
    absolute_area = simpson(np.abs(signal), dx=time_interval)
    
    # Calculate differential
    differential = np.diff(signal) # Taking numerical derivative
    
    # Calculate absolute area of the differential
    absolute_area_differential = simpson(np.abs(differential), dx=time_interval)
    
    # Calculate relative area
    relative_area = absolute_area_differential / absolute_area * 100
    # relative_area = absolute_area / total_samples
    # relative_area = np.sum(signal) * time_interval
    
    return absolute_area, relative_area, absolute_area_differential

