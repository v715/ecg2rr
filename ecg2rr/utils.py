import numpy as np


def resample_ann(resampled_t, ann_sample):
    """
    Compute the new annotation indices.
    Parameters
    ----------
    resampled_t : ndarray
        Array of signal locations as returned by scipy.signal.resample.
    ann_sample : ndarray
        Array of annotation locations.
    Returns
    -------
    ndarray
        Array of resampled annotation locations.
    """
    tmp = np.zeros(len(resampled_t), dtype="int16")
    j = 0
    break_loop = 0
    tprec = resampled_t[j]
    for i, v in enumerate(ann_sample):
        break_loop = 0
        while True:
            d = False
            if v < tprec:
                j -= 1
                tprec = resampled_t[j]

            if j + 1 == len(resampled_t):
                tmp[j] += 1
                break

            tnow = resampled_t[j + 1]
            if tprec <= v and v <= tnow:
                if v - tprec < tnow - v:
                    tmp[j] += 1
                else:
                    tmp[j + 1] += 1
                d = True
            j += 1
            tprec = tnow
            break_loop += 1
            if break_loop > 1000:
                tmp[j] += 1
                break
            if d:
                break

    idx = np.where(tmp > 0)[0].astype("int64")
    res = []
    for i in idx:
        for j in range(tmp[i]):
            res.append(i)
    assert len(res) == len(ann_sample)

    return np.asarray(res, dtype="int64")


def normalize_bound(sig, lb=0, ub=1):
    """
    Normalize a signal between the lower and upper bound.
    Parameters
    ----------
    sig : ndarray
        Original signal to be normalized.
    lb : int, float, optional
        Lower bound.
    ub : int, float, optional
        Upper bound.
    Returns
    -------
    ndarray
        Normalized signal.
    """
    mid = ub - (ub - lb) / 2
    min_v = np.min(sig)
    max_v = np.max(sig)
    mid_v = max_v - (max_v - min_v) / 2
    coef = (ub - lb) / (max_v - min_v)
    return sig * coef - (mid_v * coef) + mid


def correct_peaks(
    sig, peak_inds, search_radius, smooth_window_size, peak_dir="compare"
):
    """
    Adjust a set of detected peaks to coincide with local signal maxima.
    Parameters
    ----------
    sig : ndarray
        The 1d signal array.
    peak_inds : np array
        Array of the original peak indices.
    search_radius : int
        The radius within which the original peaks may be shifted.
    smooth_window_size : int
        The window size of the moving average filter applied on the
        signal. Peak distance is calculated on the difference between
        the original and smoothed signal.
    peak_dir : str, optional
        The expected peak direction: 'up' or 'down', 'both', or
        'compare'.
        - If 'up', the peaks will be shifted to local maxima.
        - If 'down', the peaks will be shifted to local minima.
        - If 'both', the peaks will be shifted to local maxima of the
          rectified signal.
        - If 'compare', the function will try both 'up' and 'down'
          options, and choose the direction that gives the largest mean
          distance from the smoothed signal.
    Returns
    -------
    shifted_peak_inds : ndarray
        Array of the corrected peak indices.
    """
    sig_len = sig.shape[0]
    n_peaks = len(peak_inds)

    # Subtract the smoothed signal from the original
    sig = sig - smooth(sig=sig, window_size=smooth_window_size)

    # Shift peaks to local maxima
    if peak_dir == "up":
        shifted_peak_inds = shift_peaks(
            sig=sig, peak_inds=peak_inds, search_radius=search_radius, peak_up=True
        )
    elif peak_dir == "down":
        shifted_peak_inds = shift_peaks(
            sig=sig, peak_inds=peak_inds, search_radius=search_radius, peak_up=False
        )
    elif peak_dir == "both":
        shifted_peak_inds = shift_peaks(
            sig=np.abs(sig),
            peak_inds=peak_inds,
            search_radius=search_radius,
            peak_up=True,
        )
    else:
        shifted_peak_inds_up = shift_peaks(
            sig=sig, peak_inds=peak_inds, search_radius=search_radius, peak_up=True
        )
        shifted_peak_inds_down = shift_peaks(
            sig=sig, peak_inds=peak_inds, search_radius=search_radius, peak_up=False
        )

        # Choose the direction with the biggest deviation
        up_dist = np.mean(np.abs(sig[shifted_peak_inds_up]))
        down_dist = np.mean(np.abs(sig[shifted_peak_inds_down]))

        if up_dist >= down_dist:
            shifted_peak_inds = shifted_peak_inds_up
        else:
            shifted_peak_inds = shifted_peak_inds_down

    return shifted_peak_inds


def shift_peaks(sig, peak_inds, search_radius, peak_up):
    """
    Helper function for correct_peaks. Return the shifted peaks to local
    maxima or minima within a radius.
    Parameters
    ----------
    sig : ndarray
        The 1d signal array.
    peak_inds : np array
        Array of the original peak indices.
    search_radius : int
        The radius within which the original peaks may be shifted.
    peak_up : bool
        Whether the expected peak direction is up.
    Returns
    -------
    shifted_peak_inds : ndarray
        Array of the corrected peak indices.
    """
    sig_len = sig.shape[0]
    n_peaks = len(peak_inds)
    # The indices to shift each peak ind by
    shift_inds = np.zeros(n_peaks, dtype="int")

    # Iterate through peaks
    for i in range(n_peaks):
        ind = peak_inds[i]
        local_sig = sig[
            max(0, ind - search_radius) : min(ind + search_radius, sig_len - 1)
        ]

        if peak_up:
            shift_inds[i] = np.argmax(local_sig)
        else:
            shift_inds[i] = np.argmin(local_sig)

    # May have to adjust early values
    for i in range(n_peaks):
        ind = peak_inds[i]
        if ind >= search_radius:
            break
        shift_inds[i] -= search_radius - ind

    shifted_peak_inds = peak_inds + shift_inds - search_radius

    return shifted_peak_inds


def smooth(sig, window_size):
    """
    Apply a uniform moving average filter to a signal.
    Parameters
    ----------
    sig : ndarray
        The signal to smooth.
    window_size : int
        The width of the moving average filter.
    Returns
    -------
    ndarray
        The convolved input signal with the desired box waveform.
    """
    box = np.ones(window_size) / window_size
    return np.convolve(sig, box, mode="same")
