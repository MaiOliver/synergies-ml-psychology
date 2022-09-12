import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
# from matplotlib import rc
# rc("text", usetex=True)


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')


# location of data files; should in the future be a sub-folder of the current working directory
folder = "./data"

np.random.seed(1337)
hypothesis_two = pd.read_csv(folder + "/lr_rf_nn_hybrid_big_hypot_2.csv", delimiter=";", decimal=".")

Var_R2_test_models = {}
Var_RMSE_test_models = {}
Mean_R2_test_models = {}
Mean_RMSE_test_models = {}
procedures = hypothesis_two["Procedure"].unique()
n_samples = hypothesis_two["n_samples"].unique()
n_repeats = hypothesis_two["n_repeats"][0]

for proc in procedures:
    Var_R2_test_models[proc] = -hypothesis_two.loc[hypothesis_two["Procedure"] == proc, "Variance_R2test"]
    Var_RMSE_test_models[proc] = -hypothesis_two.loc[hypothesis_two["Procedure"] == proc, "Variance_RMSEtest"]

    Mean_R2_test_models[proc] = hypothesis_two.loc[hypothesis_two["Procedure"] == proc, "Mean_R2test"].clip(lower=0.0)
    # Mean_R2_test_models[proc] = np.clip(savitzky_golay(
    #    np.array(hypothesis_two.loc[hypothesis_two["Procedure"] == proc, "Mean_R2test"]), 9, 7),
    #    a_min=0.0, a_max=None)
    Mean_RMSE_test_models[proc] = hypothesis_two.loc[hypothesis_two["Procedure"] == proc, "Mean_RMSEtest"]


def make_plot(variables, n_fig, title, xlabel, ylabel, xlim=None, ylim=None, n_sample=n_samples):
    plt.figure(n_fig)
    alpha = 0.4
    colors = [[1., alpha, alpha], [alpha, 0.8, alpha], [alpha, alpha, 1.], "k", "#FFA500"]
    for i, proc in enumerate(procedures):
        c = colors[i]
        plt.plot(n_sample, variables[proc], label=proc, color=c)
    plt.legend()
    plt.tight_layout()
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)


title = f"{n_repeats} repetitions per sample"
xlable = "Sample Size"
ylabels = ["-Variance R²", "-Variance RMSE", "Mean R²", "Mean RMSE"]
for i, var in enumerate([Var_R2_test_models, Var_RMSE_test_models, Mean_R2_test_models, Mean_RMSE_test_models]):
    make_plot(var, i, title, xlable, ylabels[i])
plt.show()
