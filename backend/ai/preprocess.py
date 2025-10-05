import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
import batman
import pandas as pd
from extract_features_custom import extract_features_dataframe


def remove_cosmic_rays(time, flux, n_sigma=5):
    """
    Technically as sigma clipping
    """
    flux = flux.copy()
    for i in range(1, len(flux)-1):
        local_mean = (flux[i-1] + flux[i+1]) / 2
        local_std = np.std([flux[i-1], flux[i+1]])
        if flux[i] > local_mean + n_sigma * local_std: # removes outliers with values more than n_sigma(5 in our case) standard deviations from the local mean(mean of neighbors)
            flux[i] = local_mean
    return flux


def detrend_flux(time, flux, flux_err=None, n_bins=200, n_iter=5, sigma_clip=3, s=1e-3):
    """
    Returns time_masked, flux_masked[, flux_err_masked (optional)]
    """
    time = np.asarray(time)
    flux = np.asarray(flux)
    if flux_err is not None:
        flux_err = np.asarray(flux_err)
    time = time[:len(flux)]
    # 1) удалить NaN и привести все к одной длине
    finite = np.isfinite(time) & np.isfinite(flux)
    if flux_err is not None:
        finite &= np.isfinite(flux_err)
    if finite.sum() == 0:
        # пустая кривая
        if flux_err is not None:
            return np.array([]), np.array([]), np.array([])
        return np.array([]), np.array([])

    time = time[finite]
    flux = flux[finite]
    flux = flux[:len(time)]
    flux_err = flux_err[:len(time)]
    time = time[:len(flux)]
    if flux_err is not None:
        flux_err = flux_err[finite]

    n = len(flux)
    mask = np.ones(n, dtype=bool)

    for _ in range(n_iter):
        if mask.sum() < 4:
            break

        bins = np.linspace(time.min(), time.max(), n_bins + 1)
        inds = np.digitize(time, bins)  # 1..n_bins
        bin_centers = []
        bin_means = []
        for b in range(1, n_bins + 1):
            sel = (inds == b) & mask
            if sel.any():
                bin_centers.append(time[sel].mean())
                bin_means.append(flux[sel].mean())

        if len(bin_centers) < 4:
            break

        try:
            spline = UnivariateSpline(bin_centers, bin_means, s=s)
            model = spline(time)
            model = np.where(model == 0, 1.0, model)
        except Exception:
            break

        flat = flux / model
        med = np.median(flat[mask])
        std = np.std(flat[mask])
        new_mask = flat > (med - sigma_clip * std)

        if new_mask.shape != mask.shape:
            new_mask = new_mask[:mask.shape[0]]

        mask = mask & new_mask

    if flux_err is None:
        return time[mask], flux[mask]
    else:
        return time[mask], flux[mask], flux_err[mask]


def resample_uniform(time, flux, flux_err=None, dt=29.4/60/24):  # dt in days (default Kepler ~29.4 min)
    time = np.asarray(time)
    flux = np.asarray(flux)
    if flux_err is not None:
        flux_err = np.asarray(flux_err)

    # ensure monotonic increasing time
    order = np.argsort(time)
    time = time[order]
    flux = flux[order]
    if flux_err is not None:
        flux_err = flux_err[order]

    new_time = np.arange(time.min(), time.max(), dt)
    if len(time) < 2 or len(new_time) == 0:
        return time, flux, flux_err if flux_err is not None else (time, flux)

    f_interp = interp1d(time, flux, kind='linear', bounds_error=False, fill_value="extrapolate")
    new_flux = f_interp(new_time)

    if flux_err is not None:
        e_interp = interp1d(time, flux_err, kind='linear', bounds_error=False, fill_value="extrapolate")
        new_err = e_interp(new_time)
        return new_time, new_flux, new_err
    return new_time, new_flux, None


def compute_snr(flux):
    flux = np.asarray(flux)
    if len(flux) == 0:
        return 0.0
    signal = np.nanmax(flux) - np.nanmin(flux)
    noise = np.nanstd(flux)
    return 0.0 if noise == 0 else signal / noise


def inject_transit(
    time,
    flux,
    rp=None,         # Rp/R*, float or None (then sampled)
    a=None,          # a/R*, semi-major axis
    inc=None,        # inclination in degrees
    t0=None,         # mid-transit time
    period=None,     # orbital period (irrelevant for single transit)
    u=None,          # limb darkening coefficients [u1, u2]
    randomize=True,  # if True - samples random parameters when None
    supersample_factor=7,  # Long exposure time
    exp_time=0.0007,       # duration of exposure (in days for Kepler/TESS)
    seed=None
):
    """
    Injects synthetic transit into the light curve.
    Returns flux_injected, flux_model, used_params
    """
    rng = np.random.default_rng(seed)

    if randomize:
        rp = rp if rp is not None else rng.uniform(0.01, 0.15)
        a = a if a is not None else rng.uniform(5, 25)
        inc = inc if inc is not None else rng.uniform(85, 90)
        t0 = t0 if t0 is not None else rng.choice(time)
        period = period if period is not None else rng.uniform(1, 10)
        u = u if u is not None else [rng.uniform(0.1, 0.6), rng.uniform(0.0, 0.5)]
    else:
        if any(x is None for x in [rp, a, inc, t0, period, u]):
            raise ValueError("If randomize=False, all parameters must be provided.")


    params = batman.TransitParams()
    params.t0 = t0
    params.per = period
    params.rp = rp
    params.a = a
    params.inc = inc
    params.ecc = 0.0
    params.w = 90.0
    params.u = u
    params.limb_dark = "quadratic"

    model = batman.TransitModel(params, time,
                                supersample_factor=supersample_factor,
                                exp_time=exp_time)
    flux_model = model.light_curve(params)

    flux_injected = flux * flux_model # Injection

    used_params = dict(rp=rp, a=a, inc=inc, t0=t0, period=period, u=u)
    return flux_injected, flux_model, used_params


def preprocess_row(time, flux, flux_err=None, dt=0.0204, snr_thresh=12, inject_transits=True, transit_injection_rate=0.4):
    """
    Preprocesses a single row of the dataset.

    """
    flux = remove_cosmic_rays(time, flux)
    time, flux, flux_err = detrend_flux(time, flux, flux_err)
    time, flux, flux_err = resample_uniform(time, flux, flux_err, dt=dt)
    if compute_snr(flux) > snr_thresh:
        return None, None, None
    if inject_transits:
        if np.random.random() < transit_injection_rate:
            flux, flux_model, used_params = inject_transit(time, flux)
    return time, flux, flux_err


def preprocess_dataset(df, dt=0.0204, snr_thresh=12, inject_transits=True, transit_injection_rate=0.4):
    processed_rows = []

    for idx, row in df.iterrows():
        time = row['time']
        flux = row['flux']
        flux_err = row['flux_err']

        time, flux, flux_err = preprocess_row(time, flux, flux_err, dt=dt, snr_thresh=snr_thresh, inject_transits=inject_transits, transit_injection_rate=transit_injection_rate)

        if time is None:
            continue
        
        flux = flux[:len(time)]
        time = time[:len(flux)]
        flux_err = flux_err[:len(flux)]

        processed_rows.append({
            "time": time,
            "flux": flux,
            "flux_err": flux_err,
            "label": row['label']
        })

    return pd.DataFrame(processed_rows)


def pipeline_df(df, dt=0.0204, snr_thresh=12, inject_transits=False, transit_injection_rate=0.4):
    df = preprocess_dataset(df, dt=dt, snr_thresh=snr_thresh, inject_transits=inject_transits, transit_injection_rate=transit_injection_rate)
    df = extract_features_dataframe(df, use_flux_err=True)
    return df