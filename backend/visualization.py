#!/usr/bin/env python3
"""
tpf_csv_anim.py

Create an animation from:
 - a target pixel file (downloaded via lightkurve), OR
 - a CSV file with columns time,flux,flux_err,kepid,disposition (header optional).

Modes:
 1) Use TPF frames directly:
    python tpf_csv_anim.py --tpf_target "KIC 8462852" --quarter 16 --out tpf_movie.mp4
    (uses the actual TPF frames)

 2) Use CSV to create frames using a TPF reference frame as template:
    python tpf_csv_anim.py your.csv --tpf_target "KIC 8462852" --quarter 16 --out scaled_movie.mp4
    (loads TPF to get spatial template, scales it per CSV flux)

 3) CSV-only (no TPF available) -> synthetic Gaussian star frames:
    python tpf_csv_anim.py your.csv --out synthetic.mp4

Notes:
 - MP4 saving requires ffmpeg installed. If missing, the script will try to create a GIF using Pillow.
 - Install dependencies in your venv:
    pip install numpy pandas matplotlib lightkurve astropy pillow
"""
from __future__ import annotations
import argparse
import os
import sys
import shutil
import math
from typing import Tuple, Optional

import numpy as np
import pandas as pd

# Use Agg so saving works on headless servers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.patches import Circle
from matplotlib.colors import LogNorm

# lightkurve may not be installed in user's environment; it's optional
try:
    from lightkurve import search_targetpixelfile
    _HAS_LIGHTKURVE = True
except Exception:
    _HAS_LIGHTKURVE = False

try:
    from astropy.visualization import ImageNormalize, AsinhStretch
    _HAS_ASTROPY_VIZ = True
except Exception:
    _HAS_ASTROPY_VIZ = False


# -------------------------
# CSV loader (fits your format)
# -------------------------
def load_lightcurve_csv(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reads CSV with header containing time,flux,flux_err (case-insensitive),
    or without header if columns are in order: time,flux,flux_err,kepid,disposition.

    Returns: (time, flux, flux_err) as 1D numpy arrays (float).
    """
    # try header first
    try:
        df = pd.read_csv(path, header=0)
        cols = [c.lower() for c in df.columns]
        if "time" in cols and "flux" in cols and "flux_err" in cols:
            time = df.iloc[:, cols.index("time")].astype(float).to_numpy()
            flux = df.iloc[:, cols.index("flux")].astype(float).to_numpy()
            flux_err = df.iloc[:, cols.index("flux_err")].astype(float).to_numpy()
            return time, flux, flux_err
    except Exception:
        # fall through to no-header attempt
        pass

    # fallback - assume no header, columns: time,flux,flux_err,...
    df2 = pd.read_csv(path, header=None)
    if df2.shape[1] >= 3:
        time = df2.iloc[:, 0].astype(float).to_numpy()
        flux = df2.iloc[:, 1].astype(float).to_numpy()
        flux_err = df2.iloc[:, 2].astype(float).to_numpy()
        return time, flux, flux_err

    raise RuntimeError("CSV not recognized: need time,flux,flux_err (header) or at least 3 columns (no header).")


# -------------------------
# TPF loader
# -------------------------
def download_tpf(target_name: str, quarter: Optional[int] = None):
    if not _HAS_LIGHTKURVE:
        raise RuntimeError("lightkurve is not installed in this environment. Install with `pip install lightkurve`.")
    print(f"Searching/downloading TPF for '{target_name}' quarter={quarter} ...")
    if quarter is None:
        search = search_targetpixelfile(target_name)
    else:
        search = search_targetpixelfile(target_name, quarter=quarter)
    if len(search) == 0:
        raise RuntimeError(f"No TPF found for {target_name} (quarter={quarter})")
    tpf = search.download()
    print("Downloaded TPF:", getattr(tpf, "filename", "<in-memory>"))
    return tpf


# -------------------------
# helpers to get frames (nframes, ny, nx)
# -------------------------
def get_frames_from_tpf(tpf) -> np.ndarray:
    """
    Return a numpy array of shape (nframes, ny, nx) from a lightkurve TargetPixelFile.
    Fills masked/bad values with 0.
    """
    # Many lightkurve tpf objects expose `flux` as a numpy array or MaskedArray
    arr = getattr(tpf, "flux", None)
    if arr is None:
        # try tpf.flux.value
        try:
            arr = tpf.flux.value
        except Exception:
            raise RuntimeError("Cannot extract flux array from TPF")
    # ensure numpy array and fill mask
    arr = np.asarray(arr)
    if hasattr(arr, "mask"):
        # masked array
        arr = np.where(np.asarray(arr.mask), 0.0, np.asarray(arr.data))
    # convert to float
    arr = arr.astype(float)
    return arr  # shape (nframes, ny, nx)


def build_scaled_frames_from_csv_and_tpf(csv_flux: np.ndarray, tpf_frames: np.ndarray) -> np.ndarray:
    """
    Build frames by scaling the TPF reference frame (median over time) by CSV flux values.
    - csv_flux: 1D array of fluxes (length nframes_csv)
    - tpf_frames: 3D array (nframes_tpf, ny, nx)
    Returns 3D array (nframes_csv, ny, nx)
    """
    ref = np.nanmedian(tpf_frames, axis=0)  # shape (ny, nx)
    median_flux = np.nanmedian(csv_flux[np.isfinite(csv_flux)])
    if median_flux == 0 or not np.isfinite(median_flux):
        median_flux = 1.0
    # scale factor per frame
    scale = (csv_flux / median_flux).astype(float)  # length nframes_csv
    # create frames
    frames = np.array([ref * s for s in scale])
    return frames


# -------------------------
# synthetic PSF fallback (CSV-only)
# -------------------------
def gaussian_psf(nx: int, ny: int, x0: float, y0: float, sigma: float) -> np.ndarray:
    y = np.arange(ny)
    x = np.arange(nx)
    xx, yy = np.meshgrid(x, y)
    rr2 = (xx - x0) ** 2 + (yy - y0) ** 2
    psf = np.exp(-0.5 * rr2 / (sigma ** 2))
    psf /= psf.max()
    return psf


def build_synthetic_frames_from_csv(csv_flux: np.ndarray, nx: int = 101, ny: int = 101,
                                   sigma: float = 2.0, peak_scale: float = 1000.0,
                                   flux_err: Optional[np.ndarray] = None, add_noise: bool = True) -> np.ndarray:
    median_flux = np.nanmedian(csv_flux[np.isfinite(csv_flux)])
    if median_flux == 0 or not np.isfinite(median_flux):
        median_flux = 1.0
    amplitudes = (csv_flux / median_flux) * peak_scale
    x0 = (nx - 1) / 2.0
    y0 = (ny - 1) / 2.0
    psf = gaussian_psf(nx, ny, x0, y0, sigma)
    frames = np.empty((len(amplitudes), ny, nx), dtype=float)
    for i, amp in enumerate(amplitudes):
        frame = amp * psf
        if add_noise and flux_err is not None and np.isfinite(flux_err[i]) and flux_err[i] > 0:
            noise_std = (flux_err[i] / median_flux) * peak_scale
            frame = frame + np.random.normal(0.0, noise_std, size=frame.shape)
        frames[i] = frame
    return frames


# -------------------------
# Animation builder from frames + optional lightcurve inset
# -------------------------
def animate_frames(frames: np.ndarray, time: Optional[np.ndarray] = None, flux: Optional[np.ndarray] = None,
                   outpath: Optional[str] = None, interval: int = 150, mark_center: bool = True,
                   stretch: str = "percentile", vmin_pct: float = 5.0, vmax_pct: float = 99.0) -> Tuple[FuncAnimation, plt.Figure]:
    """
    frames: ndarray (nframes, ny, nx)
    time, flux: optional 1D arrays for lightcurve inset and vertical marker
    """
    nframes, ny, nx = frames.shape
    print(f"Animating {nframes} frames of size {nx}x{ny} (interval={interval} ms)")

    # compute display normalization (global percentiles)
    stack_values = frames.reshape(-1)
    vmin = float(np.nanpercentile(stack_values, vmin_pct))
    vmax = float(np.nanpercentile(stack_values, vmax_pct))
    if vmin == vmax:
        vmin = np.nanmin(stack_values)
        vmax = np.nanmax(stack_values)
        if vmin == vmax:
            vmax = vmin + 1.0

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(frames[0], origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_xlim(-0.5, nx - 0.5)
    ax.set_ylim(-0.5, ny - 0.5)
    title = ax.set_title("")
    circ = None
    if mark_center:
        circ = Circle(((nx - 1) / 2.0, (ny - 1) / 2.0), radius=max(1.5, min(nx, ny) * 0.03),
                      edgecolor="red", facecolor="none", lw=1.0)
        ax.add_patch(circ)

    # add lightcurve inset if provided
    ax_lc = None
    vline = None
    if time is not None and flux is not None:
        ax_lc = fig.add_axes([0.62, 0.68, 0.33, 0.25])
        ax_lc.plot(time, flux, lw=0.8, alpha=0.9)
        vline = ax_lc.axvline(time[0], color="red", lw=1.0)
        ax_lc.set_xlabel("time")
        ax_lc.set_ylabel("flux")
        ax_lc.tick_params(labelsize=8)

    def init():
        im.set_data(frames[0])
        if vline is not None:
            vline.set_xdata(time[0])
        if circ is not None:
            # center stays same for template-based frames; if you want to mark peak, compute argmax
            circ.set_center(((nx - 1) / 2.0, (ny - 1) / 2.0))
        title.set_text(f"frame 0")
        return tuple([im] + ([circ] if circ is not None else []) + ([vline] if vline is not None else []))

    def update(i):
        im.set_data(frames[i])
        if vline is not None:
            vline.set_xdata(time[i])
        title.set_text(f"frame {i}   " + (f"time={time[i]:.6f} flux={flux[i]:.6g}" if (time is not None and flux is not None) else ""))
        return tuple([im] + ([circ] if circ is not None else []) + ([vline] if vline is not None else []))

    anim = FuncAnimation(fig, update, frames=range(nframes), init_func=init, interval=interval, blit=False)

    # saving logic with ffmpeg/Pillow fallback
    if outpath:
        outpath = str(outpath)
        parent = os.path.dirname(outpath)
        if parent and not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)

        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path:
            print("ffmpeg found:", ffmpeg_path)
        else:
            print("ffmpeg not found on PATH. will fallback to GIF via Pillow if possible (or fail).")

        print("Saving animation to:", outpath)
        try:
            if outpath.lower().endswith(".mp4"):
                if ffmpeg_path:
                    writer = FFMpegWriter(fps=max(1, int(1000 / interval)))
                    anim.save(outpath, writer=writer)
                    print("Saved MP4:", outpath)
                else:
                    # fallback to GIF via Pillow
                    gif_path = os.path.splitext(outpath)[0] + ".gif"
                    print("No ffmpeg -> falling back to GIF:", gif_path)
                    writer = PillowWriter(fps=max(1, int(1000 / interval)))
                    anim.save(gif_path, writer=writer)
                    print("Saved GIF fallback:", gif_path)
            elif outpath.lower().endswith(".gif"):
                writer = PillowWriter(fps=max(1, int(1000 / interval)))
                anim.save(outpath, writer=writer)
                print("Saved GIF:", outpath)
            else:
                anim.save(outpath)
                print("Saved animation:", outpath)
        except Exception as e:
            print("ERROR saving animation:", repr(e))
            raise
        finally:
            plt.close(fig)
    return anim, fig


# -------------------------
# CLI and orchestration
# -------------------------
def parse_args(argv):
    p = argparse.ArgumentParser(description="Create an animation from a TPF and/or CSV (time,flux,flux_err,...)")
    p.add_argument("csv", nargs="?", help="CSV file (time,flux,flux_err,...) — optional (if omitted and --tpf_target given, uses tpf frames).")
    p.add_argument("--tpf_target", type=str, help='Search string for lightkurve, e.g. "KIC 8462852"')
    p.add_argument("--quarter", type=int, default=None, help="Quarter (for tpf search)")
    p.add_argument("--tpf_file", type=str, help="Local TPF file path (.fits) to use instead of downloading")
    p.add_argument("--out", "-o", help="Output file (.mp4 or .gif)", required=True)
    p.add_argument("--interval", type=int, default=150, help="ms between frames")
    p.add_argument("--nx", type=int, default=101, help="Width for synthetic frames (if CSV-only)")
    p.add_argument("--ny", type=int, default=101, help="Height for synthetic frames (if CSV-only)")
    p.add_argument("--sigma", type=float, default=2.0, help="PSF sigma for synthetic frames")
    p.add_argument("--peak", type=float, default=1000.0, help="Peak scaling for synthetic frames")
    p.add_argument("--no-noise", action="store_true", help="Do not add noise to synthetic frames")
    p.add_argument("--use_tpf_frames", action="store_true", help="If using a TPF, use its real frames instead of scaling a reference frame")
    return p.parse_args(argv)


def main(argv):
    args = parse_args(argv)

    csv_path = args.csv
    tpf_obj = None
    tpf_frames = None
    csv_time = csv_flux = csv_flux_err = None

    # load CSV if provided
    if csv_path:
        if not os.path.exists(csv_path):
            print("CSV not found:", csv_path); sys.exit(2)
        print("Loading CSV:", csv_path)
        try:
            csv_time, csv_flux, csv_flux_err = load_lightcurve_csv(csv_path)
            print(f"Loaded CSV rows: {len(csv_flux)}")
        except Exception as e:
            print("Failed to read CSV:", e); sys.exit(3)

    # load/download TPF if requested
    if args.tpf_file:
        if not os.path.exists(args.tpf_file):
            print("TPF file not found:", args.tpf_file); sys.exit(4)
        if not _HAS_LIGHTKURVE:
            print("lightkurve not installed; cannot load local TPF file with lightkurve. Install lightkurve.")
            sys.exit(5)
        from lightkurve import TargetPixelFile
        tpf_obj = TargetPixelFile(args.tpf_file)
        tpf_frames = get_frames_from_tpf(tpf_obj)
        print("Loaded local TPF frames shape:", tpf_frames.shape)
    elif args.tpf_target:
        try:
            tpf_obj = download_tpf(args.tpf_target, quarter=args.quarter)
            tpf_frames = get_frames_from_tpf(tpf_obj)
            print("Downloaded TPF frames shape:", tpf_frames.shape)
        except Exception as e:
            print("Failed to get TPF:", e)
            # continue — we can fallback to synthetic frames if CSV present
            tpf_obj = None
            tpf_frames = None

    frames = None

    # Case A: TPF frames requested to be used directly (and CSV not provided, or use_tpf_frames set)
    if tpf_frames is not None and args.use_tpf_frames and (csv_path is None):
        print("Using TPF frames directly for animation.")
        frames = tpf_frames  # shape (n, ny, nx)

    # Case B: CSV provided and TPF available -> scale reference frame per flux
    if csv_path and tpf_frames is not None and not args.use_tpf_frames:
        print("Building frames by scaling TPF reference frame using CSV flux values.")
        frames = build_scaled_frames_from_csv_and_tpf(csv_flux, tpf_frames)
        print("Built frames shape:", frames.shape)

    # Case C: CSV-only -> synthetic frames
    if frames is None and csv_path:
        print("Building synthetic frames from CSV (no TPF).")
        frames = build_synthetic_frames_from_csv(csv_flux, nx=args.nx, ny=args.ny,
                                                 sigma=args.sigma, peak_scale=args.peak,
                                                 flux_err=csv_flux_err, add_noise=not args.no_noise)
        print("Synthetic frames shape:", frames.shape)

    # Case D: no CSV, but tpf frames exist -> animate tpf frames
    if frames is None and tpf_frames is not None:
        print("No CSV provided; animating TPF's native frames.")
        frames = tpf_frames

    if frames is None:
        print("Nothing to animate. Provide a CSV or a TPF target/file."); sys.exit(6)

    # sanity: large number of frames warning
    if frames.shape[0] > 2000:
        print(f"WARNING: You are about to animate {frames.shape[0]} frames. This may take a long time and produce a large file.")

    # animate and save
    try:
        anim, fig = animate_frames(frames, time=csv_time, flux=csv_flux, outpath=args.out, interval=args.interval)
        print("Animation complete.")
    except Exception as e:
        print("Error creating/saving animation:", repr(e))
        sys.exit(7)


if __name__ == "__main__":
    main(sys.argv[1:])
