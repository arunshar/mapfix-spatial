"""Reproduced georeferencing evidence: thin-plate-spline (TPS) GCP fit against
the REAL Web Mercator distortion, with held-out ground RMSE.

The interactive demo (`app.html`/`server.py`) uses a synthetic parametric
"Mercator repair" profile. This script instead uses the genuine Web Mercator
projection as the ground-truth distortion over a real geographic grid of ground
control points (GCPs), and reports held-out RMSE for three registrations:

  raw    : read the Web Mercator map as if it were equirectangular (no fit)
  affine : 1st-order (6-parameter) polynomial fit on the training GCPs
  tps    : thin-plate spline fit on the training GCPs

The distortion is 100% real (Web Mercator stretches latitude nonlinearly toward
the poles); no simulated noise is added. A linear/affine registration leaves the
curvature residual that the TPS removes. RMSE is reported as true ground distance
(haversine) on held-out GCPs. Pure numpy, deterministic, no network, no GPU.

Run:  python scripts/evidence_gcp_rmse.py
"""
from __future__ import annotations

import math

import numpy as np

EARTH_R_M = 6_371_000.0


def web_mercator_pseudo(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Web Mercator in 'pseudo-degree' map units: x=lon, y=Mercator-y in degrees.

    A map drawn in Web Mercator and read as equirectangular places a point at
    (lon, mercator_y_deg) instead of its true (lon, lat); that gap is the real
    distortion this project repairs.
    """
    latc = np.clip(lat, -85.0, 85.0)
    y = np.degrees(np.log(np.tan(np.pi / 4 + np.radians(latc) / 2)))
    return np.stack([lon, y], axis=-1)


def haversine_m(lon1, lat1, lon2, lat2) -> np.ndarray:
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
    return 2 * EARTH_R_M * np.arcsin(np.sqrt(a))


def ground_rmse(pred_lonlat: np.ndarray, true_lonlat: np.ndarray) -> float:
    d = haversine_m(pred_lonlat[:, 0], pred_lonlat[:, 1], true_lonlat[:, 0], true_lonlat[:, 1])
    return float(np.sqrt((d ** 2).mean()))


def _tps_kernel(r2: np.ndarray) -> np.ndarray:
    out = np.zeros_like(r2)
    nz = r2 > 1e-12
    out[nz] = r2[nz] * np.log(np.sqrt(r2[nz]))
    return out


def fit_tps(src: np.ndarray, dst: np.ndarray):
    n = src.shape[0]
    d2 = ((src[:, None, :] - src[None, :, :]) ** 2).sum(-1)
    K = _tps_kernel(d2)
    P = np.hstack([np.ones((n, 1)), src])
    A = np.zeros((n + 3, n + 3))
    A[:n, :n] = K
    A[:n, n:] = P
    A[n:, :n] = P.T
    rhs = np.vstack([dst, np.zeros((3, 2))])
    params = np.linalg.solve(A + 1e-9 * np.eye(n + 3), rhs)
    w, a = params[:n], params[n:]

    def predict(q: np.ndarray) -> np.ndarray:
        dq2 = ((q[:, None, :] - src[None, :, :]) ** 2).sum(-1)
        return _tps_kernel(dq2) @ w + np.hstack([np.ones((q.shape[0], 1)), q]) @ a

    return predict


def fit_affine(src: np.ndarray, dst: np.ndarray):
    P = np.hstack([np.ones((src.shape[0], 1)), src])
    coef, *_ = np.linalg.lstsq(P, dst, rcond=None)
    return lambda q: np.hstack([np.ones((q.shape[0], 1)), q]) @ coef


def main() -> int:
    # Real geographic grid of GCPs over a mid/high-latitude region (Europe band,
    # where the Web Mercator curvature is pronounced). 12x12 regular grid.
    lats = np.linspace(48.0, 62.0, 12)
    lons = np.linspace(-10.0, 30.0, 12)
    LON, LAT = np.meshgrid(lons, lats)
    lon_f, lat_f = LON.ravel(), LAT.ravel()
    dst = np.stack([lon_f, lat_f], axis=-1)          # true WGS84 (lon, lat)
    src = web_mercator_pseudo(lat_f, lon_f)          # distorted Web Mercator map space

    # Deterministic ~70/30 train/test split (every 3rd grid node held out).
    idx = np.arange(dst.shape[0])
    test_mask = (idx % 3 == 0)
    tr, te = ~test_mask, test_mask
    src_tr, dst_tr, src_te, dst_te = src[tr], dst[tr], src[te], dst[te]

    raw = ground_rmse(src_te, dst_te)                # uncorrected
    aff = fit_affine(src_tr, dst_tr)
    tps = fit_tps(src_tr, dst_tr)
    rmse_aff = ground_rmse(aff(src_te), dst_te)
    rmse_tps = ground_rmse(tps(src_te), dst_te)

    print(f"GCPs: {dst.shape[0]} real grid points (48-62N, -10-30E) | "
          f"train {tr.sum()} / held-out test {te.sum()}")
    print(f"  raw    (no fit)    held-out ground RMSE: {raw/1000:8.1f} km")
    print(f"  affine (1st-order) held-out ground RMSE: {rmse_aff/1000:8.2f} km")
    print(f"  tps    (spline)    held-out ground RMSE: {rmse_tps:8.1f} m")
    print(f"  TPS reduces held-out RMSE by {100*(1-rmse_tps/raw):.2f}% vs raw "
          f"and {100*(1-rmse_tps/rmse_aff):.2f}% vs the affine baseline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
