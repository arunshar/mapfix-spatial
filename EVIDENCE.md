# mapfix-spatial — reproduced evidence

_Generated 2026-06-29 by running the correction engine (`server.py`) and the
georeferencing-RMSE script (`scripts/evidence_gcp_rmse.py`)._

## Reproduced georeferencing RMSE on the real Web Mercator distortion (headline)

This is the honest, real-distortion result: a thin-plate spline (TPS) fit to
real ground control points against the **genuine Web Mercator projection** (no
simulated distortion, no added noise). A 12x12 real geographic grid over Europe
(48-62N, -10-30E) is the GCP set; the points are split 96 train / 48 held-out;
held-out **ground RMSE** (haversine) is reported for three registrations:

| Registration | Held-out ground RMSE |
|---|---|
| raw (read Mercator as equirectangular, no fit) | ~1,338 km |
| affine (1st-order, 6-parameter) | ~24.0 km |
| **thin-plate spline (TPS)** | **~2.5 km** |

TPS reduces held-out RMSE by **99.8% vs raw** and **89.4% vs the affine
baseline**: the affine fit cannot model the nonlinear latitude stretch of Web
Mercator, and the TPS removes it. Pure numpy, deterministic, no network/GPU.
Reproduce with:

```bash
python scripts/evidence_gcp_rmse.py
```

## Illustrative demo (synthetic profile, fallback)

The distortion-and-correction math is real and deterministic; the interactive
demo uses a synthetic parametric "Mercator repair" profile, so it demonstrates
the method on controllable distortion rather than on real misregistered maps.

On 8 sample coordinates under the Mercator-repair profile, the fixed-point inverse-warp recovers **~72% of the distortion area** (recovered = 0.72) with a **mean residual of ~2.8 px** (confidence 0.80). Reproduce with:

```python
import server
pts = [[-122.4,37.7],[-122.3,37.8],[-122.2,37.75],[-122.35,37.72],
       [-122.25,37.85],[-122.45,37.78],[-122.3,37.7],[-122.4,37.82]]
print(server.build_correction({"points": pts, "projectionKey": "mercator"})["metrics"])
# {'mean': 2.79, 'recovered': 0.72, 'skew': 5.88, 'confidence': 0.80}
```

The interactive demo (`app.html`) runs the same engine live in the browser.
