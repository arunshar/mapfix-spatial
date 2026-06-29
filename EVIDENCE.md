# mapfix-spatial — reproduced evidence

_Generated 2026-06-29T17:14:20Z by running the deterministic correction engine (`server.py`) on sample points._

The distortion-and-correction math is real and deterministic. The distortion itself is a synthetic parametric model (the "Mercator repair" profile), so this demonstrates the method on controllable distortion, not on real misregistered maps.

## Reproduced demo (headline number)

On 8 sample coordinates under the Mercator-repair profile, the fixed-point inverse-warp recovers **~72% of the distortion area** (recovered = 0.72) with a **mean residual of ~2.8 px** (confidence 0.80). Reproduce with:

```python
import server
pts = [[-122.4,37.7],[-122.3,37.8],[-122.2,37.75],[-122.35,37.72],
       [-122.25,37.85],[-122.45,37.78],[-122.3,37.7],[-122.4,37.82]]
print(server.build_correction({"points": pts, "projectionKey": "mercator"})["metrics"])
# {'mean': 2.79, 'recovered': 0.72, 'skew': 5.88, 'confidence': 0.80}
```

The interactive demo (`app.html`) runs the same engine live in the browser.
