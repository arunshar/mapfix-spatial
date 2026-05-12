from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency for static hosts
    load_dotenv = None

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - handled at runtime
    OpenAI = None


ROOT = Path(__file__).resolve().parent
PROJECTION_TUNING = {
    "mercator": {"curve": 0.74, "shear": 0.42, "label": "Mercator repair"},
    "equalEarth": {"curve": 0.48, "shear": 0.28, "label": "Equal Earth"},
    "albers": {"curve": 0.62, "shear": -0.22, "label": "Albers local"},
    "tangent": {"curve": 0.28, "shear": 0.12, "label": "Local tangent"},
}


def load_environment() -> None:
    if load_dotenv is None:
        return

    load_dotenv(ROOT / ".env")
    load_dotenv(ROOT.parent / ".env")


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def bounds_for(points: list[list[float]]) -> dict[str, float]:
    lon_values = [point[0] for point in points]
    lat_values = [point[1] for point in points]
    min_lon = min(lon_values)
    max_lon = max(lon_values)
    min_lat = min(lat_values)
    max_lat = max(lat_values)
    lon_pad = max((max_lon - min_lon) * 0.28, 0.04)
    lat_pad = max((max_lat - min_lat) * 0.28, 0.04)

    return {
        "minLon": min_lon - lon_pad,
        "maxLon": max_lon + lon_pad,
        "minLat": min_lat - lat_pad,
        "maxLat": max_lat + lat_pad,
    }


def normalize_point(point: list[float], bounds: dict[str, float]) -> dict[str, float]:
    lon, lat = point
    return {
        "x": (lon - bounds["minLon"]) / (bounds["maxLon"] - bounds["minLon"]),
        "y": 1 - (lat - bounds["minLat"]) / (bounds["maxLat"] - bounds["minLat"]),
    }


def denormalize_point(point: dict[str, float], bounds: dict[str, float]) -> list[float]:
    return [
        bounds["minLon"] + point["x"] * (bounds["maxLon"] - bounds["minLon"]),
        bounds["minLat"] + (1 - point["y"]) * (bounds["maxLat"] - bounds["minLat"]),
    ]


def distortion_vector(
    point: dict[str, float], noise: float, projection: dict[str, float | str], seed: float
) -> dict[str, float]:
    centered_x = point["x"] - 0.5
    centered_y = point["y"] - 0.5
    wave_x = math.sin((point["y"] * 2.8 + seed) * math.pi)
    wave_y = math.cos((point["x"] * 2.2 - seed) * math.pi)
    radial = centered_x * centered_x + centered_y * centered_y

    return {
        "x": noise
        * (
            float(projection["shear"]) * centered_y * 0.18
            + float(projection["curve"]) * wave_x * 0.055
            + centered_x * radial * 0.28
        ),
        "y": noise
        * (
            -float(projection["shear"]) * centered_x * 0.14
            + float(projection["curve"]) * wave_y * 0.045
            - centered_y * radial * 0.22
        ),
    }


def apply_distortion(
    point: dict[str, float], noise: float, projection: dict[str, float | str], seed: float
) -> dict[str, float]:
    vector = distortion_vector(point, noise, projection, seed)
    return {
        "x": clamp(point["x"] + vector["x"], -0.1, 1.1),
        "y": clamp(point["y"] + vector["y"], -0.1, 1.1),
    }


def correct_point(
    raw_point: dict[str, float],
    noise: float,
    projection: dict[str, float | str],
    strength: float,
    seed: float,
) -> dict[str, float]:
    estimate = dict(raw_point)
    for _ in range(5):
        vector = distortion_vector(estimate, noise, projection, seed)
        estimate = {
            "x": raw_point["x"] - vector["x"] * strength,
            "y": raw_point["y"] - vector["y"] * strength,
        }

    return {"x": clamp(estimate["x"], -0.08, 1.08), "y": clamp(estimate["y"], -0.08, 1.08)}


def gradient_at(
    point: dict[str, float], noise: float, projection: dict[str, float | str], seed: float
) -> dict[str, float]:
    epsilon = 0.004
    base = distortion_vector(point, noise, projection, seed)
    dx = distortion_vector({"x": point["x"] + epsilon, "y": point["y"]}, noise, projection, seed)
    dy = distortion_vector({"x": point["x"], "y": point["y"] + epsilon}, noise, projection, seed)
    x_value = (dx["x"] - base["x"] + dy["x"] - base["x"]) / epsilon
    y_value = (dx["y"] - base["y"] + dy["y"] - base["y"]) / epsilon

    return {"x": x_value, "y": y_value, "magnitude": math.hypot(x_value, y_value)}


def metrics_for(
    normalized: list[dict[str, float]],
    raw_points: list[dict[str, float]],
    corrected_points: list[dict[str, float]],
    strength: float,
    noise: float,
) -> dict[str, float]:
    errors = [
        math.hypot(point["x"] - normalized[index]["x"], point["y"] - normalized[index]["y"]) * 1000
        for index, point in enumerate(corrected_points)
    ]
    raw_errors = [
        math.hypot(point["x"] - normalized[index]["x"], point["y"] - normalized[index]["y"]) * 1000
        for index, point in enumerate(raw_points)
    ]
    mean = sum(errors) / len(errors)
    raw_mean = sum(raw_errors) / len(raw_errors)
    recovered = 1 if raw_mean == 0 else clamp(1 - mean / raw_mean, 0, 1)
    skew = clamp(noise * 64 * (1 - strength * 0.68), 0, 42)
    confidence = clamp(0.58 + recovered * 0.35 - noise * 0.15, 0.42, 0.98)

    return {"mean": mean, "recovered": recovered, "skew": skew, "confidence": confidence}


def coordinate_rows(
    original_points: list[list[float]],
    raw_points: list[dict[str, float]],
    corrected_points: list[dict[str, float]],
    bounds: dict[str, float],
) -> list[dict[str, Any]]:
    rows = []
    for index, original in enumerate(original_points[:12]):
        target = normalize_point(original, bounds)
        raw_geo = denormalize_point(raw_points[index], bounds)
        fixed_geo = denormalize_point(corrected_points[index], bounds)
        residual = (
            math.hypot(corrected_points[index]["x"] - target["x"], corrected_points[index]["y"] - target["y"])
            * 1000
        )
        rows.append(
            {
                "id": f"P{index + 1:02d}",
                "rawLon": raw_geo[0],
                "rawLat": raw_geo[1],
                "fixedLon": fixed_geo[0],
                "fixedLat": fixed_geo[1],
                "residual": residual,
            }
        )
    return rows


def build_correction(payload: dict[str, Any]) -> dict[str, Any]:
    points = payload.get("points")
    if not isinstance(points, list) or len(points) < 3:
        raise ValueError("At least three coordinate points are required.")

    clean_points = [[float(point[0]), float(point[1])] for point in points]
    projection_key = str(payload.get("projectionKey") or "mercator")
    projection = PROJECTION_TUNING.get(projection_key, PROJECTION_TUNING["mercator"])
    strength = clamp(float(payload.get("strength", 0.72)), 0, 1)
    noise = clamp(float(payload.get("noise", 0.18)), 0, 0.42)
    seed = clamp(float(payload.get("seed", 0.38)), 0, 1)
    bounds = bounds_for(clean_points)
    normalized = [normalize_point(point, bounds) for point in clean_points]
    raw_points = [apply_distortion(point, noise, projection, seed) for point in normalized]
    corrected_points = [
        correct_point(point, noise, projection, strength, seed) for point in raw_points
    ]
    gradients = [gradient_at(point, noise, projection, seed) for point in corrected_points]

    return {
        "projection": projection,
        "projectionKey": projection_key,
        "sampleLabel": str(payload.get("sampleLabel") or "Untitled sample"),
        "strength": strength,
        "noise": noise,
        "seed": seed,
        "bounds": bounds,
        "normalized": normalized,
        "rawPoints": raw_points,
        "correctedPoints": corrected_points,
        "gradients": gradients,
        "metrics": metrics_for(normalized, raw_points, corrected_points, strength, noise),
        "coordinates": coordinate_rows(clean_points, raw_points, corrected_points, bounds),
        "pointCount": len(clean_points),
    }


def extract_response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text)

    chunks = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                chunks.append(str(text))
    return "\n".join(chunks)


def default_ai_summary(correction: dict[str, Any], status: str, detail: str | None = None) -> dict[str, Any]:
    metrics = correction["metrics"]
    return {
        "status": status,
        "diagnosis": f"{correction['sampleLabel']} processed with {correction['projection']['label']}.",
        "gradientSummary": f"Mean residual {metrics['mean']:.2f}px with {metrics['recovered']:.0%} area recovery.",
        "correctionSummary": "Server-side correction completed with the deterministic MapFix gradient model.",
        "qualityFlags": ["OpenAI key unavailable" if status == "local" else "OpenAI call fell back locally"],
        "nextStep": "Set OPENAI_API_KEY and rerun for live GPT analysis." if status == "local" else "Retry the live call after checking server logs.",
        "detail": detail,
    }


def openai_client() -> Any | None:
    if OpenAI is None or not os.getenv("OPENAI_API_KEY"):
        return None
    return OpenAI()


def run_gpt_analysis(correction: dict[str, Any]) -> dict[str, Any]:
    client = openai_client()
    if client is None:
        return default_ai_summary(correction, "local")

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "status": {"type": "string"},
            "diagnosis": {"type": "string"},
            "gradientSummary": {"type": "string"},
            "correctionSummary": {"type": "string"},
            "qualityFlags": {"type": "array", "items": {"type": "string"}},
            "nextStep": {"type": "string"},
        },
        "required": [
            "status",
            "diagnosis",
            "gradientSummary",
            "correctionSummary",
            "qualityFlags",
            "nextStep",
        ],
    }
    compact_payload = {
        "sample": correction["sampleLabel"],
        "projection": correction["projection"]["label"],
        "strength": correction["strength"],
        "noise": correction["noise"],
        "metrics": correction["metrics"],
        "coordinates": correction["coordinates"],
        "gradientSample": correction["gradients"][:8],
    }

    response = client.responses.create(
        model=os.getenv("OPENAI_TEXT_MODEL", "gpt-5.5"),
        instructions=(
            "You are a geospatial correction analyst. Inspect the MapFix correction payload, "
            "summarize the distortion gradient and repair quality, and return only JSON that "
            "matches the provided schema. Keep each string concise."
        ),
        input=json.dumps(compact_payload),
        max_output_tokens=500,
        text={
            "format": {
                "type": "json_schema",
                "name": "mapfix_analysis",
                "schema": schema,
                "strict": True,
            }
        },
    )
    raw_text = extract_response_text(response)
    analysis = json.loads(raw_text)
    analysis["status"] = "openai"
    return analysis


def image_prompt(payload: dict[str, Any]) -> str:
    metrics = payload.get("metrics") or {}
    sample = payload.get("sampleLabel") or "spatial sample"
    projection = payload.get("projectionLabel") or "corrected local projection"
    points = payload.get("coordinates") or []
    point_preview = points[:10]

    return (
        "Create a clean technical map projection render for Distortion-Aware MapFix. "
        "Show a distortion-free corrected route or point field on a subtle gridded map plane. "
        "Use teal corrected geometry, coral faint raw geometry, and small gold gradient arrows. "
        "No UI chrome, no title text, no labels, no watermark. "
        f"Dataset: {sample}. Projection: {projection}. "
        f"Mean residual: {metrics.get('mean', 0):.2f}px. Area recovery: {metrics.get('recovered', 0):.0%}. "
        f"Coordinate preview: {json.dumps(point_preview)}"
    )


def run_image_generation(payload: dict[str, Any]) -> dict[str, Any]:
    client = openai_client()
    if client is None:
        return {
            "available": False,
            "status": "local",
            "message": "Set OPENAI_API_KEY to enable Image API rendering.",
        }

    result = client.images.generate(
        model=os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1"),
        prompt=image_prompt(payload),
        size=os.getenv("OPENAI_IMAGE_SIZE", "1536x1024"),
        quality=os.getenv("OPENAI_IMAGE_QUALITY", "medium"),
        output_format="png",
        response_format="b64_json",
    )
    image = (result.data or [None])[0]
    if image is None or not image.b64_json:
        raise RuntimeError("Image API did not return image data.")

    return {
        "available": True,
        "status": "openai",
        "model": os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1"),
        "imageDataUrl": f"data:image/png;base64,{image.b64_json}",
        "revisedPrompt": image.revised_prompt,
    }


class MapFixHandler(SimpleHTTPRequestHandler):
    server_version = "MapFixSpatial/1.0"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[mapfix] {self.address_string()} - {format % args}")

    def read_json(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            return {}
        raw_body = self.rfile.read(content_length)
        return json.loads(raw_body.decode("utf-8"))

    def send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/api/health":
            self.send_json(
                {
                    "ok": True,
                    "openaiConfigured": bool(os.getenv("OPENAI_API_KEY")),
                    "sdkAvailable": OpenAI is not None,
                    "textModel": os.getenv("OPENAI_TEXT_MODEL", "gpt-5.5"),
                    "imageModel": os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1"),
                    "generatedAt": datetime.now(timezone.utc).isoformat(),
                }
            )
            return

        super().do_GET()

    def do_POST(self) -> None:
        try:
            if self.path == "/api/correct":
                correction = build_correction(self.read_json())
                try:
                    ai_summary = run_gpt_analysis(correction)
                    mode = ai_summary.get("status", "openai")
                except Exception as exc:
                    ai_summary = default_ai_summary(correction, "fallback", str(exc))
                    mode = "fallback"

                self.send_json(
                    {
                        **correction,
                        "mode": mode,
                        "ai": ai_summary,
                        "model": os.getenv("OPENAI_TEXT_MODEL", "gpt-5.5") if mode == "openai" else None,
                        "generatedAt": datetime.now(timezone.utc).isoformat(),
                    }
                )
                return

            if self.path == "/api/render-map":
                self.send_json(run_image_generation(self.read_json()))
                return

            self.send_error(HTTPStatus.NOT_FOUND, "Unknown API route.")
        except ValueError as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)


def main() -> None:
    load_environment()
    port = int(os.getenv("PORT", "4173"))
    host = os.getenv("HOST", "127.0.0.1")
    server = ThreadingHTTPServer((host, port), MapFixHandler)
    print(f"MapFix Spatial running at http://{host}:{port}")
    print("OpenAI backend:", "enabled" if os.getenv("OPENAI_API_KEY") else "disabled")
    server.serve_forever()


if __name__ == "__main__":
    main()
