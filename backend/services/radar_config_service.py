"""
backend/services/radar_config_service.py
-----------------------------------------
Validates and parses the user-provided radar_type_config.json.

The expected schema (produced by optimize_placement.py wizard) is:
{
  "<type_name>": {
    "count":          int,     # number of radars of this type to place
    "range_m":        float,   # max radar range in metres
    "h_ant":          float,   # antenna height above ground in metres
    "theta_min":      float,   # min elevation angle (deg), typically -5
    "theta_max":      float,   # max elevation angle (deg), typically 10-20
    "az_halfwidth":   float,   # azimuth half-width (deg), e.g. 60
    "energy_budget":  float,   # max ON-slots per 24h (scheduler constraint)
    "cooling_L":      int,     # rolling window length L
    "cooling_C":      int,     # max ON within rolling window C
    "frequency_bands": [str]   # list of band names, e.g. ["VHF","UHF"]
  },
  ...
}
"""

import json
from typing import Any, Dict, List, Tuple

REQUIRED_KEYS = {
    "count", "range_m", "h_ant", "theta_min", "theta_max",
    "az_halfwidth", "energy_budget", "cooling_L", "cooling_C",
    "frequency_bands",
}

SUPPORTED_FREQUENCY_BANDS = {
    "VHF", "UHF", "L_BAND", "S_BAND", "C_BAND", "X_BAND", "DEFAULT"
}


def validate_radar_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate a radar_type_config dict.
    Returns (is_valid, list_of_error_messages).
    """
    errors: List[str] = []

    if not isinstance(config, dict):
        return False, ["Configuration must be a JSON object keyed by radar type name."]

    if len(config) == 0:
        return False, ["Configuration must define at least one radar type."]

    total_radars = 0

    for type_name, cfg in config.items():
        prefix = f"Type '{type_name}'"

        if not isinstance(cfg, dict):
            errors.append(f"{prefix}: value must be a JSON object.")
            continue

        missing = REQUIRED_KEYS - set(cfg.keys())
        if missing:
            errors.append(f"{prefix}: missing required fields: {sorted(missing)}")
            continue

        # count
        if not isinstance(cfg["count"], int) or cfg["count"] < 1:
            errors.append(f"{prefix}: 'count' must be a positive integer, got {cfg['count']!r}")
        else:
            total_radars += cfg["count"]

        # range_m
        if not isinstance(cfg["range_m"], (int, float)) or cfg["range_m"] <= 0:
            errors.append(f"{prefix}: 'range_m' must be a positive number.")
        elif cfg["range_m"] > 2_000_000:
            errors.append(f"{prefix}: 'range_m' = {cfg['range_m']} m is unrealistically large (>2000 km).")

        # h_ant
        if not isinstance(cfg["h_ant"], (int, float)) or cfg["h_ant"] < 0:
            errors.append(f"{prefix}: 'h_ant' must be >= 0.")

        # theta_min / theta_max
        theta_min = cfg.get("theta_min", 0)
        theta_max = cfg.get("theta_max", 0)
        if not isinstance(theta_min, (int, float)):
            errors.append(f"{prefix}: 'theta_min' must be a number.")
        if not isinstance(theta_max, (int, float)):
            errors.append(f"{prefix}: 'theta_max' must be a number.")
        if isinstance(theta_min, (int, float)) and isinstance(theta_max, (int, float)):
            if theta_min >= theta_max:
                errors.append(f"{prefix}: 'theta_min' ({theta_min}) must be < 'theta_max' ({theta_max}).")
            if not (-90 <= theta_min <= 90 and -90 <= theta_max <= 90):
                errors.append(f"{prefix}: elevation angles must be in [-90, 90] degrees.")

        # az_halfwidth
        az = cfg.get("az_halfwidth", 0)
        if not isinstance(az, (int, float)) or not (0 < az <= 180):
            errors.append(f"{prefix}: 'az_halfwidth' must be in (0, 180] degrees, got {az!r}")

        # energy_budget
        budget = cfg.get("energy_budget", 0)
        if not isinstance(budget, (int, float)) or budget <= 0:
            errors.append(f"{prefix}: 'energy_budget' must be a positive number.")
        elif budget > 24:
            errors.append(f"{prefix}: 'energy_budget' = {budget} exceeds 24 (max time slots).")

        # cooling_L, cooling_C
        L = cfg.get("cooling_L", 0)
        C = cfg.get("cooling_C", 0)
        if not isinstance(L, int) or L < 1:
            errors.append(f"{prefix}: 'cooling_L' must be a positive integer.")
        if not isinstance(C, int) or C < 0:
            errors.append(f"{prefix}: 'cooling_C' must be a non-negative integer.")

        # frequency_bands
        bands = cfg.get("frequency_bands", [])
        if not isinstance(bands, list) or len(bands) == 0:
            errors.append(f"{prefix}: 'frequency_bands' must be a non-empty list.")
        else:
            for b in bands:
                if not isinstance(b, str) or not b.strip():
                    errors.append(f"{prefix}: frequency band names must be non-empty strings.")
            if len(bands) != len(set(bands)):
                errors.append(f"{prefix}: 'frequency_bands' contains duplicates.")

    if total_radars == 0 and not errors:
        errors.append("Total radar count across all types must be >= 1.")

    return len(errors) == 0, errors


def get_config_summary(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return a summary dict for the validated config."""
    total_radars = sum(v["count"] for v in config.values())
    all_bands = set()
    for v in config.values():
        all_bands.update(v.get("frequency_bands", []))
    return {
        "num_types": len(config),
        "total_radars": total_radars,
        "types": list(config.keys()),
        "frequency_bands": sorted(all_bands),
        "per_type": {
            k: {
                "count": v["count"],
                "range_km": round(v["range_m"] / 1000, 1),
                "energy_budget": v["energy_budget"],
                "frequency_bands": v.get("frequency_bands", []),
            }
            for k, v in config.items()
        },
    }
