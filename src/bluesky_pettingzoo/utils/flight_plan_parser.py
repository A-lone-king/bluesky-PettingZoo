"""Flight plan data parser — supports CSV and JSON formats.

Parses flight plan files into structured data for the FlightPlanScenario.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class WaypointData:
    """Single waypoint in a flight plan."""

    name: str
    lat: float
    lon: float
    alt: float  # feet
    speed: float | None = None  # knots, None = no constraint


@dataclass(frozen=True)
class FlightPlanData:
    """Complete flight plan for one aircraft."""

    flight_id: str
    aircraft_type: str
    origin: str
    destination: str
    entry_time: float  # seconds from episode start
    cruise_alt: float  # feet
    cruise_speed: float  # knots
    waypoints: list[WaypointData] = field(default_factory=list)


class FlightPlanParser:
    """Parse flight plan files (CSV/JSON) into FlightPlanData objects.

    CSV format columns:
        flight_id, aircraft_type, origin, destination, entry_time,
        cruise_alt, cruise_speed, waypoints

    The 'waypoints' column should be a JSON array string:
        '[{"name":"WP1","lat":52.0,"lon":4.0,"alt":10000}]'

    JSON format:
        [
            {
                "flight_id": "FL001",
                "aircraft_type": "B738",
                ...
                "waypoints": [{"name":"WP1","lat":52.0,"lon":4.0,"alt":10000}]
            }
        ]
    """

    @staticmethod
    def parse(file_path: Path) -> list[FlightPlanData]:
        """Parse a flight plan file.

        Args:
            file_path: Path to CSV or JSON file.

        Returns:
            List of FlightPlanData objects.

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If file format is unsupported or data is invalid.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Flight plan file not found: {file_path}")

        suffix = file_path.suffix.lower()
        if suffix == ".csv":
            return FlightPlanParser._parse_csv(file_path)
        elif suffix == ".json":
            return FlightPlanParser._parse_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}. Use .csv or .json")

    @staticmethod
    def _parse_csv(file_path: Path) -> list[FlightPlanData]:
        """Parse CSV flight plan file."""
        plans: list[FlightPlanData] = []
        with open(file_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                waypoints = FlightPlanParser._parse_waypoints_str(
                    row.get("waypoints", "[]")
                )
                plans.append(
                    FlightPlanData(
                        flight_id=row["flight_id"],
                        aircraft_type=row.get("aircraft_type", "B738"),
                        origin=row.get("origin", ""),
                        destination=row.get("destination", ""),
                        entry_time=float(row.get("entry_time", 0)),
                        cruise_alt=float(row.get("cruise_alt", 35000)),
                        cruise_speed=float(row.get("cruise_speed", 450)),
                        waypoints=waypoints,
                    )
                )
        return plans

    @staticmethod
    def _parse_json(file_path: Path) -> list[FlightPlanData]:
        """Parse JSON flight plan file."""
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("JSON flight plan must be a list of flight objects")

        plans: list[FlightPlanData] = []
        for item in data:
            waypoints = [
                WaypointData(
                    name=wp["name"],
                    lat=wp["lat"],
                    lon=wp["lon"],
                    alt=wp.get("alt", 35000),
                    speed=wp.get("speed"),
                )
                for wp in item.get("waypoints", [])
            ]
            plans.append(
                FlightPlanData(
                    flight_id=item["flight_id"],
                    aircraft_type=item.get("aircraft_type", "B738"),
                    origin=item.get("origin", ""),
                    destination=item.get("destination", ""),
                    entry_time=float(item.get("entry_time", 0)),
                    cruise_alt=float(item.get("cruise_alt", 35000)),
                    cruise_speed=float(item.get("cruise_speed", 450)),
                    waypoints=waypoints,
                )
            )
        return plans

    @staticmethod
    def _parse_waypoints_str(waypoints_str: str) -> list[WaypointData]:
        """Parse waypoints from a JSON string in CSV column."""
        try:
            data = json.loads(waypoints_str)
        except json.JSONDecodeError:
            return []

        if not isinstance(data, list):
            return []

        return [
            WaypointData(
                name=wp.get("name", f"WP{i}"),
                lat=wp["lat"],
                lon=wp["lon"],
                alt=wp.get("alt", 35000),
                speed=wp.get("speed"),
            )
            for i, wp in enumerate(data)
        ]

    @staticmethod
    def validate(plans: list[FlightPlanData]) -> list[str]:
        """Validate flight plan data.

        Args:
            plans: List of flight plans to validate.

        Returns:
            List of error messages (empty if valid).
        """
        errors: list[str] = []
        seen_ids: set[str] = set()

        for plan in plans:
            # Check duplicate flight IDs
            if plan.flight_id in seen_ids:
                errors.append(f"Duplicate flight_id: {plan.flight_id}")
            seen_ids.add(plan.flight_id)

            # Check required fields
            if not plan.flight_id:
                errors.append("Empty flight_id")
            if not plan.origin:
                errors.append(f"Flight {plan.flight_id}: empty origin")
            if not plan.destination:
                errors.append(f"Flight {plan.flight_id}: empty destination")

            # Check altitude range
            if plan.cruise_alt < 1000 or plan.cruise_alt > 60000:
                errors.append(
                    f"Flight {plan.flight_id}: cruise_alt {plan.cruise_alt} out of range [1000, 60000]"
                )

            # Check speed range
            if plan.cruise_speed < 100 or plan.cruise_speed > 600:
                errors.append(
                    f"Flight {plan.flight_id}: cruise_speed {plan.cruise_speed} out of range [100, 600]"
                )

            # Check waypoint coordinates
            for wp in plan.waypoints:
                if not (-90 <= wp.lat <= 90):
                    errors.append(
                        f"Flight {plan.flight_id}, WP {wp.name}: lat {wp.lat} out of range"
                    )
                if not (-180 <= wp.lon <= 180):
                    errors.append(
                        f"Flight {plan.flight_id}, WP {wp.name}: lon {wp.lon} out of range"
                    )

        return errors
