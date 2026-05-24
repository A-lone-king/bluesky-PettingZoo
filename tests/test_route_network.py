"""Tests for route network: Route type, segment intersection, and RouteNav scenario."""

from __future__ import annotations

import pytest

from bluesky_pettingzoo.utils.geometry import haversine_distance, segments_intersect
from bluesky_pettingzoo.utils.types import AircraftState


# ---------------------------------------------------------------------------
# Route type tests
# ---------------------------------------------------------------------------


class TestRoute:
    """Test Route dataclass."""

    def _make_route(self):
        from bluesky_pettingzoo.utils.types import Route

        return Route(
            waypoints=[
                {"lat": 40.0, "lon": -74.0},
                {"lat": 40.5, "lon": -73.5},
                {"lat": 41.0, "lon": -73.0},
            ]
        )

    def test_route_creation(self) -> None:
        route = self._make_route()
        assert len(route.waypoints) == 3

    def test_total_distance(self) -> None:
        route = self._make_route()
        d = route.total_distance_nm()
        assert d > 0
        # Two segments of ~40-50nm each
        assert 60 < d < 140

    def test_segment_count(self) -> None:
        route = self._make_route()
        assert route.segment_count() == 2

    def test_get_segment(self) -> None:
        route = self._make_route()
        p1, p2 = route.get_segment(0)
        assert p1 == (40.0, -74.0)
        assert p2 == (40.5, -73.5)

    def test_get_segment_out_of_range(self) -> None:
        route = self._make_route()
        with pytest.raises(IndexError):
            route.get_segment(5)

    def test_single_waypoint_distance_zero(self) -> None:
        from bluesky_pettingzoo.utils.types import Route

        route = Route(waypoints=[{"lat": 40.0, "lon": -74.0}])
        assert route.total_distance_nm() == 0.0
        assert route.segment_count() == 0


# ---------------------------------------------------------------------------
# Segment intersection tests
# ---------------------------------------------------------------------------


class TestSegmentsIntersect:
    """Test 2D line segment intersection."""

    def test_crossing_segments(self) -> None:
        """Two segments forming an X should intersect."""
        assert segments_intersect((0, 0), (2, 2), (0, 2), (2, 0)) is True

    def test_parallel_segments(self) -> None:
        """Parallel segments should not intersect."""
        assert segments_intersect((0, 0), (1, 0), (0, 1), (1, 1)) is False

    def test_collinear_non_overlapping(self) -> None:
        """Collinear but non-overlapping segments should not intersect."""
        assert segments_intersect((0, 0), (1, 0), (2, 0), (3, 0)) is False

    def test_t_intersection(self) -> None:
        """T-shaped intersection should intersect."""
        assert segments_intersect((0, 1), (2, 1), (1, 0), (1, 2)) is True

    def test_shared_endpoint(self) -> None:
        """Segments sharing an endpoint should intersect."""
        assert segments_intersect((0, 0), (1, 1), (1, 1), (2, 0)) is True

    def test_no_intersection(self) -> None:
        """Non-intersecting segments should return False."""
        assert segments_intersect((0, 0), (1, 0), (2, 2), (3, 2)) is False


# ---------------------------------------------------------------------------
# RouteNav scenario tests
# ---------------------------------------------------------------------------


class TestRouteNavScenario:
    """Test the RouteNav scenario."""

    def _make_scenario(self, num_aircraft: int = 3):
        from bluesky_pettingzoo.envs.scenarios.route_nav import RouteNavScenario

        return RouteNavScenario(num_aircraft=num_aircraft, seed=42)

    def _setup_scenario(self, num_aircraft: int = 3):
        """Create and setup a scenario, returning (scenario, agents)."""
        import numpy as np
        scenario = self._make_scenario(num_aircraft)
        rng = np.random.RandomState(42)
        bounds = {"lat_min": 39.0, "lat_max": 42.0, "lon_min": -75.0, "lon_max": -72.0}
        agents = scenario.setup(rng, bounds)
        return scenario, agents

    def test_scenario_creation(self) -> None:
        scenario = self._make_scenario()
        assert scenario._num_aircraft == 3

    def test_scenario_has_routes(self) -> None:
        scenario, agents = self._setup_scenario()
        routes = scenario.get_routes()
        assert len(routes) == 3

    def test_get_route_for_agent(self) -> None:
        scenario, agents = self._setup_scenario()
        for agent in agents:
            route = scenario.get_route(agent)
            assert route is not None
            assert route.segment_count() >= 1

    def test_scenario_setup(self) -> None:
        scenario, agents = self._setup_scenario()
        assert len(agents) == 3
        assert all(isinstance(a, str) for a in agents)

    def test_scenario_goal(self) -> None:
        scenario, agents = self._setup_scenario()
        for agent in agents:
            goal = scenario.get_goal(agent)
            assert "lat" in goal
            assert "lon" in goal

    def test_route_crossings_exist(self) -> None:
        """At least some routes should cross each other."""
        scenario, _ = self._setup_scenario(num_aircraft=4)
        routes = scenario.get_routes()
        route_list = list(routes.values())
        crossings = 0
        for i in range(len(route_list)):
            for j in range(i + 1, len(route_list)):
                r1, r2 = route_list[i], route_list[j]
                for s1 in range(r1.segment_count()):
                    p1, p2 = r1.get_segment(s1)
                    for s2 in range(r2.segment_count()):
                        p3, p4 = r2.get_segment(s2)
                        if segments_intersect(p1, p2, p3, p4):
                            crossings += 1
        # At least one crossing in a 4-route network
        assert crossings >= 1


# ---------------------------------------------------------------------------
# Route conflict penalty tests
# ---------------------------------------------------------------------------


class TestRouteConflictPenalty:
    """Test route-based conflict detection in ConflictPenalty."""

    def test_route_conflict_when_aircraft_near_crossing(self) -> None:
        """Two aircraft near a route crossing point should get conflict penalty."""
        from bluesky_pettingzoo.rewards.components.conflict import ConflictPenalty
        from bluesky_pettingzoo.utils.types import DiscreteAction

        config = {
            "components": {
                "conflict": {
                    "nmac_penalty": -100,
                    "warning_penalty": -10,
                    "separation_penalty": -5,
                    "thresholds": {
                        "nmac_horizontal_nm": 5,
                        "nmac_vertical_ft": 1000,
                        "warning_horizontal_nm": 10,
                        "warning_vertical_ft": 2000,
                    },
                }
            }
        }
        comp = ConflictPenalty(config)

        # Two aircraft within NMAC distance
        state_a = AircraftState(
            id="A", lat=40.0, lon=-74.0, alt=35000,
            hdg=90, tas=450, vs=0,
        )
        state_b = AircraftState(
            id="B", lat=40.005, lon=-74.005, alt=35000,
            hdg=270, tas=450, vs=0,
        )
        action = DiscreteAction(heading_idx=2, altitude_idx=2, speed_idx=2)
        all_states = {"A": state_a, "B": state_b}

        result = comp.compute("A", state_a, action, state_a, all_states)
        assert result < 0  # Should get penalty

    def test_route_conflict_penalty_with_route_data(self) -> None:
        """ConflictPenalty should accept optional route data for route-aware detection."""
        from bluesky_pettingzoo.rewards.components.conflict import ConflictPenalty

        config = {
            "components": {
                "conflict": {
                    "nmac_penalty": -100,
                    "warning_penalty": -10,
                    "separation_penalty": -5,
                    "thresholds": {
                        "nmac_horizontal_nm": 5,
                        "nmac_vertical_ft": 1000,
                        "warning_horizontal_nm": 10,
                        "warning_vertical_ft": 2000,
                    },
                }
            }
        }
        comp = ConflictPenalty(config)
        # Should have set_routes method
        assert hasattr(comp, "set_routes")
