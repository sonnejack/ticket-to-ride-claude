"""
Ticket to Ride USA Map - Board representation with cities and routes.
"""
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set
from enum import IntEnum


class RouteColor(IntEnum):
    """Route colors - X means any color can be used."""
    ANY = 0      # X - gray routes
    RED = 1      # R
    ORANGE = 2   # O
    YELLOW = 3   # Y
    GREEN = 4    # G
    BLUE = 5     # B
    PURPLE = 6   # P (pink)
    BLACK = 7    # K
    WHITE = 8    # W
    WILD = 9     # Locomotive/wild cards


COLOR_MAP = {
    'X': RouteColor.ANY,
    'R': RouteColor.RED,
    'O': RouteColor.ORANGE,
    'Y': RouteColor.YELLOW,
    'G': RouteColor.GREEN,
    'B': RouteColor.BLUE,
    'P': RouteColor.PURPLE,
    'K': RouteColor.BLACK,
    'W': RouteColor.WHITE,
}

COLOR_NAMES = {
    RouteColor.ANY: 'Gray',
    RouteColor.RED: 'Red',
    RouteColor.ORANGE: 'Orange',
    RouteColor.YELLOW: 'Yellow',
    RouteColor.GREEN: 'Green',
    RouteColor.BLUE: 'Blue',
    RouteColor.PURPLE: 'Purple',
    RouteColor.BLACK: 'Black',
    RouteColor.WHITE: 'White',
    RouteColor.WILD: 'Wild',
}


# Points awarded for claiming routes by length
ROUTE_POINTS = {
    1: 1,
    2: 2,
    3: 4,
    4: 7,
    5: 10,
    6: 15,
}


@dataclass
class Route:
    """A single route between two cities."""
    city1: str
    city2: str
    length: int
    color: RouteColor
    route_id: int
    claimed_by: Optional[int] = None  # Player index or None

    def __hash__(self):
        return hash(self.route_id)

    def __eq__(self, other):
        if isinstance(other, Route):
            return self.route_id == other.route_id
        return False

    def points(self) -> int:
        """Points awarded for claiming this route."""
        return ROUTE_POINTS.get(self.length, 0)

    def connects(self, city: str) -> bool:
        """Check if this route connects to a city."""
        return city == self.city1 or city == self.city2

    def other_city(self, city: str) -> str:
        """Get the other city this route connects to."""
        if city == self.city1:
            return self.city2
        elif city == self.city2:
            return self.city1
        raise ValueError(f"City {city} not in route")


class Board:
    """The Ticket to Ride USA game board."""

    # All 36 cities on the USA map
    CITIES = [
        "Atlanta", "Boston", "Calgary", "Charleston", "Chicago",
        "Dallas", "Denver", "Duluth", "El Paso", "Helena",
        "Houston", "Kansas City", "Las Vegas", "Little Rock", "Los Angeles",
        "Miami", "Montreal", "Nashville", "New Orleans", "New York",
        "Oklahoma City", "Omaha", "Phoenix", "Pittsburgh", "Portland",
        "Raleigh", "Salt Lake City", "San Francisco", "Santa Fe", "Sault St. Marie",
        "Seattle", "Saint Louis", "Toronto", "Vancouver", "Washington",
        "Winnipeg"
    ]

    # Raw route data from provided CSV
    ROUTE_DATA = [
        ("Vancouver", "Calgary", 3, "X"),
        ("Vancouver", "Seattle", 1, "X"),
        ("Vancouver", "Seattle", 1, "X"),
        ("Seattle", "Calgary", 4, "X"),
        ("Seattle", "Helena", 6, "Y"),
        ("Seattle", "Portland", 1, "X"),
        ("Seattle", "Portland", 1, "X"),
        ("Portland", "Salt Lake City", 6, "B"),
        ("Portland", "San Francisco", 5, "G"),
        ("Portland", "San Francisco", 5, "P"),
        ("San Francisco", "Salt Lake City", 5, "O"),
        ("San Francisco", "Salt Lake City", 5, "W"),
        ("San Francisco", "Los Angeles", 3, "Y"),
        ("San Francisco", "Los Angeles", 3, "P"),
        ("Los Angeles", "Las Vegas", 2, "X"),
        ("Los Angeles", "Phoenix", 3, "X"),
        ("Los Angeles", "El Paso", 6, "K"),
        ("Calgary", "Winnipeg", 6, "W"),
        ("Calgary", "Helena", 4, "X"),
        ("Helena", "Winnipeg", 4, "B"),
        ("Helena", "Salt Lake City", 3, "P"),
        ("Helena", "Denver", 4, "G"),
        ("Helena", "Duluth", 6, "O"),
        ("Helena", "Omaha", 5, "R"),
        ("Salt Lake City", "Denver", 3, "R"),
        ("Salt Lake City", "Denver", 3, "Y"),
        ("Las Vegas", "Salt Lake City", 3, "O"),
        ("Phoenix", "Denver", 5, "W"),
        ("Phoenix", "Santa Fe", 3, "X"),
        ("Phoenix", "El Paso", 3, "X"),
        ("Winnipeg", "Sault St. Marie", 6, "X"),
        ("Winnipeg", "Duluth", 4, "K"),
        ("Duluth", "Sault St. Marie", 3, "X"),
        ("Duluth", "Toronto", 6, "P"),
        ("Duluth", "Chicago", 3, "R"),
        ("Duluth", "Omaha", 2, "X"),
        ("Duluth", "Omaha", 2, "X"),
        ("Omaha", "Chicago", 4, "B"),
        ("Omaha", "Kansas City", 1, "X"),
        ("Omaha", "Kansas City", 1, "X"),
        ("Kansas City", "Saint Louis", 2, "B"),
        ("Kansas City", "Saint Louis", 2, "P"),
        ("Kansas City", "Oklahoma City", 2, "X"),
        ("Kansas City", "Oklahoma City", 2, "X"),
        ("Oklahoma City", "Little Rock", 2, "X"),
        ("Oklahoma City", "Dallas", 2, "X"),
        ("Oklahoma City", "Dallas", 2, "X"),
        ("Dallas", "Little Rock", 2, "X"),
        ("Dallas", "Houston", 1, "X"),
        ("Dallas", "Houston", 1, "X"),
        ("Houston", "New Orleans", 2, "X"),
        ("El Paso", "Houston", 6, "G"),
        ("El Paso", "Dallas", 4, "R"),
        ("El Paso", "Oklahoma City", 5, "Y"),
        ("El Paso", "Santa Fe", 2, "X"),
        ("Santa Fe", "Oklahoma City", 3, "B"),
        ("Oklahoma City", "Denver", 4, "R"),
        ("Santa Fe", "Denver", 2, "X"),
        ("Denver", "Kansas City", 4, "K"),
        ("Denver", "Kansas City", 4, "O"),
        ("Denver", "Omaha", 4, "P"),
        ("New Orleans", "Miami", 6, "R"),
        ("New Orleans", "Atlanta", 4, "O"),
        ("New Orleans", "Atlanta", 4, "Y"),
        ("New Orleans", "Little Rock", 3, "G"),
        ("Little Rock", "Nashville", 3, "W"),
        ("Little Rock", "Saint Louis", 2, "X"),
        ("Saint Louis", "Nashville", 2, "X"),
        ("Saint Louis", "Pittsburgh", 5, "G"),
        ("Saint Louis", "Chicago", 2, "G"),
        ("Saint Louis", "Chicago", 2, "W"),
        ("Chicago", "Pittsburgh", 3, "K"),
        ("Chicago", "Pittsburgh", 3, "O"),
        ("Chicago", "Toronto", 4, "W"),
        ("Sault St. Marie", "Montreal", 5, "K"),
        ("Toronto", "Montreal", 3, "X"),
        ("Sault St. Marie", "Toronto", 2, "X"),
        ("Toronto", "Pittsburgh", 2, "X"),
        ("Pittsburgh", "New York", 2, "W"),
        ("Pittsburgh", "New York", 2, "G"),
        ("Pittsburgh", "Washington", 2, "X"),
        ("Pittsburgh", "Raleigh", 2, "X"),
        ("Nashville", "Raleigh", 3, "K"),
        ("Nashville", "Atlanta", 1, "X"),
        ("Nashville", "Pittsburgh", 4, "Y"),
        ("Atlanta", "Miami", 5, "B"),
        ("Atlanta", "Charleston", 2, "X"),
        ("Atlanta", "Raleigh", 2, "X"),
        ("Atlanta", "Raleigh", 2, "X"),
        ("Charleston", "Miami", 4, "P"),
        ("Raleigh", "Charleston", 2, "X"),
        ("Raleigh", "Washington", 2, "X"),
        ("Raleigh", "Washington", 2, "X"),
        ("Washington", "New York", 2, "O"),
        ("Washington", "New York", 2, "K"),
        ("New York", "Boston", 2, "Y"),
        ("New York", "Boston", 2, "R"),
        ("New York", "Montreal", 3, "B"),
        ("Boston", "Montreal", 2, "X"),
        ("Boston", "Montreal", 2, "X"),
    ]

    def __init__(self):
        """Initialize the board with all routes."""
        self.city_to_idx: Dict[str, int] = {city: i for i, city in enumerate(self.CITIES)}
        self.idx_to_city: Dict[int, str] = {i: city for i, city in enumerate(self.CITIES)}

        # Create all routes
        self.routes: List[Route] = []
        for i, (city1, city2, length, color_code) in enumerate(self.ROUTE_DATA):
            route = Route(
                city1=city1,
                city2=city2,
                length=length,
                color=COLOR_MAP[color_code],
                route_id=i
            )
            self.routes.append(route)

        # Build adjacency data structures
        self._build_adjacency()

    def _build_adjacency(self):
        """Build adjacency lists and route lookup structures."""
        # City -> list of routes connecting to it
        self.city_routes: Dict[str, List[Route]] = {city: [] for city in self.CITIES}

        # (city1, city2) -> list of routes between them (can be multiple/double routes)
        self.city_pair_routes: Dict[Tuple[str, str], List[Route]] = {}

        for route in self.routes:
            self.city_routes[route.city1].append(route)
            self.city_routes[route.city2].append(route)

            # Normalize city pair key (alphabetical order)
            key = tuple(sorted([route.city1, route.city2]))
            if key not in self.city_pair_routes:
                self.city_pair_routes[key] = []
            self.city_pair_routes[key].append(route)

    def get_routes_between(self, city1: str, city2: str) -> List[Route]:
        """Get all routes between two cities."""
        key = tuple(sorted([city1, city2]))
        return self.city_pair_routes.get(key, [])

    def get_available_routes(self, num_players: int = 4) -> List[Route]:
        """Get all unclaimed routes, respecting double route rules."""
        available = []
        for route in self.routes:
            if route.claimed_by is None:
                # For 2-3 players, only one route of a double can be claimed
                if num_players <= 3:
                    key = tuple(sorted([route.city1, route.city2]))
                    parallel_routes = self.city_pair_routes[key]
                    if len(parallel_routes) > 1:
                        # Check if any parallel route is claimed
                        if any(r.claimed_by is not None for r in parallel_routes):
                            continue
                available.append(route)
        return available

    def get_available_routes_for_player(self, player_idx: int, num_players: int = 4) -> List[Route]:
        """Get routes a specific player can claim (respecting double route rules)."""
        available = []
        for route in self.routes:
            if route.claimed_by is None:
                key = tuple(sorted([route.city1, route.city2]))
                parallel_routes = self.city_pair_routes[key]

                # In 2-3 player games, only one of a double route can be claimed
                if num_players <= 3 and len(parallel_routes) > 1:
                    if any(r.claimed_by is not None for r in parallel_routes):
                        continue

                # In 4+ player games, same player can't claim both routes
                if len(parallel_routes) > 1:
                    if any(r.claimed_by == player_idx for r in parallel_routes):
                        continue

                available.append(route)
        return available

    def claim_route(self, route_id: int, player_idx: int) -> int:
        """Claim a route for a player. Returns points earned."""
        route = self.routes[route_id]
        if route.claimed_by is not None:
            raise ValueError(f"Route {route_id} already claimed")
        route.claimed_by = player_idx
        return route.points()

    def reset(self):
        """Reset all routes to unclaimed."""
        for route in self.routes:
            route.claimed_by = None

    def get_player_routes(self, player_idx: int) -> List[Route]:
        """Get all routes claimed by a player."""
        return [r for r in self.routes if r.claimed_by == player_idx]

    def get_connected_cities(self, player_idx: int) -> Dict[str, Set[str]]:
        """Get connectivity graph for a player's claimed routes."""
        connectivity: Dict[str, Set[str]] = {city: set() for city in self.CITIES}
        for route in self.get_player_routes(player_idx):
            connectivity[route.city1].add(route.city2)
            connectivity[route.city2].add(route.city1)
        return connectivity

    def cities_connected(self, player_idx: int, city1: str, city2: str) -> bool:
        """Check if two cities are connected for a player using BFS."""
        if city1 == city2:
            return True

        connectivity = self.get_connected_cities(player_idx)
        visited = {city1}
        queue = [city1]

        while queue:
            current = queue.pop(0)
            for neighbor in connectivity[current]:
                if neighbor == city2:
                    return True
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return False

    def longest_continuous_path(self, player_idx: int) -> int:
        """Calculate longest continuous path for a player using DFS."""
        player_routes = self.get_player_routes(player_idx)
        if not player_routes:
            return 0

        # Build adjacency with route lengths
        adj: Dict[str, List[Tuple[str, int, int]]] = {city: [] for city in self.CITIES}
        for route in player_routes:
            adj[route.city1].append((route.city2, route.length, route.route_id))
            adj[route.city2].append((route.city1, route.length, route.route_id))

        max_length = 0

        def dfs(city: str, used_routes: Set[int], current_length: int):
            nonlocal max_length
            max_length = max(max_length, current_length)

            for neighbor, length, route_id in adj[city]:
                if route_id not in used_routes:
                    used_routes.add(route_id)
                    dfs(neighbor, used_routes, current_length + length)
                    used_routes.remove(route_id)

        # Start DFS from each city
        for city in self.CITIES:
            if adj[city]:  # Only start from cities with connections
                dfs(city, set(), 0)

        return max_length

    def to_numpy(self) -> np.ndarray:
        """Convert route ownership to numpy array for state representation."""
        # Shape: (num_routes,) - value is player_idx + 1, or 0 if unclaimed
        state = np.zeros(len(self.routes), dtype=np.int8)
        for i, route in enumerate(self.routes):
            if route.claimed_by is not None:
                state[i] = route.claimed_by + 1
        return state

    def __repr__(self):
        claimed = sum(1 for r in self.routes if r.claimed_by is not None)
        return f"Board({len(self.routes)} routes, {claimed} claimed)"
