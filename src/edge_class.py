from typing import Any


class Edge:
    def __init__(self, p1, p2):
        # Convert to tuples for consistent hashing and comparison
        self.p1 = tuple(p1) if hasattr(p1, '__iter__') and not isinstance(p1, str) else p1
        self.p2 = tuple(p2) if hasattr(p2, '__iter__') and not isinstance(p2, str) else p2

    def __eq__(self, other):
        if not isinstance(other, Edge):
            return False
        return self.p1 == other.p1 and self.p2 == other.p2
    
    def __hash__(self):
        return hash((self.p1, self.p2))