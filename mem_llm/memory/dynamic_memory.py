"""
Short-Term Memory (STM) implementation for the DPeM mechanism.

STM stores knowledge in a key-value format where:
- Keys are knowledge type labels (e.g., "common-sense", "user-specific")
- Values are knowledge items (k_i)

Retrieval is done via closest-match (Levenshtein distance) as described
in Section 2.4.3 of the paper: R_c finds the knowledge stored in STM
that is closest to the query in terms of Levenshtein distance.

STM is refreshed periodically after certain rounds.
"""

import logging
from collections import OrderedDict

try:
    import Levenshtein as lev
except ImportError:
    # Fallback: simple Levenshtein distance implementation
    class _LevenshteinFallback:
        @staticmethod
        def distance(s1, s2):
            if len(s1) < len(s2):
                return _LevenshteinFallback.distance(s2, s1)
            if len(s2) == 0:
                return len(s1)
            prev_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                curr_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = prev_row[j + 1] + 1
                    deletions = curr_row[j] + 1
                    substitutions = prev_row[j] + (c1 != c2)
                    curr_row.append(min(insertions, deletions, substitutions))
                prev_row = curr_row
            return prev_row[-1]
    lev = _LevenshteinFallback()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Short_Term_Memory:
    """Short-Term Memory with fuzzy key matching via Levenshtein distance.

    Implements the STM component of DPeM (Section 2.3.1):
    - M_STM = {..., k_type : k_j, ...}
    - Retrieved in the order in which it is stored (closest-match)
    - Refreshed periodically after certain rounds
    - Limited storage capacity

    Args:
        key_process_func (callable, optional): A string -> string function
            used to preprocess keys before storage/query. Default: None.
        min_sim_threshold (int): Maximum Levenshtein distance for a match
            to be considered valid. Default: 6.
        max_capacity (int): Maximum number of entries in STM. Default: 100.
    """

    def __init__(self, key_process_func: callable = None,
                 min_sim_threshold: int = 6, max_capacity: int = 100):
        self.memory = OrderedDict()
        self.min_sim_threshold = min_sim_threshold
        self.key_process_func = key_process_func
        self.max_capacity = max_capacity
        # Flag table (ft) to track frequency of appearance for each k_i
        # Used by Executive Process to determine STM -> LTM transit
        self.frequency_table = {}

    def parse_key(self, key: str) -> str:
        """Preprocess a key using the key_process_func if available."""
        if self.key_process_func:
            return self.key_process_func(key)
        return key

    def __setitem__(self, key: str, value):
        """Store a knowledge item in STM.

        Args:
            key: The knowledge key (will be preprocessed).
            value: The knowledge value (k_i).
        """
        key = self.parse_key(key)
        self.memory[key] = value
        # Update frequency table
        self.frequency_table[key] = self.frequency_table.get(key, 0) + 1
        # Enforce capacity limit (remove oldest if exceeded)
        if len(self.memory) > self.max_capacity:
            oldest_key = next(iter(self.memory))
            del self.memory[oldest_key]
            if oldest_key in self.frequency_table:
                del self.frequency_table[oldest_key]

    def get_closest(self, key: str, return_score: bool = True):
        """Retrieve the closest matching knowledge item.

        Uses Levenshtein distance to find the closest key in STM (R_c retriever).

        Args:
            key: The query string.
            return_score: Whether to return the distance score.

        Returns:
            Tuple of (closest_key, value, score) or (None, None, None) if no match.
        """
        if len(self) == 0:
            return None, None, None
        key = self.parse_key(key)
        closest_key, closest_key_score = self._find_closest_key(key)
        if closest_key:
            if return_score:
                return closest_key, self.memory[closest_key], closest_key_score
            else:
                return closest_key, self.memory[closest_key], None
        return None, None, None

    def _find_closest_key(self, word: str):
        """Find the key in memory closest to the query word (Levenshtein distance).

        Args:
            word: The query string.

        Returns:
            Tuple of (closest_key, min_distance) or (None, min_distance).
        """
        min_dist = self.min_sim_threshold
        logger.debug(f"Finding closest key for word: {word}")
        closest_key = None
        for key in self.memory:
            dist = lev.distance(word, key)
            logger.debug(f"Distance between '{word}' and '{key}' is {dist}")
            if dist < min_dist:
                min_dist = dist
                closest_key = key
        if closest_key:
            return closest_key, min_dist
        return None, min_dist

    def get_frequency(self, key: str) -> int:
        """Get the access frequency of a key in the flag table."""
        key = self.parse_key(key)
        return self.frequency_table.get(key, 0)

    def increment_frequency(self, key: str):
        """Increment the frequency counter for a key."""
        key = self.parse_key(key)
        self.frequency_table[key] = self.frequency_table.get(key, 0) + 1

    def refresh(self):
        """Refresh STM by clearing all entries (called periodically)."""
        self.memory.clear()
        self.frequency_table.clear()
        logger.info("STM refreshed.")

    def get_all_items(self) -> list:
        """Return all items in STM as a list of (key, value) tuples."""
        return list(self.memory.items())

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, item):
        return self.memory[item]

    def __contains__(self, key):
        key = self.parse_key(key)
        return key in self.memory

    def __str__(self) -> str:
        return " || ".join([f"{k}: {v}" for (k, v) in self.memory.items()])
