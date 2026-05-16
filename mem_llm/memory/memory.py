"""
Unified Memory (M) for the DPeM mechanism.

Implements the full memory structure as described in Section 2.3.1:
    M = [M_working, M_STM, M_LTM]

The Memory class coordinates:
- Working Memory: buffer for newly detected information (refreshed each iteration)
- Short-Term Memory (STM): relevant and recent knowledge (refreshed periodically)
- Long-Term Memory (LTM): frequently accessed knowledge (never refreshed)

The transit mechanism (Executive Process, Section 2.3.3) moves knowledge
from STM to LTM when the frequency of access reaches a threshold θ.
"""

import logging
from .dynamic_memory import Short_Term_Memory
from .static_memory import Long_Term_Memory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WorkingMemory:
    """Working Memory: buffer for storing newly detected information.

    Properties (from Table 1 in the paper):
    - Refreshed: Each Iteration
    - Storage: Limited
    - Supports Lookup: No

    Working memory stores notes (nt_s = C(d_i)) from the coordinator
    during the learning step of the Rehearsal Process.
    """

    def __init__(self, max_capacity: int = 50):
        self.notes = []
        self.max_capacity = max_capacity

    def add_note(self, note: str):
        """Add a note from the coordinator's learning step."""
        self.notes.append(note)
        if len(self.notes) > self.max_capacity:
            self.notes.pop(0)

    def get_notes(self) -> list:
        """Return all current notes."""
        return self.notes

    def refresh(self):
        """Clear working memory (called after each iteration)."""
        self.notes = []

    def __len__(self):
        return len(self.notes)

    def __str__(self):
        return " | ".join(self.notes)


class Memory:
    """Unified DPeM Memory coordinating Working Memory, STM, and LTM.

    Implements the full memory formation (Equation 1):
        M_working = {nt0, ..., nt_i, ...}
        M_STM = {..., k_type : k_j, ...}
        M_LTM = {..., k_type : k_f, ...}
        M = [M_working, M_STM, M_LTM]

    Args:
        transit_threshold (int): Frequency threshold θ for STM -> LTM transit.
            Default: 3.
        stm_key_process_func (callable, optional): Key preprocessing for STM.
        ltm_model_name (str): Sentence transformer model for LTM embeddings.
        ltm_checkpoint_path (str, optional): Path to trained LTM encoder.
        stm_capacity (int): Maximum entries in STM. Default: 100.
        working_capacity (int): Maximum notes in working memory. Default: 50.
    """

    def __init__(self, transit_threshold: int = 3,
                 stm_key_process_func: callable = None,
                 ltm_model_name: str = "all-MiniLM-L6-v2",
                 ltm_checkpoint_path: str = None,
                 stm_capacity: int = 100,
                 working_capacity: int = 50):
        self.working_memory = WorkingMemory(max_capacity=working_capacity)
        self.stm = Short_Term_Memory(
            key_process_func=stm_key_process_func,
            max_capacity=stm_capacity,
        )
        self.ltm = Long_Term_Memory(
            model_name=ltm_model_name,
            checkpoint_path=ltm_checkpoint_path,
        )
        self.transit_threshold = transit_threshold

    def add_to_working_memory(self, note: str):
        """Add a note to working memory (Learning step of Rehearsal Process).

        Args:
            note: The note extracted by coordinator C from dialogue d_i.
        """
        self.working_memory.add_note(note)

    def add_to_stm(self, key: str, value: str):
        """Add a knowledge item to STM (Summarizing step -> Executive Process).

        The key represents the knowledge type and the value is the knowledge item k_i.

        Args:
            key: Knowledge type label (e.g., "common-sense: fever treatment").
            value: The knowledge content.
        """
        self.stm[key] = value

    def add_to_ltm(self, key: str, value: str):
        """Directly add a knowledge item to LTM.

        Args:
            key: Knowledge key.
            value: Knowledge content.
        """
        self.ltm[key] = value

    def transit(self):
        """Execute the STM -> LTM transit (Executive Process, Section 2.3.3).

        When the frequency of a knowledge item k_i in the flag table (ft)
        reaches the predetermined threshold θ, k_i is transferred to LTM.

        Returns:
            list: Keys that were transited from STM to LTM.
        """
        transited_keys = []
        keys_to_transit = []

        # Check frequency table for items exceeding threshold
        for key, freq in self.stm.frequency_table.items():
            if freq >= self.transit_threshold:
                keys_to_transit.append(key)

        # Transit qualifying items from STM to LTM
        for key in keys_to_transit:
            if key in self.stm.memory:
                value = self.stm.memory[key]
                # Only add to LTM if not already present
                if key not in self.ltm:
                    self.ltm[key] = value
                    transited_keys.append(key)
                    logger.info(f"Transited '{key}' from STM to LTM (freq={self.stm.frequency_table[key]})")

        return transited_keys

    def retrieve(self, query: str) -> str:
        """Retrieve knowledge from memory given a query (Memory Utilization).

        Implements Equation 3: p = Retriever(x)
        Searches both STM (Levenshtein) and LTM (semantic similarity).

        Args:
            query: The new user query x.

        Returns:
            str: Retrieved knowledge prompt p to be prepended to the query.
        """
        retrieved_items = []

        # Search STM using closest-match retriever R_c
        stm_key, stm_value, stm_score = self.stm.get_closest(query)
        if stm_key is not None:
            retrieved_items.append(f"[STM] {stm_key}: {stm_value}")
            # Increment frequency when accessed
            self.stm.increment_frequency(stm_key)

        # Search LTM using semantic-match retriever R_s
        ltm_key, ltm_value, ltm_score = self.ltm.get_closest(query)
        if ltm_key is not None:
            retrieved_items.append(f"[LTM] {ltm_key}: {ltm_value}")

        if retrieved_items:
            return "\n".join(retrieved_items)
        return ""

    def refresh_working_memory(self):
        """Refresh working memory (called after each iteration)."""
        self.working_memory.refresh()

    def refresh_stm(self):
        """Refresh STM (called periodically after certain rounds)."""
        # Before refreshing, transit high-frequency items to LTM
        self.transit()
        self.stm.refresh()

    def get_memory_state(self) -> dict:
        """Return the current state of all memory components."""
        return {
            "working_memory": {
                "size": len(self.working_memory),
                "notes": self.working_memory.get_notes(),
            },
            "stm": {
                "size": len(self.stm),
                "items": self.stm.get_all_items(),
                "frequency_table": dict(self.stm.frequency_table),
            },
            "ltm": {
                "size": len(self.ltm),
                "items": self.ltm.get_all_items(),
            },
        }

    def __str__(self):
        return (
            f"Memory State:\n"
            f"  Working Memory ({len(self.working_memory)} notes)\n"
            f"  STM ({len(self.stm)} items)\n"
            f"  LTM ({len(self.ltm)} items)"
        )
