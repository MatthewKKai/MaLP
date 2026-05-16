"""
Long-Term Memory (LTM) implementation for the DPeM mechanism.

LTM stores knowledge that has been frequently accessed in STM and
transited based on the frequency threshold θ. Knowledge in LTM is:
- Never refreshed (persistent)
- Unlimited storage capacity
- Retrieved via semantic similarity (cosine similarity with encoder)

As described in Section 2.4.3: R_s is a semantic-match retriever for LTM.
We train an encoder to obtain semantic embeddings and retrieve knowledge
in LTM based on cosine similarity.
"""

import logging
import os
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from sentence_transformers import (
        SentenceTransformer,
        InputExample,
        losses,
        evaluation,
        util,
        models,
    )
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Long_Term_Memory:
    """Long-Term Memory with semantic retrieval via sentence embeddings.

    Implements the LTM component of DPeM (Section 2.3.1):
    - M_LTM = {..., k_type : k_f, ...}
    - Never refreshed, unlimited storage
    - Retrieved through association (semantic similarity)
    - k_f denotes frequently visited k_j from M_STM

    Args:
        model_name (str): Name/path of the transformer model for embeddings.
            Default: "all-MiniLM-L6-v2".
        checkpoint_path (str, optional): Path to a trained model checkpoint.
        match_threshold (float): Minimum cosine similarity for a valid match.
            Default: 0.5.
        key_process_func (callable, optional): Key preprocessing function.
        mode (str): "eval" for inference, "train" for training the encoder.
            Default: "eval".
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2",
                 checkpoint_path: str = None, match_threshold: float = 0.5,
                 key_process_func: callable = None, mode: str = "eval"):
        self.match_threshold = match_threshold
        self.key_process_func = key_process_func
        self.memory = OrderedDict()
        self.memory_embedding = []

        if SENTENCE_TRANSFORMERS_AVAILABLE:
            if checkpoint_path and os.path.exists(checkpoint_path):
                logger.info(f"Loading SentenceTransformer from checkpoint: {checkpoint_path}")
                self.model = SentenceTransformer(checkpoint_path, device="cpu")
            else:
                logger.info(f"Initializing SentenceTransformer with model: {model_name}")
                try:
                    # Try loading as a pre-trained model directly
                    self.model = SentenceTransformer(model_name, device="cpu")
                except Exception:
                    # Build from components if direct loading fails
                    word_embedding_model = models.Transformer(model_name, max_seq_length=256)
                    pooling_model = models.Pooling(
                        word_embedding_model.get_word_embedding_dimension()
                    )
                    dense_model = models.Dense(
                        in_features=pooling_model.get_sentence_embedding_dimension(),
                        out_features=256,
                        activation_function=nn.Tanh(),
                    )
                    self.model = SentenceTransformer(
                        modules=[word_embedding_model, pooling_model, dense_model],
                        device="cpu",
                    )
            logger.info("LTM model initialized successfully.")
        else:
            logger.warning(
                "sentence-transformers not installed. LTM will use basic matching. "
                "Install with: pip install sentence-transformers"
            )
            self.model = None

    def load_model(self, model_path: str):
        """Load a trained model from a checkpoint path."""
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.model = SentenceTransformer(model_path, device="cpu")
            self.model.eval()
            logger.info(f"Model loaded from {model_path}")
        else:
            logger.warning("sentence-transformers not available, cannot load model.")

    def parse_key(self, key: str) -> str:
        """Preprocess a key using the key_process_func if available."""
        if self.key_process_func:
            return self.key_process_func(key)
        return key

    def read_examples(self, path: str, read_for_eval: bool = False):
        """Read training/evaluation examples from a JSON file.

        Expected format: list of dicts with 'sentence1', 'sentence2', 'score' keys.

        Args:
            path: Path to the JSON file.
            read_for_eval: If True, returns separate lists for evaluation.

        Returns:
            If read_for_eval: (sentences1, sentences2, scores)
            Otherwise: list of InputExample objects.
        """
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if read_for_eval:
            sentences1 = [item["sentence1"] for item in data]
            sentences2 = [item["sentence2"] for item in data]
            scores = [float(item["score"]) for item in data]
            return sentences1, sentences2, scores
        else:
            examples = []
            for item in data:
                examples.append(
                    InputExample(
                        texts=[item["sentence1"], item["sentence2"]],
                        label=float(item["score"]),
                    )
                )
            return examples

    def train(self, train_path: str, test_path: str, outpath: str,
              batch_size: int = 64, epochs: int = 1):
        """Train the semantic encoder for LTM retrieval.

        Args:
            train_path: Path to training data JSON.
            test_path: Path to test/eval data JSON.
            outpath: Path to save the trained model.
            batch_size: Training batch size. Default: 64.
            epochs: Number of training epochs. Default: 1.
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.error("sentence-transformers required for training.")
            return

        train_examples = self.read_examples(train_path)
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        train_loss = losses.CosineSimilarityLoss(self.model)

        eval_sentences1, eval_sentences2, eval_scores = self.read_examples(
            test_path, read_for_eval=True
        )
        evaluator = evaluation.EmbeddingSimilarityEvaluator(
            eval_sentences1, eval_sentences2, eval_scores
        )

        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=500,
            evaluator=evaluator,
            evaluation_steps=500,
            output_path=outpath,
        )

        logger.info(f"LTM encoder model saved to {outpath}")

    def __setitem__(self, key: str, value):
        """Store a knowledge item in LTM with its embedding.

        Args:
            key: The knowledge key (will be preprocessed and embedded).
            value: The knowledge value.
        """
        key = self.parse_key(key)
        self.memory[key] = value
        if self.model is not None:
            embedding = self.model.encode(key, convert_to_tensor=True)
            self.memory_embedding.append(embedding)
        else:
            self.memory_embedding.append(None)

    def get_closest(self, key: str):
        """Retrieve the closest matching knowledge item via semantic similarity.

        Uses cosine similarity between query embedding and stored embeddings (R_s retriever).

        Args:
            key: The query string.

        Returns:
            Tuple of (closest_key, value, score) or (None, None, None) if no match.
        """
        if len(self) == 0:
            return None, None, None
        key = self.parse_key(key)
        closest_key, closest_key_score = self._find_closest_key(key)
        if closest_key:
            return closest_key, self.memory[closest_key], closest_key_score
        return None, None, None

    @torch.no_grad()
    def _find_closest_key(self, query: str):
        """Find the closest key in LTM using cosine similarity.

        Args:
            query: The query string.

        Returns:
            Tuple of (closest_key, score) or (None, None).
        """
        if self.model is None:
            # Fallback: exact match
            if query in self.memory:
                return query, 1.0
            return None, None

        query_embedding = self.model.encode(query, convert_to_tensor=True)

        if len(self.memory_embedding) == 0:
            return None, None

        # Stack all embeddings and compute cosine similarity
        all_embeddings = torch.stack(self.memory_embedding)
        scores = util.cos_sim(query_embedding, all_embeddings).squeeze(0)
        max_score_idx = torch.argmax(scores).item()
        closest_key = list(self.memory.keys())[max_score_idx]
        closest_key_score = scores[max_score_idx].item()

        if closest_key_score >= self.match_threshold:
            return closest_key, closest_key_score
        return None, None

    def get_all_items(self) -> list:
        """Return all items in LTM as a list of (key, value) tuples."""
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
