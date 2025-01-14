# from collections import OrderDict

# class Long_Term_Memory():
#     def __init__(self):
#         self.long_term_memory = {}

#     def __setitem__(self):
#         pass

#     def __getitem__(self):
#         pass

#     def __len__(self):
#         return len(self.long_term_memory)

    
class Long_Term_Memory():
    def __init__(self, model_name: str, checkpoint_path: str = None, match_threshold: float = 0.9,
    key_process_func: callable = None, mode: str = "eval"):
        if mode == "eval":
            assert checkpoint_path is not None, "checkpoint_path must be provided for eval mode"
        word_embedding_model = models.Transformer(model_name, max_seq_length=256)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        dense_model = models.Dense(
            in_features=pooling_model.get_sentence_embedding_dimension(),
            out_features=256,
            activation_function=nn.Tanh(),
        )
        logging.info("Initializing model...")
        self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model], device="cpu")
        logging.info("Model initialized")
        self.memory = OrderedDict()
        self.memory_embedding = []
        self.match_threshold = match_threshold
        self.key_process_func = key_process_func
        if checkpoint_path:
            logging.info(f"Loading model from checkpoint {checkpoint_path}")
            self.load_model(checkpoint_path)

    def load_model(self, model_path: str):
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()

    def parse_key(self, key: str) -> str:
        if self.key_process_func:
            return self.key_process_func(key)
        else:
            return key

    def train(self, train_path: str, test_path: str, outpath: str, batch_size: int = 64, epochs: int = 1):

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
        )

        # save model
        torch.save(self.model.state_dict(), outpath)
    
    def __setitem__(self, key, value):
        key = self.parse_key(key)
        self.memory[key] = value
        self.memory_embedding.append(self.model.encode(key))

    
    def get_closest(self, key):
        if len(self) == 0:
            return None, None, None
        key = self.parse_key(key)
        closest_key, closest_key_score = self._find_closest_key(key)
        if closest_key:
            return closest_key, self[closest_key], closest_key_score
        else:
            return None, None, None
    
    @torch.no_grad()
    def _find_closest_key(self, query):
        query_embedding = self.model.encode(query)
        
        # find closest key
        scores = util.cos_sim(query_embedding, self.memory_embedding).squeeze(0)
        max_score_idx = torch.argmax(scores).item()
        closest_key = list(self.memory.keys())[max_score_idx]

        # get closest key score
        closest_key_score = scores[max_score_idx].item()

        if closest_key_score > self.match_threshold:
            return closest_key, closest_key_score
        else:
            return None, None


    def __len__(self):
        return len(self.memory)

    def __getitem__(self, item):
        return self.memory[item]

    def __str__(self) -> str:
        return " || ".join([f"{k}: {v}" for (k, v) in self.memory.items()])