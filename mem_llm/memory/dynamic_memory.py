from collections import OrderedDict

# class Short_Term_Memory(object):
#     def __init__(self):
#         self.memory = OrderedDict()

#     def erase(self, item):
#         pass

#     def __setitem__(self):
#         pass

#     def __getitem__(self):
#         pass

#     def __len__(self):
#         return len(self.memory)

class Short_Term_Memory():
    def __init__(self, key_process_func: callable, min_sim_threshold: int = 6):
        """A memory that supports fuzzy matching of keys. The fuzzy matching is done by using levenshtein distance.

        Args:
            key_process_func (callable): A string --> string function, that is used to parse the keys before they
            are used to query or added.
            min_sim_threshold (int, optional): [description]. Defaults to 6.
        """
        self.memory = dict()
        self.min_sim_threshold = min_sim_threshold
        self.key_process_func = key_process_func
        self.min_sim_threshold = min_sim_threshold

    def parse_key(self, key: str) -> str:
        if self.key_process_func:
            return self.key_process_func(key)
        else:
            return key

    
    def __setitem__(self, key, value):
        key = self.parse_key(key)
        self.memory[key] = value

    def get_closest(self, key, return_score: bool = True):
        if len(self) == 0:
            return None, None, None
        key = self.parse_key(key)
        closest_key, closest_key_score = self._find_closest_key(key)
        if closest_key:
            if return_score:
                return closest_key, self[closest_key], closest_key_score
        else:
            return None, None, None
    

    def _find_closest_key(self, word):
        # find the key in self.memory that is closest to the word in terms of levenshtein distance
        min_dist = self.min_sim_threshold
        logging.debug("Finding closest key for word: {}".format(word))
        closest_key = None
        for key in self.memory:
            dist = lev.distance(word, key)
            logging.debug("Distance between {} and {} is {}".format(word, key, dist))
            if dist < min_dist:
                min_dist = dist
                closest_key = key
        if closest_key:
            return closest_key, min_dist
        else:
            return None, min_dist

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, item):
        return self.memory[item]

    def __str__(self) -> str:
        return " || ".join([f"{k}: {v}" for (k, v) in self.memory.items()])