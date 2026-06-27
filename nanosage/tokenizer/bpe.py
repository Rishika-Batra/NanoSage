import json
import os
import re

class BPETokenizer:
    def __init__(self):
        # Base vocabulary consists of individual byte values (0-255)
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}  # (int, int) -> int (new token id)
        
        # Special tokens setup
        self.special_tokens = {"<|endoftext|>": 256}
        self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}
        
        # Add special tokens to vocab mapping
        for token, idx in self.special_tokens.items():
            self.vocab[idx] = token.encode("utf-8", errors="replace")

    def _get_stats(self, ids):
        """Count frequencies of adjacent token pairs."""
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def _merge(self, ids, pair, idx):
        """Replace all occurrences of pair in ids with idx."""
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def train(self, text, vocab_size, verbose=False):
        """Train the BPE tokenizer on a given text to achieve target vocab_size."""
        assert vocab_size >= 257, "vocab_size must be at least 257 to cover bytes + 1 special token"
        num_merges = vocab_size - 257
        
        # Convert text to UTF-8 bytes representation (list of integers 0-255)
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)
        
        merges = {}
        vocab = {i: bytes([i]) for i in range(256)}
        
        # Special token index start
        current_idx = 257
        
        if verbose:
            print(f"Training BPE tokenizer. Input bytes length: {len(ids)}")
            print(f"Goal: perform {num_merges} merges to reach vocab size {vocab_size}")

        for i in range(num_merges):
            stats = self._get_stats(ids)
            if not stats:
                break  # No more pairs to merge
            
            # Find the most frequent pair
            best_pair = max(stats, key=stats.get)
            
            # Record merge
            merges[best_pair] = current_idx
            vocab[current_idx] = vocab[best_pair[0]] + vocab[best_pair[1]]
            
            # Apply merge to token sequence
            ids = self._merge(ids, best_pair, current_idx)
            
            if verbose and (i + 1) % max(1, num_merges // 10) == 0:
                print(f"Merge {i+1}/{num_merges}: merged {best_pair} into {current_idx}")
                
            current_idx += 1
            
        self.merges = merges
        self.vocab = vocab
        
        # Ensure special tokens are present in final vocabulary
        for token, idx in self.special_tokens.items():
            self.vocab[idx] = token.encode("utf-8", errors="replace")

    def encode(self, text, allowed_special="all"):
        """
        Encode string into token IDs.
        allowed_special: "all" or set/list of special tokens allowed in string.
        """
        if not text:
            return []
            
        # Parse special tokens first if present
        if allowed_special == "all":
            special_pattern = re.compile("|".join(re.escape(t) for t in self.special_tokens.keys()))
        elif allowed_special:
            special_pattern = re.compile("|".join(re.escape(t) for t in allowed_special))
        else:
            special_pattern = None

        if not special_pattern:
            return self._encode_chunk(text)

        # Split text by special tokens
        parts = special_pattern.split(text)
        specials = special_pattern.findall(text)
        
        ids = []
        for i, part in enumerate(parts):
            ids.extend(self._encode_chunk(part))
            if i < len(specials):
                ids.append(self.special_tokens[specials[i]])
                
        return ids

    def _encode_chunk(self, text_chunk):
        """Encode a piece of text that contains no special tokens."""
        text_bytes = text_chunk.encode("utf-8")
        ids = list(text_bytes)
        
        # Iteratively apply merges in the exact order they were learned
        while len(ids) >= 2:
            stats = self._get_stats(ids)
            # Find the pair in ids that has the lowest merge rank (was merged first)
            pair_to_merge = min(
                (pair for pair in stats if pair in self.merges),
                key=lambda p: self.merges[p],
                default=None
            )
            
            if pair_to_merge is None:
                break
                
            new_idx = self.merges[pair_to_merge]
            ids = self._merge(ids, pair_to_merge, new_idx)
            
        return ids

    def decode(self, ids):
        """Decode token IDs back to a string."""
        byte_parts = []
        for idx in ids:
            if idx in self.vocab:
                byte_parts.append(self.vocab[idx])
            else:
                raise ValueError(f"Token ID {idx} is not in vocabulary")
        
        # Concatenate all byte strings
        concatenated_bytes = b"".join(byte_parts)
        # Decode utf-8 with error replacement
        return concatenated_bytes.decode("utf-8", errors="replace")

    def save(self, file_path):
        """Save the tokenizer merges and vocabulary to a JSON file."""
        # Convert tuple keys of merges dictionary to string for JSON serialization
        serialized_merges = {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}
        
        state = {
            "merges": serialized_merges,
            "special_tokens": self.special_tokens
        }
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=4)

    def load(self, file_path):
        """Load tokenizer state from a JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            state = json.load(f)
            
        serialized_merges = state.get("merges", {})
        self.special_tokens = state.get("special_tokens", {"<|endoftext|>": 256})
        self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}
        
        # Reconstruct merges
        self.merges = {}
        for pair_str, val in serialized_merges.items():
            k1, k2 = map(int, pair_str.split(","))
            self.merges[(k1, k2)] = val
            
        # Reconstruct vocabulary
        self.vocab = {i: bytes([i]) for i in range(256)}
        
        # Re-apply merges in ascending order of target index
        sorted_merges = sorted(self.merges.items(), key=lambda x: x[1])
        for (p0, p1), target_idx in sorted_merges:
            self.vocab[target_idx] = self.vocab[p0] + self.vocab[p1]
            
        # Add special tokens
        for token, idx in self.special_tokens.items():
            self.vocab[idx] = token.encode("utf-8", errors="replace")
