class Sequence:
    """Represents a single sequence in decoding."""

    def __init__(self, seq_id, prompt_tokens):
        self.seq_id = seq_id
        self.prompt_tokens = prompt_tokens
        self.output_tokens = []
        self.kv_cache = []

    @property
    def next_token(self):
        if len(self.output_tokens) == 0:
            return None
        return self.output_tokens[-1]
