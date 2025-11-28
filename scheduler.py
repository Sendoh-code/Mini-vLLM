class MiniScheduler:
    def __init__(self):
        pass

    def batch_prefill(self, prompts):
        """Batch prefill for multiple prompts."""
        raise NotImplementedError

    def batch_decode(self, sequences):
        """Batch decode one step for multiple sequences."""
        raise NotImplementedError
