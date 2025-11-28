class KVCacheManager:
    def __init__(self):
        # req_id -> past_key_values
        self.cache_store = {}

    def has_cache(self, req_id):
        return req_id in self.cache_store

    def get_cache(self, req_id):
        return self.cache_store.get(req_id, None)

    def update_cache(self, req_id, past_kv):
        self.cache_store[req_id] = past_kv

    def remove(self, req_id):
        if req_id in self.cache_store:
            del self.cache_store[req_id]
