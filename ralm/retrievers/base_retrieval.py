class BaseRetriever:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def retrieve(self, sequence_input_ids, dataset, k=1):
        raise NotImplementedError
