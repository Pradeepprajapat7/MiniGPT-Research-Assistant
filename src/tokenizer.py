
class CharTokenizer:
    def __init__(self, text: str):
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)

        self.stoi = {
            char: index
            for index, char in enumerate(self.chars)
        }

        self.itos = {
            index: char
            for char, index in self.stoi.items()
        }

    def encode(self, text: str) -> list[int]:
        unknown = set(text) - set(self.stoi)
        if unknown:
            raise ValueError(f"Unknown characters: {unknown}")

        return [self.stoi[char] for char in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(
            self.itos[token_id]
            for token_id in token_ids
        )
