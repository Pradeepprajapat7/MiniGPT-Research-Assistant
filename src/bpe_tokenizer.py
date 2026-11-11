
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


def train_bpe(input_file, vocab_size=1000):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"]
    )

    tokenizer.train([input_file], trainer)
    tokenizer.save("checkpoints/tokenizer.json")

    return tokenizer


def load_bpe():
    return Tokenizer.from_file(
        "checkpoints/tokenizer.json"
    )
