


SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"

class Vocabulary(object):
    def __init__(self, name):
        self.name = name
        self.token2count = {}
        self.index2token = [SOS_TOKEN, EOS_TOKEN]
        self.token2index = {token: idx for idx, token in enumerate(self.index2token)}

        self.sos_token_id = self.token2index[SOS_TOKEN]
        self.eos_token_id = self.token2index[EOS_TOKEN]

    @classmethod
    def from_index2token(self, index2token):
        vocab = Vocabulary("unknown")
        vocab.index2token = index2token
        vocab.token2index = {token: idx for idx, token in enumerate(vocab.index2token)}
        vocab.sos_token_id = vocab.token2index[SOS_TOKEN]
        vocab.eos_token_id = vocab.token2index[EOS_TOKEN]
        return vocab

    def add_token(self, token: str):
        if token not in self.token2index:
            self.token2index[token] = len(self.index2token)
            self.token2count[token] = 1
            self.index2token.append(token)
        else:
            self.token2count[token] += 1

    def add_sequence(self, sequence: list[str]):
        for token in sequence:
            self.add_token(token)

    def __len__(self):
        return len(self.index2token)

    def toJSON(self):
        return {
            "name": self.name,
            "index2token": self.index2token,
            "sos_token_id": self.sos_token_id,
            "eos_token_id": self.eos_token_id,
        }