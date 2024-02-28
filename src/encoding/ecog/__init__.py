from mne.decoding import ReceptiveField




class TemporalReceptiveField(ReceptiveField):

    def score(self, X, y):
        # parent class returns one score per output
        scores = super().score(X, y)
        return scores.mean()
    
    def score_multidimensional(self, X, y):
        return super().score(X, y)