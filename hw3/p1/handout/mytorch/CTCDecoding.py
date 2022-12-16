import numpy as np


class GreedySearchDecoder(object):

    def __init__(self, symbol_set):
        
        """Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        """

        self.symbol_set = symbol_set

    def decode(self, y_probs):
        
        """Perform greedy search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
            batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        decoded_path [str]:
            compressed symbol sequence i.e. without blanks or repeated symbols

        path_prob [float]:
            forward probability of the greedy path

        """

        blank = 0
        batch_compressed = []
        batch_path_probs = []
        (_, T, batch_size) = y_probs.shape

        for batch_itr in range(batch_size):

            decoded_path = []
            path_prob = 1

            for seq_itr in range(T):

                max_loc = np.argmax(y_probs[:, seq_itr, batch_itr])
                symbol = self.symbol_set[max_loc-1]
                
                if len(decoded_path) == 0:
                    decoded_path.append(symbol)

                elif decoded_path[-1] != symbol:
                    decoded_path.append(symbol)
                
                path_prob *= y_probs[max_loc, seq_itr, batch_itr]

            batch_compressed.append(''.join(decoded_path))
            batch_path_probs.append(path_prob)
        
        # for autograder assuming batch size 1
        return batch_compressed[0], batch_path_probs[0]

class BeamSearchDecoder(object):

    PathScore, BlankPathScore = {}, {}

    def __init__(self, symbol_set, beam_width):

        """Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        beam_width [int]:
            beam width for selecting top-k hypotheses for expansion

        """

        self.symbol_set = symbol_set
        self.beam_width = beam_width
    
    def InitializePaths(self, y):
        
        InitialPathsWithFinalBlank = ['']
        InitialBlankPathScore = {'': y[0]}

        InitialPathsWithFinalSymbol = []
        InitialPathScore = {}

        for i, sym in enumerate(self.symbol_set):

            InitialPathsWithFinalSymbol.append(sym)
            InitialPathScore[sym] = y[i+1]

        return (InitialPathsWithFinalBlank, InitialPathsWithFinalSymbol,
                InitialBlankPathScore, InitialPathScore)

    def Prune(self, PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore):
        
        PrunedBlankPathScore = {}
        PrunedPathScore = {}

        scorelist = []

        for p in PathsWithTerminalBlank:
            scorelist.append(BlankPathScore[p][0])
 
        for p in PathsWithTerminalSymbol:
            scorelist.append(PathScore[p][0])
        
        scorelist = sorted(scorelist, reverse=True)

        if self.beam_width < len(scorelist):
            cutoff = scorelist[self.beam_width-1]
        
        else:
            cutoff = scorelist[-1]

        PrunedPathsWithTerminalBlank = []

        for p in PathsWithTerminalBlank:
            if BlankPathScore[p] >= cutoff:
                PrunedPathsWithTerminalBlank.append(p)
                PrunedBlankPathScore[p] = BlankPathScore[p]
        
        PrunedPathsWithTerminalSymbol = []

        for p in PathsWithTerminalSymbol:
            if PathScore[p] >= cutoff:
                PrunedPathsWithTerminalSymbol.append(p)
                PrunedPathScore[p] = PathScore[p]
        
        return PrunedPathsWithTerminalBlank, PrunedPathsWithTerminalSymbol, PrunedBlankPathScore, PrunedPathScore

    def ExtendWithBlank(self, PathsWithTerminalBlank, PathsWithTerminalSymbol, y):
        
        UpdatedPathsWithTerminalBlank = []
        UpdatedBlankPathScore = {}

        for path in PathsWithTerminalBlank:

            UpdatedPathsWithTerminalBlank.append(path)
            UpdatedBlankPathScore[path] = self.BlankPathScore[path] * y[0]
        
        for path in PathsWithTerminalSymbol:
            
            if path in UpdatedPathsWithTerminalBlank:
                UpdatedBlankPathScore[path] += self.PathScore[path] * y[0]
            
            else:
                UpdatedPathsWithTerminalBlank.append(path)
                UpdatedBlankPathScore[path] = self.PathScore[path] * y[0]
        
        return UpdatedPathsWithTerminalBlank, UpdatedBlankPathScore

    def ExtendWithSymbol(self, PathsWithTerminalBlank, PathsWithTerminalSymbol, y):
        
        UpdatedPathsWithTerminalSymbol = []
        UpdatedPathScore = {}

        for path in PathsWithTerminalBlank:
            
            for i, c in enumerate(self.symbol_set):
            
                newpath = path + c
            
                UpdatedPathsWithTerminalSymbol.append(newpath)
                UpdatedPathScore[newpath] = self.BlankPathScore[path] * y[i+1]
        
        for path in PathsWithTerminalSymbol:

            for i, c in enumerate(self.symbol_set):

                if c == path[-1]:
                    newpath = path
                
                else:
                    newpath = path + c

                if newpath in UpdatedPathsWithTerminalSymbol:
                    UpdatedPathScore[newpath] += self.PathScore[path] * y[i+1]

                else:
                    UpdatedPathsWithTerminalSymbol.append(newpath)
                    UpdatedPathScore[newpath] = self.PathScore[path] * y[i+1]
        
        return UpdatedPathsWithTerminalSymbol, UpdatedPathScore

    def MergeIdenticalPaths(self, PathsWithTerminalBlank, BlankPathScore, PathsWithTerminalSymbol, PathScore):
        
        MergedPaths = PathsWithTerminalSymbol
        FinalPathScore = PathScore

        for p in PathsWithTerminalBlank:
            
            if p in MergedPaths:
                FinalPathScore[p] += BlankPathScore[p][0]
            
            else:
                MergedPaths.append(p)
                FinalPathScore[p] = BlankPathScore[p][0]
        
        return MergedPaths, FinalPathScore

    def decode(self, y_probs):

        """Perform beam search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
			batch size for part 1 will remain 1, but if you plan to use your
			implementation for part 2 you need to incorporate batch_size

        Returns
        -------
        
        forward_path [str]:
            the symbol sequence with the best path score (forward probability)

        merged_path_scores [dict]:
            all the final merged paths with their scores

        """

        (_, T, batch_size) = y_probs.shape
        bestPath, FinalPathScore = None, None
        

        (NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol,
         NewBlankPathScore, NewPathScore) = self.InitializePaths(y_probs[:,0])

        for t in range(1, T):

            (PathsWithTerminalBlank, PathsWithTerminalSymbol,
                self.BlankPathScore, self.PathScore) = self.Prune(NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol,
                                                                NewBlankPathScore, NewPathScore)
            
            NewPathsWithTerminalBlank, NewBlankPathScore = self.ExtendWithBlank(PathsWithTerminalBlank,
                                                                                PathsWithTerminalSymbol, y_probs[:, t])
            
            NewPathsWithTerminalSymbol, NewPathScore = self.ExtendWithSymbol(PathsWithTerminalBlank, PathsWithTerminalSymbol, y_probs[:, t])
            NewPathsWithTerminalSymbol = list(NewPathScore.keys())

        MergedPaths, FinalPathScore = self.MergeIdenticalPaths(NewPathsWithTerminalBlank, NewBlankPathScore, 
                                                                NewPathsWithTerminalSymbol, NewPathScore)
        
        bestPath = sorted(FinalPathScore, key=FinalPathScore.get, reverse=False)[-1]

        return bestPath, FinalPathScore