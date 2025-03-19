import torch
from itertools import groupby
import torch.nn.functional as F

class Decode(object):
    def __init__(self, gloss_dict, num_classes, search_mode, blank_id=0, beam_width=10):
        self.i2g_dict = dict((v[0], k) for k, v in gloss_dict.items())
        self.g2i_dict = {v: k for k, v in self.i2g_dict.items()}
        self.num_classes = num_classes
        self.search_mode = search_mode
        self.blank_id = blank_id
        self.beam_width = beam_width
        self.vocab = [chr(x) for x in range(20000, 20000 + num_classes)]

    def decode(self, nn_output, vid_lgt, batch_first=True, probs=False):
        if not batch_first:
            nn_output = nn_output.permute(1, 0, 2)
        if self.search_mode == "max":
            return self.MaxDecode(nn_output, vid_lgt)
        else:
            return self.BeamSearch(nn_output, vid_lgt, probs)

    def BeamSearch(self, nn_output, vid_lgt, probs=False):
        """
        This method performs beam search decoding on the network outputs.
        It expects nn_output of shape (B, T, N) where each row has been
        passed through a softmax (unless probs is True) so that they are probabilities.
        vid_lgt is a tensor containing the valid sequence length (T) for each sample in the batch.
        The result is returned as a list for each batch element, where each element is a list of tuples:
        (decoded gloss string, index in the decoded sequence).
        """
        if not probs:
            nn_output = nn_output.softmax(dim=-1)
        nn_output = nn_output.cpu()
        vid_lgt = vid_lgt.cpu()
        ret_list = []
        batch_size = nn_output.shape[0]
        for i in range(batch_size):
            T = int(vid_lgt[i].item())
            probs_i = nn_output[i][:T]
            decoded_seq, score = self._beam_search_decode_single(probs_i)
            dedup = [token for token, _ in groupby(decoded_seq)]
            ret_list.append([(self.i2g_dict[int(token)], idx) for idx, token in enumerate(dedup)])
        return ret_list

    def _beam_search_decode_single(self, probs):
        """
        Perform beam search decoding on a single sequence.

        Args:
            probs: Tensor of shape (T, V) where T is time steps and V is vocabulary size.
                   Each row is assumed to be a probability distribution over the vocabulary.

        Returns:
            final_seq: The decoded sequence (a list of token indices) after applying CTC merging rules.
            best_prob: The probability score associated with the best sequence.

        This implementation uses the standard CTC beam search algorithm. At each time step,
        it maintains a set of candidate prefixes (each a tuple of integers) along with two probability
        components: one for sequences ending in the blank and one for sequences ending in a non-blank.
        """
        T, V = probs.shape
        beam = {(): (1.0, 0.0)}  # value: (p_blank, p_non_blank)

        for t in range(T):
            new_beam = {}
            for prefix, (p_b, p_nb) in beam.items():
                total_prob = p_b + p_nb
                for c in range(V):
                    p = probs[t, c].item()
                    if c == self.blank_id:
                        prev_p_b, prev_p_nb = new_beam.get(prefix, (0.0, 0.0))
                        new_beam[prefix] = (prev_p_b + total_prob * p, prev_p_nb)
                    else:
                        new_prefix = prefix + (c,)
                        if len(prefix) > 0 and prefix[-1] == c:
                            prev_p_b, prev_p_nb = new_beam.get(new_prefix, (0.0, 0.0))
                            new_beam[new_prefix] = (prev_p_b, prev_p_nb + p_b * p)
                        else:
                            prev_p_b, prev_p_nb = new_beam.get(new_prefix, (0.0, 0.0))
                            new_beam[new_prefix] = (prev_p_b, prev_p_nb + total_prob * p)
            beam = dict(sorted(new_beam.items(), key=lambda x: sum(x[1]), reverse=True)[:self.beam_width])

        best_prefix, best_prob_tuple = max(beam.items(), key=lambda x: sum(x[1]))
        best_prob = sum(best_prob_tuple)

        final_seq = []
        for token in best_prefix:
            if len(final_seq) == 0 or final_seq[-1] != token:
                final_seq.append(token)
        final_seq = [token for token in final_seq if token != self.blank_id]
        return final_seq, best_prob

    def MaxDecode(self, nn_output, vid_lgt):
        """
        Greedy (max) decoding: For each time step, choose the token with the maximum probability.
        After decoding each sample in the batch, consecutive duplicate tokens are merged and blanks removed.
        Returns a list (per batch element) of tuples (gloss string, index in decoded sequence).
        """

        index_list = torch.argmax(nn_output, dim=2)
        batch_size, _ = index_list.shape
        ret_list = []
        for i in range(batch_size):
            indices = index_list[i][:int(vid_lgt[i].item())]
            dedup = [key for key, _ in groupby(indices.tolist())]
            filtered = [x for x in dedup if x != self.blank_id]
            ret_list.append([(self.i2g_dict[int(token)], idx) for idx, token in enumerate(filtered)])
        return ret_list

if __name__ == "__main__":
    gloss_dict = {'A': (1,), 'B': (2,), 'C': (3,), 'blank': (0,)}
    num_classes = 4
    search_mode = "beam"

    decoder = Decode(gloss_dict, num_classes, search_mode, blank_id=0, beam_width=10)

    nn_output = torch.rand(2, 5, num_classes)
    valid_sequence_lengths = torch.tensor([5, 5])
    result = decoder.decode(nn_output, valid_sequence_lengths, batch_first=True)
    print("Beam Search Decoding Result:", result)

    decoder.search_mode = "max"
    result_max = decoder.decode(nn_output, valid_sequence_lengths, batch_first=True)
    print("Max (Greedy) Decoding Result:", result_max)
