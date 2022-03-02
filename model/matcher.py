'''
NeuralFP matcher. main method is match()
'''
from collections import defaultdict
import numpy as np
import intervaltree

class Matcher:
    """Class that acts as a manager to perform query - references match.
    """
    def __init__(self, cfg, index, references_segments):
        self.chunk_length = cfg['MATCHER']['CHUNK_LENGTH']
        self.index_nn = cfg['MATCHER']['INDEX_NN']
        self.threshold = cfg['MATCHER']['THRESHOLD']
        self.segment_length = cfg['MODEL']['DUR']  # in seconds
        self.segment_hop = cfg['MODEL']['HOP']  # in seconds
        self.overlap = cfg['MATCHER']['OVERLAP']
        self.chunk_hopsize = self.chunk_length * self.segment_hop
        self.references_segments = references_segments
        self.index = index

    def chunks(self, lst, chunk_size, overlap=0):
        '''
        Give data chunks of dimension chunk_size
        '''
        for i in range(0, len(lst), chunk_size-overlap):
            yield i, lst[i:i+chunk_size,:]

    def get_matches(self, seg_id, chunk, references):
        w_matches = defaultdict(lambda: {'score': [],  # in later steps will add a value per window
                                         'start': float('inf'),
                                         'end': -float('inf'),})
                                         #'ref_start': float('inf'),
                                         #'ref_end': -float('inf')})

        # index_search
        _, I = self.index.search(chunk, self.index_nn)  # return k similar vectors
        # offset correction
        for offset in range(len(I)):
            I[offset, :] -= offset
        # unique id
        candidates = np.unique(I)
        # Sequence match score
        for _, cid in enumerate(candidates):
            # Candidates only reduced the amount of references to perform MIPS.
            # Scores calculated following MIPS algorithm - Maximize inner product https://arxiv.org/pdf/2110.07131.pdf
            score = np.mean(np.diag(
                    np.dot(chunk, references[cid:cid + len(chunk), :].T)))
            if score >= self.threshold:
                qstart = seg_id
                qend = (seg_id + len(chunk) + 1)
                ref = self.references_segments.loc[
                    self.references_segments['segment_id']==cid, :]
                rstart = ref.intra_segment_id.item()
                refid = ref.filename.item().split('/')[-1].split('.')[0]
                difft = qstart - rstart
                unid = f'{refid}--difft{difft}'
                w_matches[unid]['score'] += [score]
                w_matches[unid]['start'] = min(w_matches[unid]['start'], qstart)
                w_matches[unid]['end'] = max(w_matches[unid]['end'], qend)
                #w_matches[unid]['ref_start'] = max(w_matches[unid]['ref_start'], rstart)
                #w_matches[unid]['ref_end'] = max(w_matches[unid]['ref_end'], rend)
        return w_matches

    def merge_matches(self, raw_matches):
        """Merge overlapping matches
        Args:
            raw_matches (list): list of matches
        Returns:
            merged_matches(list): list of merged matches.

        TODO Test with different sets of matches"""
        for i in range(1, len(raw_matches)):
            m_1 = raw_matches[i - 1]
            m_2 = raw_matches[i]
            if 'refid' in m_1 and (  # refid and difft is available in raw_matches
                    (m_1['refid'], m_1['difft']) != (m_1['refid'], m_2['difft'])):
                continue
            if m_1['end'] >= m_2['start']:
                m_2['start'] = m_1['start']
                m_2['score'] = m_1['score'] + m_2['score']
                m_1['score'] = None  # mark m_1 to be removed later
        merged_matches = []
        for match in raw_matches:
            if match['score']:
                merged_matches.append(match)
        return merged_matches

    def select_longest(self, matches):
        """Select the longest match when a variety of them overlap.
        Args:
            matches (list): List of overlapping matches.
        Returns:
            match (dict): Longest match of the input list.

        """
        # in order to avoid choosing random ref_id in case there is no longest:
        matches = sorted(matches, key=lambda x: (x['refid'], x['difft']))
        max_match_length = 0
        longest_match = {}
        for match in matches:
            match_length = match['end'] - match['start']
            if match_length > max_match_length:
                max_match_length = match_length
                longest_match = match
        return longest_match.copy()

    def postprocess_raw_matches(self, all_matches: dict):
        """Postprocess raw matches.
        Sorts all raw matches by start time and calls merge_matches to merge
        overlapping matches. Then, returns the merged matches with the correct
        format (processed_matches).

        Args:
            all_matches (dict): Matches (common peaks) between query and index.
        Returns:
            processed_matches (list): Processed matches.
        """
        processed_matches = []
        for refid_difft, matches in all_matches.items():
            sorted_matches = sorted(matches, key=lambda x: x['start'])
            merged_matches = self.merge_matches(sorted_matches)
            for match in merged_matches:
                ref_id, diff_time = [x for x in refid_difft.split('--difft')]
                match.update({'refid': ref_id,
                              'difft': diff_time})
                processed_matches.append(match)
        return processed_matches

    def select_best_matches(self, processed_matches):
        """Select best matches.
        Selects longest matches in case two different matches overlap. It also
        performs a merge after the select_longest selection is made.

        TODO Test with different sets of matches
        """
        itree = intervaltree.IntervalTree()
        for match in processed_matches:
            if match['start'] == match['end']: # this happens when matching against reference in index
                continue
            itree.addi(match['start'], match['end'], match)
        itree.split_overlaps()
        grouped_overlaps = defaultdict(list)
        for interval in itree:
            istart_iend = f"{interval.begin}__{interval.end}"
            grouped_overlaps[istart_iend].append(interval.data)

        best_matches = []
        for istart_iend, matches in grouped_overlaps.items():
            istart, iend = [float(x) for x in istart_iend.split('__')]
            selected_match = self.select_longest(matches)
            selected_match['start'] = istart
            selected_match['end'] = iend
            best_matches.append(selected_match)
        best_matches = self.merge_matches(sorted(best_matches,
                                            key=lambda x: x['start']))
        return best_matches

    def format_matches(self, matches: list):
        """Standarizes matches format.
        Args:
            matches (list): Matches to format.
        Returns:
            formatted_matches (list of dicts)
        """
        formatted_matches = []
        for match in matches:
            formatted_matches.append({
                'query_start': match['start'] * self.segment_hop,
                'query_end': match['end'] * self.segment_hop,
                'ref': match['refid'],
                'ref_start': (
                    match['start'] - float(match['difft'])) * self.segment_hop,
                'ref_end': (
                    match['end'] - float(match['difft'])) * self.segment_hop,
                'score': np.mean(match['score'])})
        return formatted_matches

    def match(self, query, references_fp):
        """Matcher main method.
        Args:
            query: Query embeddings (fp)
            references_fp: memap matrix representing all the embeddings from
                    all references
        Returns:
            formatted_matches (list of dicts): matches in the correct format
                    to be read by fpqa
        """
        all_matches = defaultdict(list)
        for seg_id, chunk in self.chunks(query,
                                         self.chunk_length,
                                         overlap=self.overlap):
            wmatches = self.get_matches(seg_id, chunk, references_fp)
            for refid_difft, match in wmatches.items():
                all_matches[refid_difft].append(match)

        processed_matches = self.postprocess_raw_matches(all_matches)
        best_matches = self.select_best_matches(processed_matches)
        formatted_matches = self.format_matches(best_matches)
        return formatted_matches
