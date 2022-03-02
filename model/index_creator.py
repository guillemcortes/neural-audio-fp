import logging

import numpy as np
import faiss


def create_index(path_data, path_shape, index_path, cfg):
    """ Create Faiss index.
    While Using IVF-PQ index if no enough training data --> try smaller code_sz and n_centroids
        code_sz power of 2
        nbits must be 8, 12 or 16, The dimension d should be a multiple of M.
    Search:
        1. Find nprobe closest buckets to target vector
        2. Find most similar k vectors.
    Default values:
        index_type='ivfpq', n_centroids=256, code_sz=64, nbits=8, nprobe=40
    """
    index_type = cfg['INDEX']['INDEX_TYPE']
    n_centroids = cfg['INDEX']['N_CENTROIDS']
    code_sz = cfg['INDEX']['CODE_SZ']
    nbits = cfg['INDEX']['NBITS']
    nprobe = cfg['INDEX']['NPROBE']
    data_shape = np.load(path_shape)
    emb_dim = int(data_shape[1]) # Fingerprint dimension
    data = np.memmap(path_data, dtype='float32', mode='r+',
                        shape=(data_shape[0], emb_dim))

    index = faiss.IndexFlatL2(emb_dim)   # build the index -- use it as quantizer
    if index_type == 'ivfpq':
        index = faiss.IndexIVFPQ(index, emb_dim, n_centroids, code_sz, nbits)
        max_nitem_train = int(np.floor(len(data)/2))

        if len(data) > max_nitem_train:
            logging.info('Training index using {:>3.2f} % of data...'.format(
                100. * max_nitem_train / len(data)))
            # shuffle and reduce training data
            sel_tr_idx = np.random.permutation(len(data))
            sel_tr_idx = sel_tr_idx[:max_nitem_train]
            index.train(data[sel_tr_idx,:])
        else:
            logging.info('Training index...')
            index.train(data) # Actually do nothing for {'l2', 'hnsw'}
        # N probe
        index.nprobe = nprobe

    logging.info('Trained index? --> %s', index.is_trained)
    index.add(data)                  # add vectors to the index
    faiss.write_index(index, index_path)
    return
