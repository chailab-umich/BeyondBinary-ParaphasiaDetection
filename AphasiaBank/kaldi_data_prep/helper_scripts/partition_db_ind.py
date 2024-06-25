import os
from collections import OrderedDict
import random

import chaipy.common as common
import chaipy.io as io

def random_partition(lst, n, rand=random.Random()):
    """ Return a `n`-part random partition of lst. We try to keep the
    chunks roughly the same size. The ordering of chunk size is also
    random. This randomness is controled by `rand`.
    """
    common.CHK_GE(len(lst), n)
    split = [[] for i in range(n)]
    # Prepare indices
    lst_indices = range(len(lst))
    rand.shuffle(lst_indices)
    split_indices = range(n)
    rand.shuffle(split_indices)
    # Add evenly split chunks
    div = len(lst) / n
    for i, split_idx in zip(range(n), split_indices):
        for lst_idx in lst_indices[i*div:(i+1)*div]:
            split[split_idx].append(lst[lst_idx])
    # Add remainders to random partitions
    rem = len(lst) % n
    common.CHK_EQ(rem, len(lst_indices) - (n * div))
    for i, lst_idx in zip(range(rem), lst_indices[n*div:]):
        split[split_indices[i]].append(lst[lst_idx])
    return split

def random_partition(lst, n, rand=random.Random()):
    """ Return a `n`-part random partition of lst. We try to keep the
    chunks roughly the same size. The ordering of chunk size is also
    random. This randomness is controled by `rand`.
    """
    common.CHK_GE(len(lst), n)
    split = [[] for i in range(n)]
    # Prepare indices
    lst_indices = range(len(lst))
    rand.shuffle(lst_indices)
    split_indices = range(n)
    rand.shuffle(split_indices)
    # Add evenly split chunks
    div = len(lst) / n
    for i, split_idx in zip(range(n), split_indices):
        for lst_idx in lst_indices[i*div:(i+1)*div]:
            split[split_idx].append(lst[lst_idx])
    # Add remainders to random partitions
    rem = len(lst) % n
    common.CHK_EQ(rem, len(lst_indices) - (n * div))
    for i, lst_idx in zip(range(rem), lst_indices[n*div:]):
        split[split_indices[i]].append(lst[lst_idx])
    return split

def main():
    desc = 'Split data into N folds, each with a train, dev, and test set. ' + \
            'We split by speaker, i.e. each database contributes a number ' + \
            'of speakers to each set. This maintains speaker-independence. ' + \
            'This script will output the list of utt names. trim.py can ' + \
            'then be used to actually trim the data.'
    parser = common.init_argparse(desc)
    parser.add_argument('spk2db', help='Kaldi spk2db')
    parser.add_argument('spk2utt', help='Kaldi spk2utt')
    parser.add_argument('output_dir', help='Where to output')
    parser.add_argument('--nfolds', type=int, default=4,
                        help='Number of folds to create')
    parser.add_argument('--devfrac', type=float, default=0.15,
                        help='Amount of training data to withhold for dev set')
    parser.add_argument('--seed', type=int, default=883, help='Random seed')
    args = parser.parse_args()

    spk2db = io.dict_read(args.spk2db, ordered=True)
    io.log('Loaded {} records from {}'.format(len(spk2db), args.spk2db))
    spk2utt = io.dict_read(args.spk2utt, ordered=True)
    io.log('Loaded {} records from {}'.format(len(spk2utt), args.spk2utt))
    common.CHK_EQ(len(spk2db), len(spk2utt))

    db2spk = common.make_reverse_index(spk2db, ordered=True)
    rand = random.Random(args.seed)

    # Mapping from db to `args.nfolds` lists of speaker names
    partitions_train = OrderedDict()
    partitions_test = OrderedDict()

    tr_dev_db = db2spk.keys()
    remove_lst = ['adler', 'elman', 'kurland']
    for r in remove_lst:
        tr_dev_db.remove(r)
    test_db = remove_lst

    for db in tr_dev_db:
        io.log('Partitioning speakers for {}'.format(db))
        partitions_train[db] = random_partition(db2spk[db], args.nfolds, rand=rand)

    for db in test_db:
        io.log('Partitioning speakers for {}'.format(db))
        partitions_test[db] = random_partition(db2spk[db], args.nfolds, rand=rand)

    # print("partitions: {}".format(partitions))
    # exit()
    # Output stuff
    # for fold in range(args.nfolds) + [-1]:
    for fold in range(args.nfolds) + [-1]:
        # fold += 10

        fold_dir = os.path.join(args.output_dir, 'Fold_{}'.format(fold + 1 + 10))
        io.log('Fold {}, outputting to {}'.format(fold + 1, fold_dir))
        set_utts, set_dirs = OrderedDict(), OrderedDict()
        for sname in (['train', 'dev', 'test'] if fold != -1 else ['train', 'dev']):
            set_utts[sname] = []
            set_dirs[sname] = os.path.join(fold_dir, sname)
            if not os.path.exists(set_dirs[sname]):
                os.makedirs(set_dirs[sname])

        # test
        for db in partitions_test.keys():
            io.log('...processing {}'.format(db))
            # Test set
            if fold != -1:
                for test_spk in partitions_test[db][fold]:
                    # print("test_spk {}".format(test_spk))
                    # print("spk2utt[test_spk]: {}".format(spk2utt[test_spk]))
                    set_utts['test'].extend(spk2utt[test_spk])
                io.log('Added {} spks to test'.format(len(partitions_test[db][fold])))


        # train
        for db in partitions_train.keys():
            io.log('...processing {}'.format(db))
            # # Test set
            # if fold != -1:
            #     for test_spk in partitions_train[db][fold]:
            #         print("spk2utt[test_spk]: {}".format(spk2utt[test_spk]))
            #         set_utts['test'].extend(spk2utt[test_spk])
            #     io.log('Added {} spks to test'.format(len(partitions_train[db][fold])))


            # Training and development set
            other_spks = []
            for other_fold in [f for f in range(args.nfolds) if f != fold]:
                other_spks.extend(partitions_train[db][other_fold])
            rand.shuffle(other_spks)
            dev_size = max(1, int(round(args.devfrac * len(other_spks))))
            # Make sure dev set is smaller than training set
            common.CHK_LT(dev_size, len(other_spks) - dev_size)
            for dev_spk in other_spks[:dev_size]:
                set_utts['dev'].extend(spk2utt[dev_spk])
            io.log('Added {} spks to dev'.format(dev_size))
            for train_spk in other_spks[dev_size:]:
                set_utts['train'].extend(spk2utt[train_spk])
            io.log('Added {} spks to train'.format(len(other_spks) - dev_size))


        # Ouput to disk
        for sname in set_utts.keys():
            fname = os.path.join(set_dirs[sname], 'utts')
            utts = set_utts[sname]
            io.log('** Writing {} utts for {} to {}'.format(
                len(utts), sname, fname
            ))
            io.write_lines(fname, utts)

if __name__ == '__main__':
    main()
