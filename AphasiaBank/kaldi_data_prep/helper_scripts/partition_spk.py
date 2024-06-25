import os
from collections import OrderedDict
import random

import chaipy.common as common
import chaipy.io as io


def main(args):
    spk2utt = io.dict_read(args.spk2utt, ordered=True)
    io.log('Loaded {} records from {}'.format(len(spk2utt), args.spk2utt))

    rand = random.Random(args.seed)
    # Shuffle utts for each spk
    for spk in spk2utt:
        if isinstance(spk2utt[spk], list):
            rand.shuffle(spk2utt[spk])

    # Output stuff
    for fold, spk in zip(range(len(spk2utt)), spk2utt):
        fold_dir = os.path.join(args.output_dir, 'Fold_{}'.format(fold + 1))
        io.log('Fold {} ({}), outputting to {}'.format(fold + 1, spk, fold_dir))
        set_utts, set_dirs = OrderedDict(), OrderedDict()
        for sname in ['train', 'dev', 'test']:
            set_utts[sname] = []
            set_dirs[sname] = os.path.join(fold_dir, sname)
            if not os.path.exists(set_dirs[sname]):
                os.makedirs(set_dirs[sname])
        # Accumulate utts for each set
        for curr_spk in spk2utt:
            print(curr_spk)
            # if isinstance(spk2utt[curr_spk], list):
            if curr_spk == spk:
                set_utts['test'].extend(spk2utt[curr_spk])
                continue
            dev_size = max(1, int(round(args.devfrac * len(spk2utt[curr_spk]))))
            # Make sure dev set is smaller than training set
            # if dev_size < 
            # common.CHK_LT(dev_size, len(spk2utt[curr_spk]) - dev_size)
            set_utts['dev'].extend(spk2utt[curr_spk][:dev_size])
            set_utts['train'].extend(spk2utt[curr_spk][dev_size:])
            io.log('{}: {} dev utts, {} train utts'.format(
                curr_spk, dev_size, len(spk2utt[curr_spk]) - dev_size
            ))
        # Ouput to disk
        for sname in set_utts.keys():
            fname = os.path.join(set_dirs[sname], 'utts')
            utts = sorted(set_utts[sname])
            io.log('** Writing {} utts for {} to {}'.format(
                len(utts), sname, fname
            ))
            io.write_lines(fname, utts)


if __name__ == '__main__':
    desc = 'Split data into N leave-one-speaker-out folds, each with a ' + \
            'train, dev, and test set. Each test set corresponds to one ' + \
            'speaker. A fraction of utts from each speaker in the training ' + \
            'set is withheld to form the dev set. This script will output ' + \
            'the list of utt names. trim.py can be used to trim the data.'
    parser = common.init_argparse(desc)
    parser.add_argument('spk2utt', help='Kaldi spk2utt')
    parser.add_argument('output_dir', help='Where to output')
    parser.add_argument('--devfrac', type=float, default=0.1,
                        help='Amount of training data to withhold for dev set')
    parser.add_argument('--seed', type=int, default=883, help='Random seed')
    args = parser.parse_args()
    # Run program
    main(args)
