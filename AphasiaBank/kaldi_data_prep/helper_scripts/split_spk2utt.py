from collections import OrderedDict

import chaipy.common as common
import chaipy.io as io


def main(args):
    spk2utt = io.dict_read(args.spk2utt, lst=True, ordered=True)
    io.log('Loaded {} spk2utt'.format(len(spk2utt)))
    utt2group = io.dict_read(args.utt2group, ordered=True)
    group2utt = common.make_reverse_index(utt2group, ordered=True)
    io.log('Loaded {} utt2group with {} groups'.format(
        len(utt2group), len(group2utt)
    ))

    for group in group2utt:
        utts = set(group2utt[group])
        group_spk2utt = OrderedDict()
        for spk in spk2utt:
            spk_utts = [u for u in spk2utt[spk] if u in utts]
            if len(spk_utts) > 0:
                group_spk2utt[spk] = spk_utts
        if len(group_spk2utt) > 0:
            io.log('Outputting for group {} with {} spks'.format(
                group, len(group_spk2utt)
            ))
            io.dict_write(
                '{}.{}'.format(args.spk2utt, group), group_spk2utt,
                fn=lambda x: ' '.join(x)
            )


if __name__ == '__main__':
    desc = 'Split spk2utt according to utt2group mapping.'
    parser = common.init_argparse(desc)
    parser.add_argument('spk2utt', help='Mapping from spk to utt')
    parser.add_argument('utt2group', help='Mapping from utt to group')
    args = parser.parse_args()
    # Run program
    main(args)

