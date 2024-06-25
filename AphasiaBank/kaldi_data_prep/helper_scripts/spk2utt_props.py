import chaipy.common as common
import chaipy.io as io

def main():
    desc = 'Convert from speaker property to utt property. Output to stdout.'
    parser = common.init_argparse(desc)
    parser.add_argument('spk_prop', help='Mapping from spk to property.')
    parser.add_argument('utt2spk', help='Kaldi utt2spk mapping.')
    parser.add_argument('--placeholder', default='NULL',
                        help='Value to use for spks not present in `spk_prop`')
    args = parser.parse_args()

    spk2prop = io.dict_read(args.spk_prop, ordered=True)
    utt2spk = io.dict_read(args.utt2spk, ordered=True)
    spk2utt = common.make_reverse_index(utt2spk, ordered=True)

    wrote = 0
    for spk in spk2utt.keys():
        prop = spk2prop[spk] if spk in spk2prop else args.placeholder
        for utt in spk2utt[spk]:
            print '{} {}'.format(utt, prop)
            wrote += 1
    io.log('Wrote {} utt props for {} spks'.format(wrote, len(spk2utt)))

if __name__ == '__main__':
    main()
