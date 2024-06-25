import numpy as np

import chaipy.common as common
import chaipy.io as io

def main():
    desc = 'Merge dicts output by clean_transcripts.py. Hard to explain!'
    parser = common.init_argparse(desc)
    parser.add_argument('in_fnames', nargs='+', help='Input files')
    parser.add_argument('out_fname', help='Output file')
    args = parser.parse_args()

    total = {}
    for f in args.in_fnames:
        with open('{}.list'.format(f), 'r') as fin:
            fl = [l.strip() for l in fin]
        fc = np.loadtxt('{}.count'.format(f), dtype=np.int)
        common.CHK_EQ(len(fl), len(fc))
        io.log('Loaded {} items from {}.list,count'.format(len(fl), f))
        for l, c in zip(fl, fc):
            if l not in total:
                total[l] = 0
            total[l] += c
    total = common.make_reverse_index(total)

    fout_list = open('{}.list'.format(args.out_fname), 'w')
    fout_count = open('{}.count'.format(args.out_fname), 'w')
    wrote = 0
    for c in reversed(sorted(total.keys())):
        for l in total[c]:
            fout_list.write('{}\n'.format(l))
            fout_count.write('{}\n'.format(c))
            wrote += 1
    fout_list.close()
    fout_count.close()
    io.log('Wrote {} lines to {}.list,count'.format(wrote, args.out_fname))

if __name__ == '__main__':
    main()
