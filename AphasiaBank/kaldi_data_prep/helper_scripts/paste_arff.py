import arff
from collections import OrderedDict

import chaipy.common as common
import chaipy.io as io


def name2row(data):
    n2r = OrderedDict()
    for i in range(len(data['data'])):
        name = data['data'][i][0]
        n2r[name] = i
    return n2r


def main(args):
    arffs = [arff.load(open(a, 'r')) for a in args.arffs]
    io.log('Loaded {} arffs'.format(len(arffs)))
    name2rows = [name2row(a) for a in arffs]
    io.log('Num rows: {}'.format([len(n2r) for n2r in name2rows]))
    keys = [
        k for k in name2rows[0].keys() if all([k in n2r for n2r in name2rows])
    ]
    io.log('Found {} common keys'.format(len(keys)))

    data = {
        'attributes': [('spk', 'STRING')],
        'data': [],
        'description': '',
        'relation': '+'.join([a['relation'] for a in arffs])
    }
    for i in range(len(arffs)):
        attrs = arffs[i]['attributes'][1:]
        if args.add_feat_idx:
            attrs = [('{}_{}'.format(i, a[0]), a[1]) for a in attrs]
        data['attributes'].extend(attrs)
    for name in keys:
        row = [name]
        for a, n2r in zip(arffs, name2rows):
            row.extend(a['data'][n2r[name]][1:])
        data['data'].append(row)
    io.log('Outputting pasted ARFF to stdout...')
    print arff.dumps(data)


if __name__ == '__main__':
    desc = 'Paste ARFFs. Output to stdout.'
    parser = common.init_argparse(desc)
    parser.add_argument('arffs', nargs='+', help='ARFFs to paste')
    parser.add_argument('--add-feat-idx', action='store_true',
                        help='Add index to distinguish features')
    args = parser.parse_args()
    # Run program
    main(args)

