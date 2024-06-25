import arff

import chaipy.common as common
import chaipy.io as io


def main(args):
    data = {
        'attributes': [],
        'data': [],
        'description': '',
        'relation': ''
    }
    for i in range(len(args.arffs)):
        io.log('Loading {}'.format(args.arffs[i]))
        a = arff.load(open(args.arffs[i], 'r'))
        if i == 0:
            data['attributes'] = a['attributes']
            data['description'] = a['description']
            data['relation'] = a['relation']
        else:
            common.CHK_EQ(len(data['attributes']), len(a['attributes']))
        data['data'].extend(a['data'])
        del a
    io.log('Outputting concatenated ARFF to stdout...')
    print arff.dumps(data)


if __name__ == '__main__':
    desc = 'Concatenate ARFFs. Output to stdout.'
    parser = common.init_argparse(desc)
    parser.add_argument('arffs', nargs='+', help='ARFFs to concatenate')
    args = parser.parse_args()
    # Run program
    main(args)

