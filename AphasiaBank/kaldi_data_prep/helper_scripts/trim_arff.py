import arff

import chaipy.common as common
import chaipy.io as io

def main(args):
    keys = io.dict_read(args.keys_fname)
    io.log('Loaded {} keys'.format(len(keys)))
    data = arff.load(open(args.arff_fname, 'r'))
    io.log('Loaded ARFF with {} rows'.format(len(data['data'])))

    data['data'] = [r for r in data['data'] if r[0] in keys]
    io.log('Outputting trimmed ARFF with {} rows'.format(len(data['data'])))
    print arff.dumps(data)


if __name__ == '__main__':
    desc = 'Trim ARFF given a set of keys. Write to stdout.'
    parser = common.init_argparse(desc)
    parser.add_argument('keys_fname', help='File containing list of keys')
    parser.add_argument('arff_fname', help='Input file')
    args = parser.parse_args()
    # Run program
    main(args)
