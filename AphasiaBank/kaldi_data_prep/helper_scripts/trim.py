import chaipy.common as common
import chaipy.io as io

def main():
    desc = 'Trim content of a file given a set of keys. Assume that the ' + \
           'first token in the file (separated by whitespace) is the key. ' + \
           'Write to stdout.'
    parser = common.init_argparse(desc)
    parser.add_argument('keys_fname', help='File containing list of keys')
    parser.add_argument('in_fname', help='Input file')
    parser.add_argument('--exclude', action='store_true',
                        help='Write out entries NOT in the key list.')
    parser.add_argument('--skip-empty', action='store_true',
                        help='Skip empty entries')
    args = parser.parse_args()

    with open(args.keys_fname, 'r') as f:
        keys = set([line.strip() for line in f])
    io.log('Loaded {} keys from {}'.format(len(keys), args.keys_fname))

    total, skipped = 0, 0
    with open(args.in_fname, 'r') as fin:
        for line in fin:
            total += 1
            ary = line.strip().split()
            key = ary[0]
            
            if (args.exclude and key in keys) or \
                    (not args.exclude and key not in keys) or \
                    (args.skip_empty and len(ary) == 1):
                skipped += 1
                continue
            print line.strip()
    io.log('Wrote {} lines, skipped {}'.format(total - skipped, skipped))

if __name__ == '__main__':
    main()
