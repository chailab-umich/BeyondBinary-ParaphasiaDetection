import chaipy.common as common
import chaipy.io as io


def main(args):
    group = None
    start = None
    lines = []
    with open(args.cha_fname, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('@G'):
                io.log('group line: {}'.format(line))
                assert start is None
                group = line.replace('@G:', '').replace('.', '').strip().lower()
                group = '_'.join(group.split())
            if group is not None and '\x15' in line:
                io.log('start line: {}'.format(line))
                start = float(line.split()[-1].replace('\x15', '').split('_')[0])
                start = start / 1000.0
                lines.append('{} {}'.format(group, start))
                group = None
                start = None
    # assert group is None and start is None, '{}'.format(lines)
    assert len(lines) > 0
    # Output
    for l in lines:
        print l


if __name__ == '__main__':
    desc = 'Output task and starting time. Output to stdout.'
    parser = common.init_argparse(desc)
    parser.add_argument('cha_fname', help='CHAT transcription file')
    args = parser.parse_args()
    main(args)

