from collections import OrderedDict

import chaipy.common as common
import chaipy.io as io


def load_spk2tasks(owav_fname):
    spk2wav = io.dict_read(owav_fname, ordered=True)
    spk2tasks = OrderedDict()
    for spk in spk2wav:
        task2time_fname = spk2wav[spk].replace('.wav', '.task2time')
        task2time = io.dict_read(task2time_fname, ordered=True, fn=float)
        spk2tasks[spk] = [(k, task2time[k]) for k in task2time]
    return spk2tasks


def find_group(tasks, start):
    for group_name, group_start in reversed(tasks):
        if start >= group_start:
            return group_name
    io.log('WARNING: cannot find group for start {} in {}'.format(start, tasks))
    # Use first group as default
    return tasks[0][0]


def main(args):
    segments = io.dict_read(args.segments, ordered=True, lst=True)
    io.log('Loaded {} from segments'.format(len(segments)))
    for k in segments:
        segments[k][1] = float(segments[k][1])
        segments[k][2] = float(segments[k][2])
    spk2tasks = load_spk2tasks(args.owav)
    io.log('Loaded {} spk2tasks'.format(len(spk2tasks)))
    alias = None
    if args.alias is not None:
        alias = io.dict_read(args.alias)
        io.log('Loaded {} aliases'.format(len(alias)))

    for utt in segments:
        spk, start, _ = segments[utt]
        group = find_group(spk2tasks[spk], start)
        if alias is not None:
            group = alias[group]
        print '{} {}'.format(utt, group)


if __name__ == '__main__':
    desc = 'Map from utt to group name. Output to stdout.'
    parser = common.init_argparse(desc)
    parser.add_argument('segments', help='Segment file')
    parser.add_argument('owav', help='Original wav mapping')
    parser.add_argument('--alias', help='Alias for group names')
    args = parser.parse_args()
    # Run program
    main(args)

