import chaipy.common as common
import chaipy.io as io

def main():
    desc = 'Combine spk props. Only include speakers who are present in ' + \
           'all props. Output to stdout.'
    parser = common.init_argparse(desc)
    parser.add_argument('spk2props', nargs='+', help='Spk props to combine')
    parser.add_argument('--sep', default='_', help='Field separator')
    args = parser.parse_args()
    
    spk2props = []
    for i in args.spk2props:
        spk2props.append(io.dict_read(i, ordered=True))
    io.log('Loaded {} spk props'.format(len(spk2props)))
    
    spks = []
    for spk in spk2props[0]:
        if all([spk in x for x in spk2props[1:]]):
            spks.append(spk)
    io.log('Found {} spks in intersection'.format(len(spks)))
    
    for spk in spks:
        print '{} {}'.format(spk, args.sep.join([x[spk] for x in spk2props]))

if __name__ == '__main__':
    main()