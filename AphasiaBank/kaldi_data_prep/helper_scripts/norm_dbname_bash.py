import chaipy.common as common
import sys
DB_MAP = {
    'acwt':'ACWT', 'adler':'Adler', 'bu':'BU', 'cmu':'CMU', 'elman':'Elman',
    'fridriksson':'Fridriksson', 'garrett':'Garrett', 'kansas':'Kansas',
    'kempler':'Kempler', 'kurland':'Kurland', 'msu':'MSU', 'scale':'SCALE',
    'star':'STAR', 'tap':'TAP', 'tcu':'TCU', 'thompson':'Thompson',
    'tucson':'Tucson', 'whiteside':'Whiteside', 'williamson':'Williamson',
    'wozniak':'Wozniak', 'wright':'Wright', 'capilouto':'Capilouto', 'msuc':'MSU',
    'richardson':'Richardson', 'umd':'UMD', 'aprocsa':'APROCSA', 'ucl':'UCL',
    'unh':'UNH', 'mba':'UMD', 'mma':'UMD', 
}

def main():
    # desc = 'Print normalized db name that matches folder name'
    # parser = common.init_argparse(desc)
    # parser.add_argument('db', help='Unnormalized db name')
    # args = parser.parse_args()
    for line in sys.stdin:
        # sys.stderr.write("DEBUG: got line: " + line)
        # sys.stdout.write(line)
        # print DB_MAP[line.lower()]
        sys.stdout.write("{}\n".format(DB_MAP[line.lower().strip()]))


if __name__ == '__main__':
    main()
