import os
from collections import Counter

import chaipy.common as common
import chaipy.io as io
from chaipy.praat import TextGrid

from get_segments import LEXICON
from clean_transcripts import clean
from get_segments import summarize_skipped


def main(args):
    text_grid_fnames = io.read_lines(args.text_grids)
    io.log('Loaded {} TextGrids'.format(len(text_grid_fnames)))
    lexicon = set(io.dict_read(args.lexicon).keys())
    io.log('Loaded lexicon containing {} words'.format(len(lexicon)))
    oov_dict = {}

    ftext = open(args.text_fname, 'w')
    ftext_raw = open('{}.raw'.format(args.text_fname), 'w')
    totwrote, totskipped = 0, []
    for fname in text_grid_fnames:
        io.log('Processing {}'.format(fname))
        name = os.path.basename(fname).split('.')[0]
        text_grid = TextGrid.from_file(fname)
        tier = None
        for t in args.tiers:
            if t in text_grid.items:
                tier = t
                break
        assert tier is not None, \
            'Cannot find {} in {}'.format(args.tiers, text_grid.items.keys())
        intervals = text_grid.items[tier].intervals
        wrote, skipped = 0, []
        for i in range(len(intervals)):
            segment = '{}-{}'.format(name, i)
            cleaned_text, _ = \
                clean(intervals[i].text, segment, lexicon, oov_dict)
            if len(cleaned_text) == 0:
                skipped.append('Text is empty after cleaning')
            elif cleaned_text[0] is None:
                skipped.append(cleaned_text[1])
            else:
                ftext.write('{} {}\n'.format(segment, ' '.join(cleaned_text)))
                ftext_raw.write('{} {}\n'.format(segment, intervals[i].text))
                wrote += 1
        io.log('Wrote {} segments, skipped {}'.format(wrote, len(skipped)))
        summarize_skipped(skipped)
        totwrote += wrote
        totskipped.extend(skipped)
    io.log('TOTAL: Wrote {} segments, skipped {}'.format(totwrote, len(totskipped)))
    summarize_skipped(totskipped)
    ftext.close()
    ftext_raw.close()

    if 'ASCII' in oov_dict:
        foov_list = open('{}.list'.format(args.oov_fname), 'w')
        foov_count = open('{}.count'.format(args.oov_fname), 'w')
        oovs = common.make_reverse_index(Counter(oov_dict['ASCII']))
        for cnt in reversed(sorted(oovs.keys())):
            for wrd in oovs[cnt]:
                foov_list.write('{}\n'.format(wrd))
                foov_count.write('{}\n'.format(cnt))
        foov_list.close()
        foov_count.close()

    assert 'UNIBET' not in oov_dict, \
            'Contains unibet: {}'.format(oov_dict['UNIBET'])


if __name__ == '__main__':
    desc = 'Convert a list of TextGrid to Kaldi text file.'
    parser = common.init_argparse(desc)
    parser.add_argument('text_grids', help='List of TextGrid files')
    parser.add_argument('text_fname', help='File to write transcripts to')
    parser.add_argument('oov_fname', help='File to write ASCII OOVs to')
    parser.add_argument('--lexicon', default=LEXICON,
                        help='Path to lexicon, used for checking OOV words')
    parser.add_argument('--tiers', nargs='+',
                        default=['PAR [main]', 'INV [main]'],
                        help='Name of TextGrid tiers to read from')
    args = parser.parse_args()
    # Run program
    main(args)

