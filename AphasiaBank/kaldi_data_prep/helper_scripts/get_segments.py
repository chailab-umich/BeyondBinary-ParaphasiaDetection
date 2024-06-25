import sys
import os
import re
from collections import Counter

import chaipy.common as common
import chaipy.io as io
from chaipy.praat import TextGrid

# from clean_transcripts import clean
# from clean_transcripts_attempted import clean
# from clean_transcript_paraphsia import clean # ASR
# from clean_transcript_Scripts import clean # ASR
# from clean_transcripts_og import clean # phoneme_target
from clean_transcript_best_word import clean # best attempt for phonological transcriptions

# ROOT_DIR = "/home/ducle/Dropbox/UMich/Research/Aphasia_Bank"
ROOT_DIR = "/z/mkperez/AphasiaBank/Duc_kaldi_prep"
EXCLUDED = ['kempler', 'garrett']
## AphasiaBank
# LEXICON = os.path.join(ROOT_DIR, 'templates', 'lexicon.txt')
## Librispeech
LEXICON = os.path.join('/z/public/kaldi/egs/librispeech/s5/data/local/dict_nosp', 'lexicon.txt')

def check_inv(inv_speech_frames, time):
    for start, end in inv_speech_frames:
        # check if par utt is in a INV utt
        if time > start and time < end:
        # if time > start + 0.1 and time < end - 0.1:
            return True
    return False

def get_inv_speech_frames(inv_intervals):
    inv_speech_intervals = []
    for i in range(len(inv_intervals)):
        if inv_intervals[i].text != "":
            start, end = inv_intervals[i].get_times(unit='s')
            inv_speech_intervals.append((start, end))
    return inv_speech_intervals

def main():
    desc = 'Extract segments and their transcripts from Praat TextGrid'
    parser = common.init_argparse(desc)
    parser.add_argument('wav_scp', help='Kaldi wav scp file')
    parser.add_argument('segments_fname', help='File to write segments to')
    parser.add_argument('text_fname', help='File to write transcripts to')
    parser.add_argument('oov_fname', help='File to write ASCII OOVs to. ' + \
                        'Will write two files, one with .list suffix ' + \
                        'containing list of words, one with .count suffix ' + \
                        'containing the number of occurrences of OOVs. ' + \
                        'Words will be written in decreasing occurrences.')
    parser.add_argument('unibet_fname', help='File to write pronunciation ' + \
                        'dictionary of unibet words to. Will write three ' + \
                        'files, two with .list and .count suffix similar ' + \
                        'to above, and one .dict containing pronunciations.')
    parser.add_argument('wlabel_fname', help='File to write word labels to. ' + \
                        'These labels will match the words in transcripts.')
    parser.add_argument('--tier', default='PAR [main]',
                        help='Name of TextGrid tier to read from')
    parser.add_argument('--lexicon', default=LEXICON,
                        help='Path to lexicon, used for checking OOV words')
    parser.add_argument('--excluded', nargs='*', default=EXCLUDED,
                        help='Exclude utterances containing any of these strings')
    args = parser.parse_args()

    wav_scp = io.dict_read(args.wav_scp, ordered=True)
    io.log('Loaded {} mappings from {}'.format(len(wav_scp), args.wav_scp))
    lexicon = set(io.dict_read(args.lexicon).keys())
    io.log('Loaded lexicon containing {} words'.format(len(lexicon)))
    io.log('Will ignore utterances containing any of: {}'.format(args.excluded))
    oov_dict = {}

    fsegments = open(args.segments_fname, 'w')
    ftext = open(args.text_fname, 'w')
    ftext_raw = open('{}.raw'.format(args.text_fname), 'w')
    fwlabel = open(args.wlabel_fname, 'w')
    totwrote, totskipped = 0, []
    for utt in wav_scp:
        if any([(exc in utt) for exc in args.excluded]):
            io.log('{} matches exclusion crieria, skipping'.format(utt))
            continue
        textgrid_fname = wav_scp[utt].replace('.wav', '.c2praat.textGrid')
        textgrid = TextGrid.from_file(textgrid_fname)
        intervals = textgrid.items[args.tier].intervals

        inv_speech_frames=[]
        if 'INV [main]' in textgrid.items:
            inv_intervals = textgrid.items['INV [main]'].intervals
            inv_speech_frames = get_inv_speech_frames(inv_intervals)
        # if utt == "tap09a":
        #     print(inv_speech_frames)
        #     exit()

        io.log('Loaded {} intervals from tier {} of {}'.format(
            len(intervals), args.tier, textgrid_fname
        ))
        wrote, skipped = 0, []
        for i in range(len(intervals)):
            io.log('load {}-{}'.format(utt,i))
            segment = '{}-{}'.format(utt, i)  # Same interval ID as Praat's
            start, end = intervals[i].get_times(unit='s')
            cleaned_text, wrd_labels = \
                    clean(intervals[i].text, segment, lexicon, oov_dict)
            if len(cleaned_text) == 0:
                skipped.append('Text is empty after cleaning')
            elif cleaned_text[0] is None:
                skipped.append(cleaned_text[1])
            elif check_inv(inv_speech_frames, start) or check_inv(inv_speech_frames, end):
                continue
            else:
                fsegments.write('{} {} {} {}\n'.format(segment, utt, start, end))
                ftext.write('{} {}\n'.format(segment, ' '.join(cleaned_text)))
                ftext_raw.write('{} {}\n'.format(segment, intervals[i].text))
                wlabel_target = []
                
                
                for wlabel in wrd_labels:
                    spoken = ' '.join(cleaned_text[int(wlabel[0]):int(wlabel[1])])

                    # target = wlabel[2].split("/")[-1]
                    target = wlabel[2].split("/")[0]
                    wlabel_target.append(target)

                # write wlabels
                fwlabel.write("{} {}\n".format(segment, " ".join(wlabel_target)))


                wrote += 1
        io.log('Wrote {} segments, skipped {}'.format(wrote, len(skipped)))
        summarize_skipped(skipped)
        totwrote += wrote
        totskipped.extend(skipped)
    io.log('TOTAL: Wrote {} segments, skipped {}'.format(totwrote, len(totskipped)))
    summarize_skipped(totskipped)
    fsegments.close()
    ftext.close()
    ftext_raw.close()
    fwlabel.close()

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

    if 'UNIBET' in oov_dict:
        funibet_dict = open('{}.dict'.format(args.unibet_fname), 'w')
        funibet_list = open('{}.list'.format(args.unibet_fname), 'w')
        funibet_count = open('{}.count'.format(args.unibet_fname), 'w')
        # Write .dict
        unibets = oov_dict['UNIBET']
        for wrd in sorted(unibets.keys()):
            for pron in unibets[wrd]:
                funibet_dict.write('{}\t{}\n'.format(wrd, pron))
        # Write .list and .count
        unibets_cnt = common.make_reverse_index(oov_dict['UNIBET_CNT'])
        for cnt in reversed(sorted(unibets_cnt.keys())):
            for wrd in unibets_cnt[cnt]:
                funibet_list.write('{}\n'.format(wrd))
                funibet_count.write('{}\n'.format(cnt))
        funibet_dict.close()
        funibet_list.close()
        funibet_count.close()


def summarize_skipped(skipped):
    if len(skipped) == 0:
        return
    skipped_cnt = Counter(skipped)
    io.log('Reasons for skipping:')
    for reason in sorted(skipped_cnt.keys()):
        io.log('\t{}\t{}'.format(skipped_cnt[reason], reason))


if __name__ == '__main__':
    main()
