import re
import hashlib
import itertools

import chaipy.common as common
import chaipy.io as io

'''

Use lexicgraphic word (best of ability)

use intended words if the spoken word is not a real-word
    -pro: training ctc is better (accurate phonetics for what was said)
    -con: utilizes fictional words

If not a real word, remove this utterance


Ignore neologisms/paraphasias without intended word
Ignore [+ exc]
'''

def __preprocess(line):
    """ Perform some preprocessing before splitting into tokens.
    """
    if len(line) == 0 or line[0] is None:
        return line
    # The explanation blocks [...] are sometimes adjacent to non-spaces,
    # we need to separate them out.
    line = re.sub(r'\]([^ ])', '] \g<1>', line)
    line = re.sub(r'([^ ])\[', '\g<1> [', line)
    # A few utterances have normal characters right before '<',
    # we need to separate them out.
    line = re.sub(r'([A-Za-z0-9])<', '\g<1> <', line)
    # A few utterances have space before ']', we need to scrap the space
    line = re.sub(r'\s+\]', ']', line)
    # There are very few occurrences of '<<', need to split them apart
    line = line.replace('<<', '< <')
    # Very few pause blocks (..), (...), etc... are adjacent to
    # non-spaces, need to split them apart
    line = re.sub(r'(\(\.+\))([^ ])', '\g<1> \g<2>', line)  # Leading pause
    line = re.sub(r'([^ ])(\(\.+\))', '\g<1> \g<2>', line)  # Trailing pause
    return line.split()

def clean(text, utt, lexicon, oov_dict):
    """ Clean up transcript. Return a 2-dim tuple where the first element is
    the cleaned transcript and the second element is the word labels.
    For the first element, if the transcript is invalid and this utterance
    should be omitted from training, return a 2-dim tuple where the first item
    is None and the second item is a string indicating reason for removal.
    """
    # print("initial: {}".format(text))
    text = __can_skip(text)
    text = __preprocess(text)
    io.log("pre remove term: {}".format(text))
    text = __remove_terminators(text, utt)
    text = __separate_blocks(text, utt)
    text = __group_explanations(text, utt)
    text = __remove_terminators(text, utt)  # May get some new tokens now, so
                                            # remove terminators again.
    io.log("pre unused tok: {}".format(text))
    text = __remove_unused_tokens(text, utt)
    text = __handle_partial_omissions(text, utt)
    text = __clean_word_tokens(text, utt)
    io.log("pre repetitions tok: {}".format(text))
    text = __proc_repetitions(text, utt)
    io.log("pre simple events tok: {}".format(text))
    text = __proc_simple_events(text, utt)
    io.log("pre fragment: {}".format(text))
    text = __proc_fragments(text, utt)
    io.log("pre compound: {}".format(text))
    text = __proc_compound_words(text, utt)
    io.log("pre special forms: {}".format(text))
    text = __proc_special_forms(text, utt)
    io.log("pre finalize: {}".format(text))
    # text = __proc_oovs(text, utt, lexicon, oov_dict)
    wlabels = __get_word_labels(text, utt)
    io.log("word_labels: {}".format(wlabels))
    ftext = __finalize(text, utt)
    io.log("finalize: {}".format(ftext))
    return (ftext, wlabels)

def __remove_terminators(text, utt):
    """ Remove terminator tokens and symbols.
    """
    if len(text) == 0 or text[0] is None:
        return text
    new_text = []
    for w in text:
        if w in [',', ';', '.', '?', '!', ':']: # Standard terminators
            continue
        elif w.startswith('+'): # Special utterance terminators
            continue
        elif re.match(r'^\[/+\]$', w):  # Repetition, retracing, reformulation
            continue
        # elif re.match(r'^\(\.+\)$', w): # Pauses
        elif re.match(r'^\([0-9]*\.+[0-9]*\)$', w): # Pauses
            continue
        elif w.decode('utf-8') in [u'\u2021', u'\u201e']:   # Satellite markers
            continue
        else:
            w = w.replace(',', '')  # Commas inside words can be safely ignored
            w = w.replace('\xe2\x80\x9c', '')   # Remove open quotes
            w = w.replace('\xe2\x80\x9d', '')   # Remove end quotes
            w = w.replace('\xe2\x88\xac', '')   # Remove whisper boundaries
            new_text.append(w)
    return new_text

def __separate_blocks(text, utt):
    """ For blocks of text like <he is playing>, the angular brackets need
    to be separated so we can easily retrieve the whole block if needed.
    """
    if len(text) == 0 or text[0] is None:
        return text
    new_text = []
    for w in text:
        # Check for 1-letter word (stand-alone brackets are allowed)
        if len(w) == 1:
            new_text.append(w)
        # If start of block, separate the opening bracket
        elif w.startswith('<') and not w.endswith('>'):
            new_text.append(w[0])
            new_text.append(w[1:])
        # If end of block, separate the closing bracket
        elif w.endswith('>') and not w.startswith('<'):
            new_text.append(w[:len(w)-1])
            new_text.append(w[len(w)-1])
        # If one whole block, separate both brackets
        elif w.startswith('<') and w.endswith('>'):
            common.CHK_NEQ(w, '<>')
            io.log('**WARN** {} - rare stand-alone block: {}'.format(utt, w))
            new_text.append(w[0])
            new_text.append(w[1:len(w)-1])
            new_text.append(w[len(w)-1])
        else:
            new_text.append(w)
    # Make sure there are matching brackets. Unlike in explanations, in word
    # blocks opening brackets don't have to match immediately.
    opening = common.find_all(new_text, '<')
    closing = common.find_all(new_text, '>')
    if len(opening) != len(closing):
        raise ValueError('{} - has {} < but {} >: {}'.format(
            utt, len(opening), len(closing), new_text
        ))
    elif len(opening) != 0:
        used = []
        for c in closing:
            matched = False
            for o in [x for x in reversed(opening) if x < c and x not in used]:
                matched = True
                used.append(o)
                break
            if not matched:
                raise ValueError('{} - unmatched >: {}'.format(utt, new_text[:c+1]))
    return new_text

def __group_explanations(text, utt):
    """ Explanation blocks like [* n:k] should be grouped together.
    """
    if len(text) == 0 or text[0] is None:
        return text
    start = -1
    new_text = []
    for i in range(len(text)):
        w = text[i]
        if w == '[' or w == ']':
            raise ValueError('{} - stand-alone [ or ]: {}'.format(utt, text[:i+1]))
        elif start == -1:       # Look for opening bracket
            if w.startswith('[') and w.endswith(']'):
                new_text.append(w)
            elif w.endswith(']'):
                raise ValueError('{} - unmatched ]: {}'.format(utt, text[:i+1]))
            elif w.startswith('['):
                start = i
            else:
                new_text.append(w)
        else:                   # Have to look for closing bracket now
            if w.startswith('['):
                raise ValueError('{} - consecutive [: {}'.format(utt, text[start:i+1]))
            elif w.endswith(']'):
                new_text.append(' '.join(text[start:i+1]))
                start = -1
    # Check that all beginning brackets are accounted for
    if start != -1:
        raise ValueError('{} - no closing ]: {}'.format(utt, text[start:]))
    return new_text

def __remove_unused_tokens(text, utt):
    """ Remove tokens that will not be used whatsoever. If we encounter a token
    that indicates the utterance should be omitted from training, return None.
    """
    if len(text) == 0 or text[0] is None:
        return text
    new_text = []
    for w in text:
        if re.match(r'\[\?\]', w):                  # Unclear word
            return (None, 'Has unclear word')
        elif w == 'xxx':                            # Unintelligible word
            return (None, 'Has unintelligible word')
        elif re.match(r'\[[<>][0-9]*\s*\]', w):     # Overlapping speech
            return (None, 'Has overlapping speech')
        elif re.match(r'\[=.*\]', w):   # Paralinguistic material & explanation
            return (None, 'Has paralinguistic material')
        elif re.match(r'\[\^.*\]', w):              # Complex local event
            continue
        elif w.startswith('0'):                     # Omitted words
            continue
        elif re.match(r'\[!+\]', w):                # Stressing
            continue
        elif re.match(r'\[%.*\]', w):               # Comment & dependent tier
            continue
        elif w == '[+ exc]':
            return (None, '[+ exc] tag')
        elif re.match(r'\[[\+\-].*\]', w):          # Post-codes and pre-codes
            continue
        else:
            new_text.append(w)
    return new_text

def __can_skip(text):
    """ Returns (None, reason) if text can be skipped; otherwise return text.
    """
    # Empty text
    if text == '':
        return (None, 'Empty text')
    # Fully unintelligible speech. If 'xxx' occurs with other words, we can
    # train it as garbage speech, or spoken noise.
    if re.match('^xxx\s*\.', text):
        return (None, 'Fully unintelligible speech')
    # Contains phonological coding elsewhere - just skip if it exists
    if 'yyy' in text:
        return (None, 'Has phonological coding elsewhere')
    # Untranscribed material
    if 'www' in text:
        return (None, 'Untranscribed material')
    # Contains only "0" (i.e., no speech, but has accompanying action)
    if re.match('^0\s*\.', text):
        return (None, 'No speech but with accompanying action')
    return text

def __handle_partial_omissions(text, utt):
    """ Partial omission is tricky. We make it an entire function for easy
    modification later on. Right now we just treat partial omissions as
    non-existent and hopefully they can be modeled with triphones.
    """
    if len(text) == 0 or text[0] is None:
        return text
    new_text = []
    for w in text:
        if re.match(r'.*\([A-Za-z\']+\).*', w):
            w = re.sub(r'\(([A-Za-z\']+)\)', '\g<1>', w)
        new_text.append(w)
    return new_text

def __clean_word_tokens(text, utt):
    """ Remove special characters within word tokens.
    """
    if len(text) == 0 or text[0] is None:
        return text
    new_text = []
    for w in text:
        if __is_word(w):
            w = re.sub(r'\$.*$', '', w)                     # Remove part-of-speech
            # Separate special forms, we'll deal with this later
            ary = w.split('@')
            ary[0] = ary[0].replace(':', '')                # Syllable elongation
            ary[0] = ary[0].replace('^', '')                # Inter-syllable pause
            if not common.is_ascii(ary[0]):
                ary[0] = ary[0].replace('\xe2\x86\x91', '') # Pitch up
                ary[0] = ary[0].replace('\xe2\x86\x93', '') # Pitch down
                ary[0] = ary[0].replace('\xcb\x90', '')     # Unicode elongation ':'
                ary[0] = ary[0].replace('\xc3\xa9', 'e')    # Script 'e' in 'fiance'
                ary[0] = ary[0].replace('\xc4\xa7', 'h')    # Similar to 'h' in 'house'
                ary[0] = ary[0].replace('\xc9\x92', 'o')    # Similar to 'o' in 'ipod'
                ary[0] = ary[0].replace('\xc9\xb8', 'f')    # Similar to 'f' in 'fringe'
                ary[0] = ary[0].replace('\xca\x94', '')     # Glottal stop
            ary[0] = ary[0].upper()                         # Conver to upper case
            w = '@'.join(ary)
        new_text.append(w)
    return new_text

def __is_word(token):
    """ Return true if token is a word, false otherwise.
    """
    # We use a very simple approach: if a token begins with a character other
    # than '&', '[', ']', '<', '>', then it's considered a word. This allows
    # us to catch unibets and words beginning with '^' (very few). If the
    # leading character is '<' AND the ending character is '>', this indicates
    # a special word and will still be treated as a word. We don't have to
    # worry about words beginning with numbers. The only one I'm aware of is
    # the leading '0' which denotes omitted word. This case should've been
    # taken care of already when removing unused tokens.
    if token.startswith('<') and token.endswith('>'):
        return True
    elif token[0] not in ['&','[',']', '<', '>']:
        return True
    else:
        return False

def __proc_simple_events(text, utt):
    """ Process simple events (starts with '&='). We can't cover every vocal
    event there is. The ones covered here span most events in this dataset.
    """
    if len(text) == 0 or text[0] is None:
        return text
    EVENT_MAPS = [
        ['<LAU>', ['&=laughs', '&=chuckles']],
        ['<SPN>', ['&=lips_smack', '&=imit', '&=clears', '&=coughs', \
                   '&=groans', '&=lips:smack', '&=sings', '&=cries']],
        ['<BRTH>', ['&=inhales', '&=sighs', '&=sniffs', '&=exhales', '&=gasps']]
    ]
    new_text = []
    for w in text:
        if w.startswith('&='):
            # for token, prefixes in EVENT_MAPS:
            #     if any([w.startswith(prefix) for prefix in prefixes]):
            #         new_text.append(token)
            # # continue
            continue
        else:
            new_text.append(w)
    return new_text

def __proc_fragments(text, utt):
    """ Process fragments (start with '&'). We'll treat all fragments as
    fillers. We'll train them separately from spoken noise.
    """
    if len(text) == 0 or text[0] is None:
        return text
    new_text = []
    for w in text:
        # Insert <FLR> (filler) tag


        if re.match(r'^&\-', w):
            continue
            # new_text.append(w[2:])
        elif re.match(r'^&\+', w):
            continue
            # io.log("pre &+: {}".format(w))
            # w = w[2:]
            # # w=w.replace('\^','')
            # w=w.replace('^','')
            # io.log("post &+: {}".format(w))
            # new_text.append(w)


        # if re.match(r'^&[^ =]', w):
        #     new_text.append('<FLR>')


        # if re.match(r'^&[^ =]', w):
        #     # keep ums
        #     new_text.append(w[1:])
        
        # # Use um/uh or whatever is after '&' as the word
        # elif re.match(r'^&=', w):
        #     # skip points, laughs, etc.
        #     continue

        ## replace attempted word w/ intended word
        # elif

        else:
            new_text.append(w)
    return new_text

def __proc_repetitions(text, utt):
    """ Look for repetition codes and repeat their targets.
    """
    if len(text) == 0 or text[0] is None:
        return text
    def find_rep(text):
        """ Return (index of rep, num reps) or None if no rep is found.
        """
        for i, w in zip(range(len(text)), text):
            matched = re.match(r'\[x ([0-9]+)\]', w)
            if matched:
                return (i, int(matched.group(1)))
        return None
    def find_rep_range(text, rep_idx, utt):
        """ Return the range of items for the repeat token at `rep_idx`. For
        example, a return value of (2, 5) means that items at index 2 through 4
        will be replicated. An error will be raised if a range cannot be found.
        """
        end_bracket = None
        for idx in reversed(range(rep_idx)):
            w = text[idx]
            # In this dataset, explanations ([]) don't go in between <> blocks
            # and repetition code. We'll throw an error if this happens.
            if w == '>':
                if idx != rep_idx - 1:
                    raise ValueError('{} - end bracket > is not'.format(utt) + \
                            ' next to repetition: {}'.format(text[idx:rep_idx+1]))
                end_bracket = idx
            elif w == '<':
                if end_bracket is None:
                    raise ValueError('{} - found start bracket'.format(utt) + \
                            ' w/o end bracket: {}'.format(text[idx:rep_idx+1]))
                else:
                    return (idx, end_bracket + 1)
            # Note that explanations ([]), fragments (&), and
            # simple events (&=) can have repetitions too.
            if w.startswith('['):
                continue
            elif end_bracket is None:
                return (idx, rep_idx)
        raise ValueError('{} - cannot find rep range: {}'.format(utt, text[:rep_idx+1]))
    def repeat(text, start, end, rep_idx, reps):
        """ Repeat items from `start` through `end-1` `reps` time.
        Will also remove the repeat token at `rep_idx`.
        """
        new_text = []
        new_text.extend(text[:start])
        new_text.extend(text[start:end] * reps)
        new_text.extend([text[i] for i in range(end, len(text)) if i != rep_idx])
        return new_text
    new_text = text
    rep = find_rep(new_text)
    while rep is not None:
        rep_idx, num_reps = rep
        start, end = find_rep_range(new_text, rep_idx, utt)
        new_text = repeat(new_text, start, end, rep_idx, num_reps)
        rep = find_rep(new_text)
    return new_text

def __find_errs(text, idx):
    """ Return the indices of error codes for token `text[idx]`.
    If there is no error, return an empty array.
    """
    err_indices = []
    for i in range(idx + 1, len(text)):
        if text[i].startswith('[:'):
            continue
        elif text[i].startswith('[*'):
            err_indices.append(i)
        else:
            break
    return err_indices

def __proc_compound_words(text, utt):
    """ Compound words are linked by '+', '-', or '_'. We need to break them
    apart. The tricky thing is to preserve special form markers after breaking.
    """
    if len(text) == 0 or text[0] is None:
        return text
    new_text = []
    for idx, w in zip(range(len(text)), text):
        
        
        if "@" in w and ((idx+1)==len(text) or (not text[idx+1].startswith('[:'))):
            io.log("skip this utt{} | w: {}".format(text,w))
            # exit()
            return []
        

        # proc replacement word
        elif w.startswith('[:') and ("@" in text[idx-1] or not common.is_ascii(text[idx-1])):
            io.log("ascii: {}".format(common.is_ascii(text[idx-1])))
            ## add intended word
            w = w.split(" ")[-1][:-1]
            new_text.append(w)


        elif __is_word(w) and not w.startswith('<') and \
                any([c in w for c in ['+', '_', '-']]):
            if any([c in w for c in ['@o', '@s', '@b', '@si', '@i']]):
                io.log('**WARN** {} - not breaking {}'.format(utt, w))
                new_text.append(w)
            else:
                err_indices = __find_errs(text, idx)
                if len(err_indices) != 0:
                    io.log('**WARN** {} - compound with errs: {}'.format(
                        utt, text[idx:err_indices[-1]+1]
                    ))
                ary = w.split('@')
                new_words = ary[0].replace('+', ' ')
                new_words = new_words.replace('_', ' ')
                new_words = new_words.replace('-', ' ')
                nw_ary = new_words.split()
                for i, nw in zip(range(len(nw_ary)), nw_ary):
                    new_text.append('@'.join([nw] + ary[1:]))
                    if len(err_indices) != 0 and i < len(nw_ary) - 1:
                        new_text.append('[* ->]')

        else:
            new_text.append(w)
    return new_text

def __proc_special_forms(text, utt):
    """ Special forms (contain '@' at the end) are the trickiest to handle.
    Note that in theory, a word can contain multiple '@'. This doesn't
    occur in this dataset, however, so we don't have to deal with that.
    """
    if len(text) == 0 or text[0] is None:
        return text
    new_text = []
    for idx, w in zip(range(len(text)), text):
        if __is_word(w) and not w.startswith('<'):
            if '@' in w:
                # ignore neologisms, and other phonological dysfluencies
                continue

            # # Neologism, onomatopoeia, language change, babbling, singing
            # if any([i in w for i in ['@n', '@o', '@s', '@b', '@si']]):
            #     # new_text.append('<SPN>')
            #     # sub_list = '|'.join(['@n', '@o', '@s', '@b', '@si'])
            #     # preserve_word = re.sub(sub_list, '', w)
            #     # new_text.append(preserve_word)
            #     # new_text.append(w.split('@')[0])

            # # Metalinguistic use, addition, dialect, unibet: just remove marker
            # # We do this for '@u' too because: (1) some unibets don't have '@u'
            # # due to annotation error and (2) they will get taken care of in
            # # __proc_oovs (unibets can be recognized due to non-ASCII chars).
            # elif any([i in w for i in ['@q', '@a', '@d', '@u']]):
            #     new_text.append(w.split('@')[0])
            # # Letters and multiple letters (we treat them the same way in case
            # # there are annotation errors). Split into individual letters.
            # elif any([i in w for i in ['@l', '@k']]):
            #     err_indices = __find_errs(text, idx)
            #     if len(err_indices) != 0:
            #         io.log('**WARN** {} - compound with errs: {}'.format(
            #             utt, text[idx:err_indices[-1]+1]
            #         ))
            #     c_ary = w.split('@')[0]
            #     for i, c in zip(range(len(c_ary)), c_ary):
            #         new_text.append(c)
            #         if len(err_indices) != 0 and i < len(c_ary) - 1:
            #             new_text.append('[* ->]')
                # new_text.append('<FLR>')
            else:
                new_text.append(w)
        
        
        # elif w.startswith('[:'):
        #     ## Only replace if previous word was unintelligible

        #     # if not __is_word(text[idx-1]) and re.match(r'^[A-Za-z_]+$', text[idx-1]):
            
            
        #     # if len(new_text) > 0:
        #     #     # remove previous word
        #     #     new_text.pop()

        #     # add intended word if previous word is not an actual word
        #     w = w.split(" ")[-1][:-1]
        #     if __is_word(w) and re.match(r'^[A-Za-z_]+$', w):
        #         new_text.append(w)
    
        else:
            new_text.append(w)
    return new_text

def __proc_oovs(text, utt, lexicon, oov_dict):
    """ Sweep through the utterance and find OOVs, defined as words that are
    not in the `lexicon`. For each OOV, if it's all in ASCII, leave it as it
    is and add the word to oov_dict['ASCII']. The pronunciation of this word
    can be guessed by the CMU tool, which is probably better than anything
    I write up myself. If it has at least one non-ASCII character, we'll have
    to automatically derive the pronunciations of the word in terms of CMU
    phones and set oov_dict['UNIBET'][word] = pronunciations. Note that this is
    NOT a good function since it has many non-transparent side effects.
    """
    if len(text) == 0 or text[0] is None:
        return text
    new_text = []
    for w in text:
        if __is_word(w) and w not in lexicon:
            if common.is_ascii(w):
                # print("non ascii word:")
                io.log('non ascii word: {}'.format(w))
                if 'ASCII' not in oov_dict:
                    oov_dict['ASCII'] = []
                oov_dict['ASCII'].append(w)
                new_text.append(w)
            else:
                # for k in ['UNIBET', 'UNIBET_CNT']:
                #     if k not in oov_dict:
                #         oov_dict[k] = {}
                # name = '<U{}>'.format(__hash(w).upper())
                # if name not in oov_dict['UNIBET']:
                #     prons = __get_pronunciations(w, utt)
                #     oov_dict['UNIBET'][name] = prons
                #     io.log('{} - unibet pronunciation for {} ({}): {}'.format(
                #         utt, name, w.encode('string-escape'), prons
                #     ))
                #     oov_dict['UNIBET_CNT'][name] = 0
                # oov_dict['UNIBET_CNT'][name] += 1
                name = "<UNK>"
                new_text.append(name)
        else:
            new_text.append(w)
    return new_text

def __preprocess_unibet(word):
    """ Do some quick preprocessing given a unibet word.
    """
    word = word.replace('X', '')    # Sound not in CMU phones, must remove
    return word

def __get_pronunciations(word, utt):
    """ Guess the pronunciations of `word` in terms of CMU phones. `word` will
    have at least one non-ASCII (i.e., IPA) character. Will return a list.
    """
    word = __preprocess_unibet(word)
    word = word.decode('utf-8')
    pmap = __get_phone_mappings()
    # The algorithm is quite simple: we go from left to right and look for the
    # largest chunk of characters that can be translated at a time.
    prons = []
    idx = 0
    while idx < len(word):
        for clen in reversed(sorted(pmap.keys())):
            if clen > (len(word) - idx):
                continue
            fragment = word[idx:idx+clen]
            if fragment in pmap[clen]:
                if pmap[clen][fragment] is not None:
                    prons.append(pmap[clen][fragment])
                idx += clen
                break
            elif clen == 1:
                wrd_str = word.encode('utf-8').encode('string-escape')
                err_str = '{} - no match at index {} '.format(utt, idx) + \
                        'for {} (pron so far: {})'.format(wrd_str, prons)
                raise ValueError(err_str)
    return [' '.join(pron) for pron in itertools.product(*prons)]

def __get_phone_mappings():
    return {
        1: {
            # Unicode first, these are actually easier to handle
            u'\u00E6':['ae'], u'\u00F0':['dh'], u'\u014B':['ng'],
            u'\u0251':['aa'], u'\u0254':['ao'], u'\u0258':['ah'],
            u'\u0259':['ah'], u'\u025A':['aa', 'aw'], u'\u025B':['eh'],
            u'\u025C':['eh', 'er'], u'\u025D':['er'], u'\u0261':['g'],
            u'\u026A':['ih'], u'\u026B':['l'], u'\u0279':['r'],
            u'\u027E':['d'], u'\u0283':['sh'],u'\u028A':['uh'],
            u'\u028C':['ah'], u'\u0292':['zh'], u'\u02A4':['jh'],
            u'\u02A7':['ch'], u'\u0329':['l'], u'\u03B8':['th'],
            u'\u0270':None, u'\u02B0':None,
            # Letters that have one-to-one correspondence
            'B':['b'], 'D':['d'], 'F':['f'], 'G':['g'], 'H':['hh'], 'K':['k'],
            'M':['m'], 'N':['n'], 'P':['p'], 'S':['s'], 'T':['t'], 'V':['v'],
            'W':['w'], 'Z':['z'],
            # Other 1-to-1 mappings
            'I':['iy'], 'U':['uh'], 'R':['r'], 'J':['y'], 'U':['uw'],
            # Added to cover all cases
            'E':['ey', 'eh'], 'L':['l'], 'O':['ow', 'aa'], 'Y':['y'],
            'A':['aa', 'ae'], 'Q':['k'], 'C':['k']
        },
        2: {
            u'E\u026A':['ey'], u'A\u026A':['ay'], u'O\u028A':['ow'],
            u'A\u028A':['aw'], u'\u0254\u026A':['oy'],
            u'\u026AR':['ih r', 'iy r'], u'\u0259\u02DE':['er'],
            u'\u025C\u02DE':['er'], u'T\u0283':['ch'], u'D\u0292':['jh'],
            u'\u0251\u026A':['ay'], u'\u0251\u028A':['aw'],
            u'\u025B\u026A':['ey'], 'ER':['er'], 'CR':['k r'],
            u'N\u02DE':['n'], u'D\u02DE':['d'], u'\u00E6\u0303':['ae']
        }
    }

def __hash(word):
    """ Return hash code for this word. This ensures that the same word will
    always map to the same name and will help us count unibets more accurately.
    Currently using MD5, but may change to others.
    """
    m = hashlib.md5()
    m.update(word)
    return m.hexdigest()

def __get_word_labels(text, utt):
    """ Return word-level error labels for text. Each error label is an
    array of strings, containing:
        <wrd_start> <wrd_end> <label1> <label1> ...
    """
    if len(text) == 0 or text[0] is None:
        return None
    wlabels_raw = []
    widx = -1
    for idx, w in zip(range(len(text)), text):
        if __is_word(w):
            widx += 1
            wlabel = [widx]
            err_indices = __find_errs(text, idx)
            if len(err_indices) == 0:
                wlabel.append('C/{}'.format(w))
            else:
                io.log('**WARN** {} - word with multiple errors: {}'.format(
                    utt, text[idx:err_indices[-1]+1]
                ))
                # Word target is shared across different error types
                target = __parse_target(text, idx, err_indices[-1] + 1, utt)
                for err_idx in err_indices:
                    err_code = __parse_err_code(text, err_idx, utt)
                    wlabel.append('{}/{}'.format(err_code, target))
            wlabels_raw.append(wlabel)
    return __finalize_wlabels(wlabels_raw, text, utt)

def __parse_err_code(text, idx, utt):
    err_code = text[idx].replace('[*', '').replace(']', '').strip()
    # Uncategorized error
    if err_code in ['']:
        return 'X'
    # Check that error is of known type
    if not any([err_code.startswith(c) for c in ['p', 's', 'n', 'd', 'm', 'f', '->']]):
        io.log('WARNING: {} - invalid err_code {} ({})'.format(
            utt, err_code, text[:idx+1]
        ))
    return err_code

def __parse_target(text, start, end, utt):
    targets = [w for w in text[start:end] if w.startswith("[:")]
    if len(targets) == 0:
        return '?'
    if len(targets) > 1:
        io.log('**WARN** {} - multiple targets {}, using 1st item ({})'.format(
            utt, targets, text[:end]
        ))
    target = targets[0]
    target = [target.replace('[:', '').replace(']', '').strip()]
    target = __clean_word_tokens(target, utt)
    target = __proc_compound_words(target, utt)
    target = ' '.join(target)
    return target if target.lower() != 'x@n' else '?'

def __finalize_wlabels(wlabels_raw, text, utt):
    """ Merge error labels of compound words into a single error label.
    """
    wlabels = []
    start = -1
    # print('wlabels_raw: {}'.format(wlabels_raw))
    for x in wlabels_raw:
        idx = x[0]
        if not any([l.startswith('->') for l in x[1:]]):
            y = [start if start != -1 else idx, idx + 1]
            y.extend(x[1:])
            y = [str(i) for i in y]
            # print('y: {}'.format(y))
            wlabels.append(y)
            # Reset
            start = -1
        elif not all([l.startswith('->') for l in x[1:]]):
            raise ValueError('{} - mixed errors {} ({})'.format(utt, x, text))
        elif start == -1:
            start = idx
        else:
            continue
    if start != -1:
        raise ValueError('{} - final start is not -1, wlabels {} ({})'.format(
            utt, wlabels, text
        ))
    return wlabels

def finalize_words(text):
    return_arr = []
    for w in text:
        if "+" in w and __is_word(w):
            w_arr = w.split("+")
        elif "_" in w and __is_word(w):
            w_arr = w.split("_")
        else:
            w_arr = [w]
        
        for sub_w in w_arr:
            # io.log("sub_w: {}".format(sub_w))
            if __is_word(sub_w) and common.is_ascii(sub_w):
                sub_w = sub_w.replace(":","")
                sub_w = sub_w.replace("'","")
                sub_w = sub_w.replace("$on","")
                sub_w = sub_w.replace("-"," ")
                return_arr.append(sub_w.upper())

    return return_arr



def __finalize(text, utt):
    """ Last function to be called in the cleaning process.
    """
    if len(text) == 0 or text[0] is None:
        return text
    # return [w for w in text if __is_word(w) and re.match(r'^[A-Za-z_]+$', w)]
    # text = [w.split("+") for w in text]
    # return [w.replace("'","") for w in text if __is_word(w) and common.is_ascii(w)] # ensure ascii only
    return finalize_words(text)

