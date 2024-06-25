#!/bin/bash
set -e

############################################################################
# Prepare AphasiaBank data, primarily for Kaldi.                           #
# NOTE: this script isn't meant to be run directly. It simply outlines the #
# steps to be taken in order. Please consult the referenced scripts.       #
############################################################################

#---------------- CONSTANTS ------------------#
DATA_DIR="/z/public/data/AphasiaBank" # root directory containing .cha files
EXP_DIR="/z/mkperez/AphasiaBank" # root experiment directory to output new files
UTILS_PATH="$KD_ROOT/../egs/wsj/s5/utils"
LISTING_DIR="$EXP_DIR/listing"
KD_DATA_DIR="$EXP_DIR/kaldi_data"
TEMPLATES_DIR="$KD_DATA_DIR/templates"

#---------------- COMMAND-LINE ARGUMENTS ---------------#
stage=0       # Control which step to run
cmd="$UTILS_PATH/run.pl"
datasets="Scripts"
# For validation split
split_by_db="False"

#---------------- MAIN PROGRAM ---------------#
. $UTILS_PATH/parse_options.sh
if [ $stage != -1 ]; then
    echo "Specific stage specified: stage=$stage"
fi

if [ $stage == -1 ] || [ $stage == 0 ]; then
    echo "Convert all CHAT transcriptions to Praat format"
    for dataset in $datasets; do
        echo "...$dataset"
        cd $DATA_DIR/$dataset
        chat2praat +e.wav +re *.cha
    done
fi


if [ $stage -lt -1 ] || [ $stage == 1 ]; then
    echo "Finding all transcriptions and wav pairs"
    mkdir -p $LISTING_DIR
    for dataset in $datasets; do
        echo "...$dataset"
        for f in wav_list praat_list; do
            fpath=$LISTING_DIR/${f}.${dataset}.txt
            [ ! -f "$fpath" ] || rm $fpath
        done
        for fpraat in `find $DATA_DIR/$dataset -iname '*.c2praat.textGrid' | sort`; do
            fwav=`echo "$fpraat" | sed -e 's/c2praat\.textGrid/wav/'`
            if [ -f "$fwav" ]; then
                echo "$fwav" >>$LISTING_DIR/wav_list.${dataset}.txt
                echo "$fpraat" >>$LISTING_DIR/praat_list.${dataset}.txt
            fi
        done
        echo "Found `cat $LISTING_DIR/wav_list.${dataset}.txt | wc -l` pairs"
        paste $LISTING_DIR/wav_list.${dataset}.txt $LISTING_DIR/praat_list.${dataset}.txt \
            >$LISTING_DIR/wav2praat.${dataset}.txt
    done
fi

if [ $stage -le -1 ] || [ $stage == 2 ]; then
    echo "Create segments"
    for dataset in $datasets; do
        echo "...$dataset"
        kd_dir=$KD_DATA_DIR/$dataset && logs_dir=$kd_dir/logs && mkdir -p $logs_dir

        echo "Original wav mapping"
        [ ! -f $kd_dir/owav.scp ] || rm $kd_dir/owav.scp
        for fwav in `cat $LISTING_DIR/wav_list.${dataset}.txt`; do
            utt=`echo "$fwav" | sed -e 's/.*\///' -e 's/\.wav//'`
            echo -e "$utt\t$fwav" >>$kd_dir/owav.scp
        done

        if [ -f $logs_dir/get-segments.log ]; then
            rm $logs_dir/get-segments.log
        fi
        echo "Segments"
        $cmd $logs_dir/get-segments.log python \
            helper_scripts/get_segments.py \
            $kd_dir/owav.scp $kd_dir/segments $kd_dir/text $kd_dir/oovs \
            $kd_dir/unibets $kd_dir/wrd_labels
        # Manually merge generated dicts with main lexicon!
    done
fi

if [ $stage -le -1 ] || [ "$stage" == "2.5" ]; then
    echo "Extract segments from original audio"
    for dataset in $datasets; do
        echo "...$dataset"
        kd_dir=$KD_DATA_DIR/$dataset && logs_dir=$kd_dir/logs && mkdir -p $logs_dir

        echo "Extract segments"
        $cmd $logs_dir/extract-segments.log extract-segments \
            scp:$kd_dir/owav.scp $kd_dir/segments ark,scp:$kd_dir/wav.ark,$kd_dir/wav.scp

        echo "Get durations"
        $cmd $logs_dir/wav-to-duration.log wav-to-duration \
            scp:$kd_dir/wav.scp ark,t:$kd_dir/durations
        # IMPORTANT! manually inspect utts with very long durations to see if
        # transcripts make sense. If not, remove them from wav.scp.
        cp $kd_dir/durations $kd_dir/utt2dur
    done
fi

if [ $stage -le -1 ] || [ $stage == 3 ]; then
    echo "Prepare Kaldi coding files"
    for dataset in $datasets; do

        # pushd $UTILS_PATH/..
        # LANG_DIR="$KD_DATA_DIR/$dataset/lang"
        # echo "prep lang model"
        # $UTILS_PATH/prepare_lang.sh $TEMPLATES_DIR '<UNK>' $KD_DATA_DIR/$dataset/tmp $LANG_DIR || exit 1;
        # popd
        # exit


        echo "...$dataset"
        kd_dir=$KD_DATA_DIR/$dataset && logs_dir=$kd_dir/logs && mkdir -p $logs_dir


        # Scripts
        cut -f1 $kd_dir/wav.scp -d ' ' >$kd_dir/utts
        cat $kd_dir/utts | sed -e 's/_.*//' >$kd_dir/spks
        # cat $kd_dir/utts | sed -n -e 's/.*_\([S][^_]*\).*/\1/p' >$kd_dir/spks # split on script
        cat $kd_dir/utts | sed -n -e 's/.*_\([S][^_]*\).*/Frid/p' > $kd_dir/dbs

        echo "utt2spk and spk2utt"
        paste $kd_dir/utts $kd_dir/spks >$kd_dir/utt2spk
        $UTILS_PATH/utt2spk_to_spk2utt.pl $kd_dir/utt2spk >$kd_dir/spk2utt
        echo "utt2db and db2utt"
        paste $kd_dir/utts $kd_dir/dbs >$kd_dir/utt2db
        $UTILS_PATH/utt2spk_to_spk2utt.pl $kd_dir/utt2db >$kd_dir/db2utt
        echo "utt2glob and glob2utt (dummy)"
        cat $kd_dir/utts | sed -e 's/$/\tglob/' >$kd_dir/utt2glob
        $UTILS_PATH/utt2spk_to_spk2utt.pl $kd_dir/utt2glob >$kd_dir/glob2utt
        echo "spk2db and db2spk"
        cat $kd_dir/utt2db | sed -e 's/-.*\t/\t/' | sort | uniq >$kd_dir/spk2db
        $UTILS_PATH/utt2spk_to_spk2utt.pl $kd_dir/spk2db >$kd_dir/db2utt
        echo "spk2spk (dummy)"
        cat $kd_dir/spk2db | sed -e 's/\([^ ]\+\)\s.*/\1\t\1/' >$kd_dir/spk2spk

        echo "Remove utterances in text that do not have MFCC"
        mv $kd_dir/text $kd_dir/text.pre-trim
        python helper_scripts/trim.py $kd_dir/utts \
            $kd_dir/text.pre-trim >$kd_dir/text
    done
fi

# Extract mfcc and mfbs in hmm-gmm script
if [ $stage -le -1 ] || [ $stage == 4 ]; then
    echo "Extract raw features"
    for dataset in $datasets; do
        echo "...$dataset"
        kd_dir=$KD_DATA_DIR/$dataset && logs_dir=$kd_dir/logs && mkdir -p $logs_dir

        echo "MFCC..."
        $cmd $logs_dir/compute-mfcc-feats.log compute-mfcc-feats \
            --config=$TEMPLATES_DIR/mfcc.config scp,p:$kd_dir/wav.scp \
            ark,scp:$kd_dir/raw_mfcc.ark,$kd_dir/raw_mfcc.scp

        echo "MFB..."
        $cmd $logs_dir/compute-fbank-feats.log compute-fbank-feats \
            --config=$TEMPLATES_DIR/mfb.config scp,p:$kd_dir/wav.scp \
            ark,scp:$kd_dir/raw_mfb.ark,$kd_dir/raw_mfb.scp
    done
fi

if [ $stage -le -1 ] || [ $stage == 5 ]; then
    echo "Perform speaker z-normalization, intended for ASR"
    for dataset in $datasets; do
        echo "...$dataset"
        kd_dir=$KD_DATA_DIR/$dataset && logs_dir=$kd_dir/logs && mkdir -p $logs_dir

        echo "Compute speaker CMVN stats"
        # $cmd $logs_dir/compute-spk-cmvn-stats.log compute-cmvn-stats \
        #     --spk2utt=ark,t:$kd_dir/spk2utt scp:$kd_dir/raw_mfcc.scp \
        #     ark,scp:$kd_dir/cmvn.ark,$kd_dir/cmvn.scp
        # $cmd $logs_dir/compute-spk-cmvn-stats-mfb.log compute-cmvn-stats \
        #     --spk2utt=ark,t:$kd_dir/spk2utt scp:$kd_dir/raw_mfb.scp \
        #     ark,scp:$kd_dir/cmvn_mfb.ark,$kd_dir/cmvn_mfb.scp

        compute-cmvn-stats \
            --spk2utt=ark,t:$kd_dir/spk2utt scp:$kd_dir/raw_mfcc.scp \
            ark,scp:$kd_dir/cmvn.ark,$kd_dir/cmvn.scp
        compute-cmvn-stats \
            --spk2utt=ark,t:$kd_dir/spk2utt scp:$kd_dir/raw_mfb.scp \
            ark,scp:$kd_dir/cmvn_mfb.ark,$kd_dir/cmvn_mfb.scp



        echo "Apply CMVN"
        # $cmd $logs_dir/apply-cmvn.log apply-cmvn \
        #     --norm-vars=true --utt2spk=ark,t:$kd_dir/utt2spk \
        #     scp:$kd_dir/cmvn.scp scp:$kd_dir/raw_mfcc.scp \
        #     ark,scp:$kd_dir/feats_mfcc_nodelta.ark,$kd_dir/feats_mfcc_nodelta.scp
        # $cmd $logs_dir/apply-cmvn-mfb.log apply-cmvn \
        #     --norm-vars=true --utt2spk=ark,t:$kd_dir/utt2spk \
        #     scp:$kd_dir/cmvn_mfb.scp scp:$kd_dir/raw_mfb.scp \
        #     ark,scp:$kd_dir/feats_mfb.ark,$kd_dir/feats_mfb.scp
        
        apply-cmvn \
            --norm-vars=true --utt2spk=ark,t:$kd_dir/utt2spk \
            scp:$kd_dir/cmvn.scp scp:$kd_dir/raw_mfcc.scp \
            ark,scp:$kd_dir/feats_mfcc_nodelta.ark,$kd_dir/feats_mfcc_nodelta.scp
        apply-cmvn \
            --norm-vars=true --utt2spk=ark,t:$kd_dir/utt2spk \
            scp:$kd_dir/cmvn_mfb.scp scp:$kd_dir/raw_mfb.scp \
            ark,scp:$kd_dir/feats_mfb.ark,$kd_dir/feats_mfb.scp


        echo "Add deltas"
        # $cmd $logs_dir/add-deltas.txt add-deltas \
        #     --delta-order=2 scp:$kd_dir/feats_mfcc_nodelta.scp \
        #     ark,scp:$kd_dir/feats_mfcc.ark,$kd_dir/feats_mfcc.scp
        add-deltas \
            --delta-order=2 scp:$kd_dir/feats_mfcc_nodelta.scp \
            ark,scp:$kd_dir/feats_mfcc.ark,$kd_dir/feats_mfcc.scp


        echo "Clean up"
        rm $kd_dir/feats_mfcc_nodelta.{ark,scp}
        # Using MFCC as the default features for now. When training nnet,
        # change the link to point to MFB instead.
        [ -L $kd_dir/feats.scp ] && rm $kd_dir/feats.scp
        ln -s feats_mfcc.scp $kd_dir/feats.scp
    done
fi


if [ $stage -le -1 ] || [ $stage == 7 ]; then
    echo "Partitioning data for cross-validation"
    for dataset in $datasets; do
        echo "...$dataset"
        kd_dir=$KD_DATA_DIR/$dataset

        echo "Output utterances"
        if [ "$split_by_db" == "True" ] || [ "$split_by_db" == "true" ]; then
            echo "Splitting by db"
            python helper_scripts/partition.py \
                --nfolds 5 \
                $kd_dir/spk2db $kd_dir/spk2utt $kd_dir/CV
        else
            echo "Splitting by spk"
            python helper_scripts/partition_spk.py \
                $kd_dir/spk2utt $kd_dir/CV
        fi

        echo "Trim content"
        for dir in `find $kd_dir/CV/Fold_*/* -maxdepth 0 -type d`; do
            echo "-- $dir --"
            files="durations segments text wav.scp utt2dur"
            files="$files utt2db utt2glob utt2spk"
            files="$files feats.scp cmvn.scp feats_mfcc.scp feats_mfb.scp"
            [ "$dataset" == "Script" ] && files="$files utt2script"
            for file in $files; do
                echo "$file"
                python helper_scripts/trim.py \
                    $dir/utts $kd_dir/$file >$dir/$file
                # If the file is in "utt2" form, also create reverse mappings
                if [ ! -z `echo "$file" | grep "^utt2"` ]; then
                    rfile=`echo "$file" | sed -e 's/utt2\(.*\)/\12utt/'`
                    echo "$file --> $rfile"
                    $UTILS_PATH/utt2spk_to_spk2utt.pl $dir/$file >$dir/$rfile
                fi
            done
        done
    done
fi
