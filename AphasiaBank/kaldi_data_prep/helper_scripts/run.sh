#!/bin/bash
set -e

############################################################################
# Prepare AphasiaBank data, primarily for Kaldi.                           #
# NOTE: this script isn't meant to be run directly. It simply outlines the #
# steps to be taken in order. Please consult the referenced scripts.       #
############################################################################

#---------------- CONSTANTS ------------------#
UTILS_PATH="$KD_ROOT/../egs/wsj/s5/utils"
ROOT_DIR="/home/ducle/Dropbox/UMich/Research/Aphasia_Bank"
SCRIPTS_DIR="$ROOT_DIR/Scripts"
DATA_DIR="$ROOT_DIR/data"
LISTING_DIR="$ROOT_DIR/listing"
KD_DATA_DIR="$ROOT_DIR/kaldi_data"
TEMPLATES_DIR="$ROOT_DIR/templates"

#---------------- COMMAND-LINE ARGUMENTS ---------------#
stage=-1        # Control which step to run
cmd="$UTILS_PATH/run.pl"
datasets="Control Aphasia"
# For validation split
split_by_db="True"
# For task-specific split
tasks="script"

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

if [ $stage == -1 ] || [ $stage == 1 ]; then
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

if [ $stage == -1 ] || [ $stage == 2 ]; then
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

        echo "Segments"
        $cmd $logs_dir/get-segments.log python \
            $SCRIPTS_DIR/data_prep/get_segments.py \
            $kd_dir/owav.scp $kd_dir/segments $kd_dir/text $kd_dir/oovs \
            $kd_dir/unibets $kd_dir/wrd_labels
        # Manually merge generated dicts with main lexicon!
    done
fi

if [ $stage == -1 ] || [ "$stage" == "2.5" ]; then
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
    done
fi

if [ $stage == -1 ] || [ $stage == 3 ]; then
    echo "Prepare Kaldi coding files"
    for dataset in $datasets; do
        echo "...$dataset"
        kd_dir=$KD_DATA_DIR/$dataset && logs_dir=$kd_dir/logs && mkdir -p $logs_dir

        cut -f1 $kd_dir/wav.scp -d ' ' >$kd_dir/utts
        cat $kd_dir/utts | sed -e 's/-.*//' >$kd_dir/spks
        cat $kd_dir/utts | sed -e 's/[0-9].*//' > $kd_dir/dbs
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

        echo "spk2gender and gender2spk, spk2age, spk2diagnosis, spk2aq"
        for f in spk2age spk2diagnosis spk2aq spk2gender; do
            [ ! -f $kd_dir/$f ] || rm $kd_dir/$f
        done
        while read line; do
            spk=`echo "$line" | cut -f1`
            db=`echo "$line" | cut -f2`
            db=`python $SCRIPTS_DIR/data_prep/norm_dbname.py $db`
            cha="$DATA_DIR/$dataset/$db/${spk}.cha"
            par=`grep '@ID' $cha | grep PAR | sed -e 's/.*:\s\+//'`
            # Gender
            gender=`echo "$par" | cut -f5 -d '|'`
            gender=`python -c "print '$gender'.lower()"`
            if [ ! -z "$gender" ]; then
                echo -e "$spk\t$gender" >>$kd_dir/spk2gender
            else
                echo "No gender for $spk"
                exit 1
            fi
            # Age
            age=`echo "$par" | cut -f4 -d '|' | sed -e 's/;.*//'`
            if [ ! -z "$age" ]; then
                echo -e "$spk\t$age" >>$kd_dir/spk2age
            else
                echo "WARNING: no age for $spk"
            fi
            # Diagnosis
            diagnosis=`echo "$par" | cut -f6 -d '|'`
            diagnosis=`python -c "print '$diagnosis'.lower()"`
            if [ ! -z "$diagnosis" ]; then
                echo -e "$spk\t$diagnosis" >>$kd_dir/spk2diagnosis
            else
                echo "WARNING: no diagnosis for $spk"
            fi
            # AQ
            aq=`echo "$par" | cut -f10 -d '|'`
            if [ ! -z "$aq" ]; then
                echo -e "$spk\t$aq" >>$kd_dir/spk2aq
            else
                echo "WARNING: no AQ for $spk"
            fi
        done < $kd_dir/spk2db

        $UTILS_PATH/utt2spk_to_spk2utt.pl $kd_dir/spk2gender >$kd_dir/gender2spk
        echo "utt2gender and gender2utt"
        [ ! -f $kd_dir/utt2gender ] || rm $kd_dir/utt2gender
        while read line; do
            utt=`echo "$line" | cut -f1`
            spk=`echo "$line" | cut -f2`
            gender=`grep "^$spk" $kd_dir/spk2gender | cut -f2`
            if [ ! -z "$gender" ]; then
                echo -e "$utt\t$gender" >>$kd_dir/utt2gender
            else
                echo "No gender for $utt"
                exit 1
            fi
        done < $kd_dir/utt2spk
        $UTILS_PATH/utt2spk_to_spk2utt.pl $kd_dir/utt2gender >$kd_dir/gender2utt

        echo "Remove utterances in text that do not have MFCC"
        mv $kd_dir/text $kd_dir/text.pre-trim
        python $SCRIPTS_DIR/data_prep/trim.py $kd_dir/utts \
            $kd_dir/text.pre-trim >$kd_dir/text
    done
fi

if [ $stage == -1 ] || [ $stage == 4 ]; then
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

if [ $stage == -1 ] || [ $stage == 5 ]; then
    echo "Perform speaker z-normalization, intended for ASR"
    for dataset in $datasets; do
        echo "...$dataset"
        kd_dir=$KD_DATA_DIR/$dataset && logs_dir=$kd_dir/logs && mkdir -p $logs_dir

        echo "Compute speaker CMVN stats"
        $cmd $logs_dir/compute-spk-cmvn-stats.log compute-cmvn-stats \
            --spk2utt=ark,t:$kd_dir/spk2utt scp:$kd_dir/raw_mfcc.scp \
            ark,scp:$kd_dir/cmvn.ark,$kd_dir/cmvn.scp
        $cmd $logs_dir/compute-spk-cmvn-stats-mfb.log compute-cmvn-stats \
            --spk2utt=ark,t:$kd_dir/spk2utt scp:$kd_dir/raw_mfb.scp \
            ark,scp:$kd_dir/cmvn_mfb.ark,$kd_dir/cmvn_mfb.scp

        echo "Apply CMVN"
        $cmd $logs_dir/apply-cmvn.log apply-cmvn \
            --norm-vars=true --utt2spk=ark,t:$kd_dir/utt2spk \
            scp:$kd_dir/cmvn.scp scp:$kd_dir/raw_mfcc.scp \
            ark,scp:$kd_dir/feats_mfcc_nodelta.ark,$kd_dir/feats_mfcc_nodelta.scp
        $cmd $logs_dir/apply-cmvn-mfb.log apply-cmvn \
            --norm-vars=true --utt2spk=ark,t:$kd_dir/utt2spk \
            scp:$kd_dir/cmvn_mfb.scp scp:$kd_dir/raw_mfb.scp \
            ark,scp:$kd_dir/feats_mfb.ark,$kd_dir/feats_mfb.scp

        echo "Add deltas"
        $cmd $logs_dir/add-deltas.txt add-deltas \
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

:<<'DEFUNCT_DO_NOT_USE'
if [ $stage == -1 ] || [ $stage == 6 ]; then
    echo "Perform z-normalization on MFCC using global Control stats, intended for i-vectors"
    for dataset in $datasets; do
        echo "...$dataset"
        kd_dir=$KD_DATA_DIR/$dataset/glob_znorm && logs_dir=$kd_dir/logs && mkdir -p $logs_dir
        ctrl_dir=$KD_DATA_DIR/Control/glob_znorm

        if [ "$dataset" == "Control" ]; then
            echo "Compute global CMVN stats"
            $cmd $logs_dir/compute-spk-cmvn-stats.log compute-cmvn-stats \
                --spk2utt=ark,t:$kd_dir/../glob2utt \
                scp:$kd_dir/../raw_mfcc.scp \
                ark,scp:$kd_dir/cmvn.ark,$kd_dir/cmvn.scp
        fi

        echo "Apply CMVN"
        $cmd $logs_dir/apply-cmvn.log apply-cmvn \
            --norm-vars=true --utt2spk=ark,t:$kd_dir/../utt2glob \
            scp:$ctrl_dir/cmvn.scp scp:$kd_dir/../raw_mfcc.scp \
            ark,scp:$kd_dir/cmvn_mfcc.ark,$kd_dir/cmvn_mfcc.scp

        echo "Add deltas"
        $cmd $logs_dir/add-deltas.txt add-deltas \
            --delta-order=2 scp:$kd_dir/cmvn_mfcc.scp \
            ark,scp:$kd_dir/feats.ark,$kd_dir/feats.scp

        echo "Copying metadata over"
        cp $kd_dir/../{text,segments,utt2spk,spk2utt} $kd_dir

        echo "Clean up"
        rm $kd_dir/cmvn_mfcc.{ark,scp}
    done
fi
DEFUNCT_DO_NOT_USE

if [ $stage == -1 ] || [ $stage == 7 ]; then
    echo "Partitioning data for cross-validation"
    for dataset in $datasets; do
        echo "...$dataset"
        kd_dir=$KD_DATA_DIR/$dataset

        echo "Output utterances"
        if [ "$split_by_db" == "True" ] || [ "$split_by_db" == "true" ]; then
            echo "Splitting by db"
            python `dirname $0`/partition.py \
                $kd_dir/spk2db $kd_dir/spk2utt $kd_dir/CV
        else
            echo "Splitting by spk"
            python `dirname $0`/partition_spk.py \
                $kd_dir/spk2utt $kd_dir/CV
        fi

        echo "Trim content"
        for dir in `find $kd_dir/CV/Fold_*/* -maxdepth 0 -type d`; do
            echo "-- $dir --"
            files="cmvn.scp durations feats_mfcc.scp segments text wav.scp"
            files="$files feats_mfb.scp utt2db utt2gender utt2glob utt2spk"
            [ "$dataset" == "Script" ] && files="$files utt2script"
            for file in $files; do
                echo "$file"
                python `dirname $0`/trim.py \
                    $dir/utts $kd_dir/$file >$dir/$file
                # If the file is in "utt2" form, also create reverse mappings
                if [ ! -z `echo "$file" | grep "^utt2"` ]; then
                    rfile=`echo "$file" | sed -e 's/utt2\(.*\)/\12utt/'`
                    echo "$file --> $rfile"
                    $UTILS_PATH/utt2spk_to_spk2utt.pl $dir/$file >$dir/$rfile
                fi
            done
            # Using MFCC as the default features for now. When training nnet,
            # change the link to point to MFB instead.
            [ -L $dir/feats.scp ] && rm $dir/feats.scp
            ln -s feats_mfcc.scp $dir/feats.scp
        done
    done
fi

if [ $stage == -1 ] || [ $stage == 8 ]; then
    echo "Mapping from utt to task name"
    for dataset in $datasets; do
        echo "...$dataset"
        kd_data="$KD_DATA_DIR/$dataset"
        wav_list="$LISTING_DIR/wav_list.${dataset}.txt"
:<<'COMMENT'
        for f in `cat $wav_list`; do
            fname=`echo "$f" | sed -e 's/\.wav$//'`
            [ ! -f ${fname}.cha ] && echo "Expect ${fname}.cha to exist!" && exit 1
            echo "Processing ${fname}.cha"
            python $SCRIPTS_DIR/data_prep/task2time.py \
                ${fname}.cha >${fname}.task2time
        done
COMMENT
        for task in $tasks; do
            python $SCRIPTS_DIR/data_prep/utt2group.py \
                $kd_data/segments.trim $kd_data/owav.scp \
                --alias $kd_data/${task}2alias \
                >$kd_data/utt2${task}

            $UTILS_PATH/utt2spk_to_spk2utt.pl $kd_data/utt2${task} \
                >$kd_data/${task}2utt
        done
    done
fi
