'''
Create Speechbrain CSV files
Input: Kaldi dir
'''
import os
import pandas as pd
import subprocess

KALDI_ROOT = "/z/mkperez/AphasiaBank/kd_updated_para/Scripts"
SB_OUTDIR = "/z/mkperez/speechbrain/AphasiaBank/data/Fridriksson_para_best_Word"

KALDI_CV_DIR = f"{KALDI_ROOT}/CV"
PARA_FILENAME = f"{KALDI_ROOT}/wrd_labels"
SB_OUTWAV_PATH=f"{SB_OUTDIR}/wavs"
OG_WAV_PATH = f"{KALDI_ROOT}/owav.scp"

def read_dict(filename):
    return_dict = {}
    with open(filename, 'r') as r:
        lines = r.readlines()
        for line in lines:
            line = line.strip()
            utt_id = line.split()[0]
            if filename==OG_WAV_PATH:
                content = line.split("\t",1)[1]
            else:
                content = line.split(" ",1)[1]
            return_dict[utt_id] = content

    return return_dict

def prepare_utt2paraphasia():
    utt2para = read_dict(PARA_FILENAME)
    for k,v in utt2para.items():
        
        para_list = []
        for para in v.split():
            if para.startswith("n:"):
                para_list.append("n")
            elif para.startswith("p:"):
                para_list.append("p")
            elif para.startswith("s:"):
                para_list.append("s")
            else:
                para_list.append("c")
                # print(f"Error finding label {para}")
                # exit(1)

        utt2para[k] = " ".join(para_list)
    return utt2para


def create_csv(rdir, wfilename):
    text_file = f"{rdir}/text"
    UTT2TEXT = read_dict(text_file)
    dur_file = f"{rdir}/durations"
    UTT2DUR = read_dict(dur_file)
    seg_file = f"{rdir}/segments"
    UTT2SEG = read_dict(seg_file)
    UTT2PARA = prepare_utt2paraphasia()
    SPK2WAV = read_dict(OG_WAV_PATH)
    

    df_list = []
    for utt_id in UTT2TEXT.keys():
        spk_id = utt_id.split("-")[0]
        wrd = UTT2TEXT[utt_id].lower()
        dur = round(float(UTT2DUR[utt_id]),2)
        paraphasia = UTT2PARA[utt_id]
        if len(wrd.split()) != len(paraphasia.split()):
            continue
        aug_para = " ".join([f"{w}/{p}" for w,p in zip(wrd.split(),paraphasia.split())])

        # check wav file creation
        seg_info = UTT2SEG[utt_id]
        og_wav_loc = SPK2WAV[spk_id]
        check_wav_file_exists(utt_id,seg_info,og_wav_loc)


        df_loc = pd.DataFrame({
            'wrd':[wrd],
            'spk_id':[spk_id],
            'ID':[utt_id],
            'wav':[f"{SB_OUTWAV_PATH}/{utt_id}.wav"],
            'duration':[dur],
            'paraphasia':[paraphasia],
            'aug_para':[aug_para],
        })
        df_list.append(df_loc)

    df = pd.concat(df_list)
    df = df.reset_index(drop=True)
    df = df.reset_index()
    df = df.rename(columns={"index":"Unnamed: 0"})
    # print(df)
    # exit()
    df.to_csv(wfilename)
    # print(f"output: {wfilename}")

def check_wav_file_exists(utt_id,seg_info,og_wav_loc):
    
    out_wav_filename = f"{SB_OUTWAV_PATH}/{utt_id}.wav"

    # create if doesnt exist
    if not os.path.exists(out_wav_filename):
        start = float(seg_info.split()[1])
        end = float(seg_info.split()[2])

        # sox
        command = [
            'sox', og_wav_loc, out_wav_filename,
            'trim', str(start), str(end)
        ]
        subprocess.run(command, check=True)



if __name__ == "__main__":
    for partition in ['test','dev','train']:
        for fold_dir in sorted(os.listdir(KALDI_CV_DIR)):
            # make file
            wfilename = f"{SB_OUTDIR}/{fold_dir}/{partition}_multi.csv"
            os.makedirs(f"{SB_OUTDIR}/{fold_dir}", exist_ok=True)

            rdir = f"{KALDI_CV_DIR}/{fold_dir}/{partition}"

            create_csv(rdir, wfilename)

