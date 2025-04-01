from matplotlib import pyplot as plt
from watermark_chunk_method import DatasetWatermark, ExtractionFailure
import csv
from random import sample
from math import ceil
import argparse
from os import fork,stat
from sys import exit
  
class TrialResult:

    def __init__(self, acc: float, bit_acc: float,  N: int, k: int, embed_len: int) -> None:
        self.acc = acc
        self.bit_acc = bit_acc
        self.N = N
        self.k = k
        self.embed_len = embed_len


CSV_CACHE = dict()

# attacks = [{'attack': 'meme_format', "param": 0}] \
#     + [{'attack': 'rotation', 'param': jj} for jj in range(1, 45, 5)] \
#     + [{'attack': 'center_crop', 'param': 0.1*jj} for jj in range(1, 10)] \
#     + [{'attack': 'resize', 'param': 0.1*jj} for jj in range(3, 10)] \
#     + [{'attack': 'blur', 'param': 1+2*jj} for jj in range(1, 7)] \
#     + [{'attack': 'jpeg', 'param': 10*jj} for jj in range(3, 11)] \
#     + [{'attack': 'contrast', 'param': 0.5*jj} for jj in range(1, 5) if jj != 2] \
#     + [{'attack': 'brightness', 'param': 0.5*jj} for jj in range(1, 5) if jj != 2] \
#     + [{'attack': 'hue', 'param': -0.5 + 0.25*jj}
#         for jj in range(0, 5) if jj != 2]

attacks = [{'attack': 'none','param': 0}] \
    + [{'attack': 'rotation', 'param': jj} for jj in range(5, 50, 5)] \
    + [{'attack': 'center_crop', 'param': 0.1*jj} for jj in range(1, 10)] \
    + [{'attack': 'resize', 'param': 0.1*jj} for jj in range(1, 10)] \
    + [{'attack': 'blur', 'param': 1+2*jj} for jj in range(1, 7)] \
    + [{'attack': 'jpeg', 'param': 10*jj} for jj in range(3, 11)] \
    + [{'attack': 'contrast', 'param': 0.5*jj} for jj in range(1, 5) if jj != 2] \
    + [{'attack': 'brightness', 'param': 0.5*jj} for jj in range(1, 5) if jj != 2] \
    + [{'attack': 'hue', 'param': -0.5 + 0.25*jj} for jj in range(0, 5) if jj != 2] \
    + [{'attack': 'meme_format','param': 0}] \
    + [{'attack': 'overlay_onto_screenshot','param': 0}] 

#attacks = [{'attack': 'hue', 'param': -0.5 + 0.25*jj} for jj in range(4, 5) if jj != 2] 

def read_csv(psnr: int, tau: int, msg_l: int, msg_set: int, wm_method: str):

    if (psnr, tau, msg_l, msg_set) in CSV_CACHE:
        return CSV_CACHE[(psnr, tau,msg_l,msg_set)]

    CSV_CACHE[(psnr, tau,msg_l,msg_set)] = []

    filename = "output/df_ml:%d_set:%d_psnr:%d_tau:%d_%s.csv" % (msg_l, msg_set, psnr, tau,wm_method)

    with open(filename, "r") as f:

        csvf = csv.reader(f)
        next(csvf)
        for row in csvf:
            # print(row)
            # row[2] = eval(row[2])
            # row[3] = eval(row[3])
            #saeed
            if row[2] == '':
                row[2]='0'
            #print(row)
            row[3] = eval(row[3])
            row[4] = eval(row[4])

            CSV_CACHE[(psnr, tau,msg_l,msg_set)].append(row)

    return CSV_CACHE[(psnr, tau,msg_l,msg_set)]


def compute_extraction_accuracy(messages: 'list[str]', tau: int, percent_leaked: float,message_path: str):
    #saeed
    with open(message_path, "r") as f:
        lines = f.readlines()
        lines = list(map(lambda r: r.strip("\n"), lines))
        original_msg = lines[0]
    
    #dwm = DatasetWatermark("01" * 50, tau / 100)
    #saeed
    dwm = DatasetWatermark(original_msg, tau / 100)
    success = 0
    trials = 100
    sum_acc = 0.
    for i in range(trials):
        if tau == 0:
            samples = sample(messages, ceil(percent_leaked * len(messages)))

            # maj = dwm.compute_bitwise_majority(samples)
            recon_msg  = dwm.compute_bitwise_majority(samples) 
            if recon_msg == dwm.message:
                success += 1
            sum_acc += sum(1 for a, b in zip(recon_msg,dwm.message) if a == b)/float(len(dwm.message))
        else:

            try:
                #saeed
                status, bit_acc = dwm.can_extract_watermark(sample(messages, ceil(percent_leaked * len(messages))), len(original_msg))
                if status:
                    success += 1
                sum_acc += bit_acc
            except ExtractionFailure as e:
                pass
    #return TrialResult(acc=success/trials, N=dwm.N, k=dwm.N - dwm.chunks_per_message, embed_len=len(messages[0]))
    return TrialResult(acc=success/trials*100., bit_acc=sum_acc/trials*100., N=dwm.N, k=dwm.N - dwm.chunks_per_message, embed_len=len(messages[0]))


def param_to_integer(param: float):

    return int(param * 100)


def read_messages(psnr, tau, msg_l, msg_set, attack, param0,wm_method):
    data_rows = read_csv(psnr, tau,msg_l, msg_set,wm_method)

    param0 = param_to_integer(param0)
    
    messages = []
    pred_acc = []
    for row in data_rows:

        #img, att, par, msg_orig, msg_decoded = row
        img, att, par, msg_orig, msg_decoded, pred = row #ML utility 
        par = param_to_integer(float(par))

        if att != attack:
            continue
        
        #if att != "meme_format":
        if (att != "meme_format" or att != "none" or att != "overlay_onto_screenshot" ):
            if par != param0:
                continue
        
        msg_orig = "".join(map(lambda r: "1" if r else "0", msg_orig))    
        msg_decoded = "".join(map(lambda r: "1" if r else "0", msg_decoded))    

        messages.append(msg_decoded)
        #ML utility 
        pred_acc.append(eval(pred))
    
    #print(pred_acc)
    return messages ,sum(pred_acc)/len(pred_acc)


def extraction_accuracy_for(psnr, tau, msg_l, msg_set, attack, param0, percent_leaked,message_path,wm_method):
    messages, ml_acc = read_messages(psnr, tau, msg_l, msg_set, attack, param0,wm_method)

    return compute_extraction_accuracy(messages, tau, percent_leaked,message_path), ml_acc


if __name__ == "__main__":

    cli = argparse.ArgumentParser()

    cli.add_argument("--psnr", type=int, nargs="+",
                     default=[20, 25, 30, 35, 40, 45], required=False) #[ 20,25, 30, 35, 40, 45]
    cli.add_argument("--tau", type=int, nargs="+",
                     default=[0, 60, 80], required=False) #0,60,80,  ==> ,20, 40, 60, 80, 100

    cli.add_argument("--msg_sets", type=int, nargs="+",
                     default=[0,1,2,3,4,5,6,7,8,9], required=False)
    
    cli.add_argument("--original_msg_length", type=int, default=100, help="Original message length")
    cli.add_argument("--wm_method", type=str, default="dwtdct", help="The watermarking method used, either SSL or dwtdct")

    cli.add_argument("--attacks", nargs="+", default=None, required=False)
    #cli.add_argument("-o", type=str, action='store',
    #                 default="out.csv", required=False)
    #saeed
    #cli.add_argument("--message_path", type=str, action='store',
    #                 default="ssl_watermarking/users/80/baseline.txt", required=False)
    params = cli.parse_args()

    if params.attacks is None:
        params.attacks = attacks
    else:
        user_attacks = []

        for att in params.attacks:

            attack, param = att.split(":")

            user_attacks.append({
                "attack": attack,
                "param": float(param)
            })

        params.attacks = user_attacks

    for msg_set in params.msg_sets:
        params.o = f'out_{params.original_msg_length}_{msg_set}_{params.wm_method}.csv'
        params.message_path = f'ssl_watermarking/users/{params.original_msg_length}/{msg_set}/baseline.txt'
        outcsv = open(params.o, "w")
        csvf = csv.DictWriter(outcsv, fieldnames=[
            "psnr", "tau", "attack", "s", "acc","bit_acc", "N", "k", "embed_len","ml_acc"])
        
        #saeed
        file_is_empty = stat(params.o).st_size == 0
        if file_is_empty:
            csvf.writeheader()
        
        for psnr in params.psnr:
            for tau in params.tau:  # [20, 40, 60, 80, 100]:
                
                #if fork() != 0:
                #   continue

                for attack in params.attacks:
                    for s in [100]: #range(0, 101, 10): #[100]: #range(0, 100 + 2, 20)
                        attack_name = attack["attack"]
                        attack_param = attack["param"]
                        #print("=====>",psnr,tau,attack)
                        result, ml_acc = extraction_accuracy_for(
                            psnr, tau, params.original_msg_length, msg_set, attack_name, attack_param, s / 100,params.message_path, params.wm_method)

                        csvf.writerow({
                            "psnr": psnr,
                            "tau": tau,
                            "attack": "%s:%f" % (attack_name, attack_param),
                            "s": s / 100,
                            "acc": result.acc,
                            "bit_acc": result.bit_acc,
                            "N": result.N,
                            "k": result.k,
                            "embed_len": result.embed_len,
                            "ml_acc": ml_acc
                        })

                        if s==100:
                            print("Exraction Accuracy (psnr=%d, tau=%d, s=%d%%, attack=%s:%.2f, N=%d, k=%d, embed_len=%d, ml_acc=%.2f) = acc:%.2f" % (
                            psnr, tau, s, attack_name, attack_param, result.N, result.k, result.embed_len,ml_acc,result.acc))

                #exit(0)


        outcsv.close()



