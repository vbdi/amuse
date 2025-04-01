import os
import time
import pprint
import argparse
import torch
import numpy as np
import pickle
import utils
import csv

from model.hidden import Hidden
from noise_layers.noiser import Noiser
from average_meter import AverageMeter
from noise_layers.identity import Identity

from watermark_chunk_method import DatasetWatermark, ExtractionFailure
import pandas as pd
import math
from random import sample
from matplotlib import pyplot as plt



def write_validation_loss(file_name, losses_accu, experiment_name, epoch, write_header=False):
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            row_to_write = ['experiment_name', 'epoch'] + [loss_name.strip() for loss_name in losses_accu.keys()]
            writer.writerow(row_to_write)
        row_to_write = [experiment_name, epoch] + ['{:.4f}'.format(loss_avg.avg) for loss_avg in losses_accu.values()]
        writer.writerow(row_to_write)

def write_bit_acc(file_name, acc_dict, experiment_name, epoch, write_header=False):
    #print(acc_dict)
    if write_header:
        mode = 'w'
    else: 
        mode = 'a'
    
    with open(file_name, mode, newline='') as csvfile:
        
        writer = csv.writer(csvfile)
        if write_header:
            row_to_write = ['experiment_name', 'epoch'] + [loss_name.strip() for loss_name in acc_dict.keys()]
            writer.writerow(row_to_write)
        row_to_write = [experiment_name, epoch] + ['{:.4f}'.format(sum(loss_avg)/len(loss_avg)) for loss_avg in acc_dict.values()]
        writer.writerow(row_to_write)

def read_message(tau, msg_set, original_msg_length): 
    if tau == int(0):
        message_path = f'./messages/{original_msg_length}/{msg_set}/baseline.txt'
    else: 
        message_path = f'./messages/{original_msg_length}/{msg_set}/message_threshold_{tau}.txt'

    with open(message_path, "r") as f:
        lines = f.readlines()
        lines = list(map(lambda r: r.strip("\n"), lines))

    message = []
    for line in lines: 
        message_here = []
        for char in line: 
            if char == '1':
                message_here.append(1)
            elif char=='0':
                message_here.append(0)
            else: 
                print("Invalid char")
        message.append(message_here)
    message = torch.Tensor(message)
    return message

def calc_bit_acc(decoded_list, tau,msg_set,original_msg_length):
    orig_message_path = \
        f'./messages/{original_msg_length}/{msg_set}/baseline.txt'
    with open(orig_message_path, "r") as f:
        lines = f.readlines()
        lines = list(map(lambda r: r.strip("\n"), lines))
        original_msg = lines[0]
    
    dwm = DatasetWatermark(original_msg, tau / 100)
    sum_acc = 0 
    success = False
    if tau == 0:
        recon_msg  = dwm.compute_bitwise_majority(decoded_list) 
        if recon_msg == dwm.message:
            success = True 
        sum_acc += sum(1 for a, b in zip(recon_msg,dwm.message) if a == b)/float(len(dwm.message))
    else:
        try:
            status, bit_acc = dwm.can_extract_watermark(decoded_list, len(original_msg))
            if status:
                success = True
            sum_acc += bit_acc
        except ExtractionFailure as e:
            pass
    return sum_acc, success

def calc_subset(decoded_list, subsets, tau,msg_set,original_msg_length, trial = 100):
    ba_array = []
    wa_array = []
    for s in subsets:
        sum_ba = 0.
        sum_wa = 0.
        for t in range(trial):
            samples = sample(decoded_list, math.ceil(s * len(decoded_list)))
            ba, wa  =  calc_bit_acc(samples,tau,msg_set,original_msg_length) 
            sum_ba += ba
            sum_wa += wa
        ba_array.append(sum_ba/float(trial))
        wa_array.append(sum_wa/float(trial))
    return ba_array, wa_array      

def calc_psnr(orig_img, wm_img):
    #print(orig_img.shape)
    orig_img = (orig_img + 1) / 2
    orig_img = orig_img.permute(1, 2, 0).cpu().numpy()
    orig_img = (np.clip(orig_img*255.0, 0, 255).astype(np.uint8)).astype(np.float64)

    wm_img = (wm_img + 1) / 2
    wm_img = wm_img.permute(1,2,0).cpu().numpy()
    wm_img = (np.clip(wm_img*255.0, 0, 255).astype(np.uint8)).astype(np.float64)

    #print(wm_img.dtype, orig_img.dtype)
    mse = np.mean((wm_img - orig_img) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    parser = argparse.ArgumentParser(description='Training of HiDDeN nets')
    # parser.add_argument('--size', '-s', default=128, type=int, help='The size of the images (images are square so this is height and width).')
    parser.add_argument('--data-dir', '-d', required=False, type=str, 
                        default="./data/", help='The directory where the data is stored.')
    parser.add_argument('--runs_root', '-r', default=os.path.join('.', 'exp'), type=str,
                        help='The root folder where data about experiments are stored.')
    parser.add_argument('--batch-size', '-b', default=32, type=int, help='Validation batch size.')

    parser.add_argument("--tau", type=int, nargs="+",
                     default=[0,60,80], required=False) #0, 60, 80

    parser.add_argument("--msg_sets", type=int, nargs="+",
                     default=[0,1,2,3,4,5,6,7,8,9], required=False) #0,1,2,3,4,5,6,7,8,9
    
    parser.add_argument("--original_msg_length", type=int, default=30, help="Original message length")
    args = parser.parse_args()


    #completed_runs = ['30bit','16bit','9bit']
    acc_all_attacks = {}
    acc_all_attacks["config"] = []
    acc_all_attacks["tau"] = []
    acc_all_attacks["set"] = []
    acc_all_attacks["bit_acc"] = []
    acc_all_attacks["word_acc"] = []
    acc_all_attacks["bit_acc_subset"] = []
    acc_all_attacks["word_acc_subset"] = []
    all_avg_psnr = {}
    for msg_set in args.msg_sets:    
        write_csv_header = True
        
        for tau in args.tau:
            if tau == int(0):
                run_name = '30bit'
            elif tau == int(60):
                run_name = '16bit'
            elif tau == int(80):
                run_name = '9bit'
            else:
                print("Run name is not supported")
            
            if tau not in all_avg_psnr:
                all_avg_psnr[tau] = []

            message_to_embed = (read_message(tau,msg_set,args.original_msg_length)).to(device)
            current_run = os.path.join(args.runs_root, run_name)
            print(f'Run folder: {current_run}')
            options_file = os.path.join(current_run, 'options-and-config.pickle')
            #train_options, hidden_config, noise_config = utils.load_options(options_file)
            train_options, hidden_config, noise_configs = utils.load_options(options_file)            
            noise_configs.append(Identity())
            for noise_config in noise_configs:
                acc = {}
                acc["tau"] = []
                acc["set"] = []
                acc["bit_acc"] = []
                acc["word_acc"] = []

                
                counter = 0
                decoded_msg_list = []
                noise_config = [noise_config]
                #print(noise_config[0])
                #print(vars(noise_config[0]))
                train_options.train_folder = os.path.join(args.data_dir, 'train')
                train_options.validation_folder = os.path.join(args.data_dir, 'val')
                train_options.batch_size = args.batch_size
                checkpoint, chpt_file_name = utils.load_last_checkpoint(os.path.join(current_run, 'checkpoints'))
                #print(f'Loaded checkpoint from file {chpt_file_name}')
                noiser = Noiser(noise_config,device=device)
                model = Hidden(hidden_config, device, noiser, tb_logger=None)
                utils.model_from_checkpoint(model, checkpoint)

                #print('Model loaded successfully. Starting validation run...')
                _, val_data = utils.get_data_loaders(hidden_config, train_options)
                file_count = len(val_data.dataset)
                if file_count % train_options.batch_size == 0:
                    steps_in_epoch = file_count // train_options.batch_size
                else:
                    steps_in_epoch = file_count // train_options.batch_size + 1

                step = 0
                avg_psnr =[]
                for image, _ in val_data:
                    step += 1
                    image = image.to(device)
                    message = message_to_embed[counter:counter+(image.shape[0])]
                    counter += (image.shape[0])

                    losses, (encoded_images, noised_images, decoded_messages) = model.validate_on_batch([image, message[:image.shape[0]]])
                    decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
                    
                    #PSNR calculation
                    if 'Identity' in str(noise_config[0]):
                        for img_inbatch_id, single_img in enumerate(image):
                            single_wm_img = encoded_images[img_inbatch_id]
                            psnr_single = calc_psnr(single_img,single_wm_img)
                            avg_psnr.append(psnr_single)

                    for one_decoded_rounded in decoded_rounded:
                        decoded_rounded_str = ''
                        for digit in one_decoded_rounded:
                            if digit == 0:
                                decoded_rounded_str = decoded_rounded_str + "0"
                            else: 
                                decoded_rounded_str = decoded_rounded_str + "1"
                        decoded_msg_list.append(decoded_rounded_str)
                    #gather messages
                    #if len(decoded_msg_list)<1:
                    #        decoded_msg_list = decoded_rounded
                    #else:
                    #        decoded_msg_list = np.concatenate((decoded_msg_list, decoded_rounded), axis=0)

                
                #cal psnr for this exp
                if 'Identity' in str(noise_config[0]):
                    psnr_final =  sum(avg_psnr) / float(len(avg_psnr))
                    #print(len(avg_psnr))
                    print(f"Average PSNR of the watermarked images set: {msg_set} and tau: {tau} is:{psnr_final}")
                    all_avg_psnr[tau].append(psnr_final)
                    
                    #apply the subset attack 
                    subsets =  np.linspace(0, 1, num=6)
                    bit_acc_subset, word_acc_subset  = calc_subset(decoded_msg_list,subsets,tau, msg_set,args.original_msg_length)
                else:
                    bit_acc_subset = []
                    word_acc_subset = []


                
                bit_acc, word_acc = calc_bit_acc(decoded_msg_list, tau,msg_set,args.original_msg_length)
                # per set results 
                acc["tau"].append(tau)
                acc["set"].append(msg_set)
                acc["bit_acc"].append(bit_acc)
                acc["word_acc"].append(word_acc)
                # results of all attacks 
                acc_all_attacks["config"].append(str(noise_config[0]))
                acc_all_attacks["tau"].append(tau)
                acc_all_attacks["set"].append(msg_set)
                acc_all_attacks["bit_acc"].append(bit_acc)
                acc_all_attacks["word_acc"].append(word_acc)
                # subsets
                acc_all_attacks["bit_acc_subset"].append(bit_acc_subset)
                acc_all_attacks["word_acc_subset"].append(word_acc_subset)
                
                write_bit_acc(os.path.join(args.runs_root, f'validation_subset_run_{args.original_msg_length}_{msg_set}.csv'), acc, 
                str(noise_config),
                checkpoint['epoch'],
                write_header=write_csv_header)
                write_csv_header = False


    df = pd.DataFrame(acc_all_attacks)
    df_attacks       = df.loc[df['config'] != 'Identity()']
    df_identity      = df.loc[df['config'] == 'Identity()']

    #attack results#
    baseline_attack_avg_ba = df_attacks.loc[df['tau'] == int(0)]['bit_acc'].mean()
    our_60_attack_avg_ba   = df_attacks.loc[df['tau'] == int(60)]['bit_acc'].mean()
    our_80_attack_avg_ba   = df_attacks.loc[df['tau'] == int(80)]['bit_acc'].mean()

    baseline_attack_avg_wa = df_attacks.loc[df['tau'] == int(0)]['word_acc'].mean()
    our_60_attack_avg_wa   = df_attacks.loc[df['tau'] == int(60)]['word_acc'].mean()
    our_80_attack_avg_wa   = df_attacks.loc[df['tau'] == int(80)]['word_acc'].mean()

    #no attack results
    baseline_identity_avg_ba = df_identity.loc[df['tau'] == int(0)]['bit_acc'].mean()
    our_60_identity_avg_ba   = df_identity.loc[df['tau'] == int(60)]['bit_acc'].mean()
    our_80_identity_avg_ba   = df_identity.loc[df['tau'] == int(80)]['bit_acc'].mean()

    baseline_identity_avg_wa = df_identity.loc[df['tau'] == int(0)]['word_acc'].mean()
    our_60_identity_avg_wa   = df_identity.loc[df['tau'] == int(60)]['word_acc'].mean()
    our_80_identity_avg_wa   = df_identity.loc[df['tau'] == int(80)]['word_acc'].mean()

    #subset attack results 
    baseline_subset_avg_ba = np.array(df_identity.loc[df['tau'] == int(0)]['bit_acc_subset'].tolist()).mean(axis=0)
    our_60_subset_avg_ba   = np.array(df_identity.loc[df['tau'] == int(60)]['bit_acc_subset'].tolist()).mean(axis=0)
    our_80_subset_avg_ba   = np.array(df_identity.loc[df['tau'] == int(80)]['bit_acc_subset'].tolist()).mean(axis=0)

    baseline_subset_avg_wa = np.array(df_identity.loc[df['tau'] == int(0)]['word_acc_subset'].tolist()).mean(axis=0)
    our_60_subset_avg_wa   = np.array(df_identity.loc[df['tau'] == int(60)]['word_acc_subset'].tolist()).mean(axis=0)
    our_80_subset_avg_wa   = np.array(df_identity.loc[df['tau'] == int(80)]['word_acc_subset'].tolist()).mean(axis=0)



    #write the summary results 
    #TODO: improve by dictionary
    #print("Robustness against attack average over message sets")
    #print(f"the bit acc for baseline: {baseline_attack_avg_ba}, proposed 60%:\
    #       {our_60_attack_avg_ba}, proposed 80%: {our_80_attack_avg_ba}")
    #print(f"the word acc for baseline: {baseline_attack_avg_wa}, proposed 60%:\
    #       {our_60_attack_avg_wa}, proposed 80%: {our_80_attack_avg_wa}")

    with open(os.path.join(args.runs_root, f'validation_subset_run_{args.original_msg_length}_summary.csv'), 
                  'w', newline='') as csvfile:

        writer = csv.writer(csvfile)
        row_to_write = ['message length', 'epoch', 'tau', 'avg_attack_ba','avg_attack_wa','avg_identity_ba','avg_identity_wa']
        writer.writerow(row_to_write)
        #baseline 
        row_to_write = [args.original_msg_length, checkpoint['epoch'], 0,
                        baseline_attack_avg_ba, baseline_attack_avg_wa,
                        baseline_identity_avg_ba, baseline_identity_avg_wa]
        writer.writerow(row_to_write)
        #proposed tau 60
        row_to_write = [args.original_msg_length, checkpoint['epoch'],60 ,
                        our_60_attack_avg_ba, our_60_attack_avg_wa,
                        our_60_identity_avg_ba, our_60_identity_avg_wa]
        writer.writerow(row_to_write)

        #proposed tau 80
        row_to_write = [args.original_msg_length, checkpoint['epoch'], 80,
                        our_80_attack_avg_ba, our_80_attack_avg_wa,
                        our_80_identity_avg_ba, our_80_identity_avg_wa]
        writer.writerow(row_to_write)
        
        writer.writerow(["==== Summary of the subset attacks"])
        row_to_write = ['message length', 'epoch', 'tau', 'subsets', 'avg_subset_attack_ba','avg_subset_attack_wa']
        writer.writerow(row_to_write)
        row_to_write = [args.original_msg_length, checkpoint['epoch'], 0, subsets, baseline_subset_avg_ba, baseline_subset_avg_wa]
        writer.writerow(row_to_write)
        row_to_write = [args.original_msg_length, checkpoint['epoch'], 60, subsets, our_60_subset_avg_ba, our_60_subset_avg_wa]
        writer.writerow(row_to_write)
        row_to_write = [args.original_msg_length, checkpoint['epoch'], 80, subsets, our_80_subset_avg_ba, our_80_subset_avg_wa]
        writer.writerow(row_to_write)
     
        #write psnr values to the csv file 
        writer.writerow(["==== Summary of PSNR values"])
        #print avg psnr
        for key in all_avg_psnr:
            #print(len(all_avg_psnr[key]))
            #print(f'average psnr for tau: {key}: {sum(all_avg_psnr[key])/float(len(all_avg_psnr[key]))}')
            writer.writerow([f'average psnr for tau: {key}: {sum(all_avg_psnr[key])/float(len(all_avg_psnr[key]))}'])
    csvfile.close()
    
    #plot subset
    subset_ba_array = [baseline_subset_avg_ba, our_60_subset_avg_ba, our_80_subset_avg_ba]
    subset_wa_array = [baseline_subset_avg_wa, our_60_subset_avg_wa, our_80_subset_avg_wa]   
    msg_length = [30,16,9]
    #plot figures
    fig1, ax1 = plt.subplots(nrows=int(1), ncols=1,figsize=(6.6,6.2) )
    fig2, ax2 = plt.subplots(nrows=int(1), ncols=1,figsize=(6.6,6.2) )
    for tau_index, threshold in enumerate(args.tau):
        if threshold==0: 
            line = '--'
            marker = '.'
        elif threshold==60:
            line = '-'
            marker = '^'
        else: 
            line = '-'
            marker = '*'
        ax1.plot(subsets*100,subset_wa_array[tau_index]*100.0,marker+line,
                    label=f'+AMUSE: {msg_length[tau_index]} bits' if threshold !=0 else f'HiDDen:{msg_length[tau_index]} bits',markersize=12, linewidth=3)
        
        ax2.plot(subsets*100,subset_ba_array[tau_index]*100.0,marker+line,
                label=f'+AMUSE: {msg_length[tau_index]} bits' if threshold !=0 else f'HiDDen:{msg_length[tau_index]} bits',markersize=12, linewidth=3)
    ax1.legend(fontsize=26)
    #ax1.yaxis.get_major_locator().set_params(integer=True)

    ax2.legend(fontsize=26)
    #ax2.yaxis.get_major_locator().set_params(integer=True)
    
    ax = ax1
    for item in([ax.title, ax.xaxis.get_label(),  ax.yaxis.get_label()] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(26)

    ax = ax2
    for item in([ax.title, ax.xaxis.get_label(),  ax.yaxis.get_label()] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(26)


    #fig1.suptitle(f'Average over all adversarial attacks') #, threshold:{threshold}                              
    fig1.supxlabel('Dataset Subset (%)',fontsize=26)
    fig1.supylabel('Word Accuracy (%)',fontsize=20)
    fig_summary_dir = f'./plots/'
    fig1.savefig(f'{fig_summary_dir}/subset_no_attack_hidden_WA.png',bbox_inches='tight')


    #fig2.suptitle(f'Average over all adversarial attacks') #, threshold:{threshold}                              
    fig2.supxlabel('Dataset Subset (%)',fontsize=26)
    fig2.supylabel('Bit Accuracy (%)',fontsize=20)
    fig_summary_dir = f'./plots/'
    fig2.savefig(f'{fig_summary_dir}/subset_no_attack_hidden_BA.png',bbox_inches='tight')
    

if __name__ == '__main__':
    main()
