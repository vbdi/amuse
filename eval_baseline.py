import os
import pprint
import argparse
import torch
import numpy as np
import utils
import csv

from model.hidden import Hidden
from noise_layers.noiser import Noiser
from noise_layers.identity import Identity
from random import choice
import pandas as pd
import math
import matplotlib.pyplot as plt
plt.tight_layout()


def read_message(msg_set, original_msg_length): 
    message_path = f'./messages/{original_msg_length}/{msg_set}/baseline.txt'

    with open(message_path, "r") as f:
        lines = f.readlines()
        lines = list(map(lambda r: r.strip("\n"), lines))
    lines = lines[0]

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
    message = torch.t(message)
    return message, lines

def compute_bitwise_majority(bitstrings: 'list[str]'):

    bit_freq = dict()

    for chunk in bitstrings:

        for i, c in enumerate(chunk):

            if c == '0':
                bit_freq[i] = bit_freq.get(i, 0) - 1
            elif c == '1':
                bit_freq[i] = bit_freq.get(i, 0) + 1

    new_chunk = ""

    keys = list(bit_freq.keys())

    keys.sort()
    for k in keys:

        if bit_freq[k] < 0:
            new_chunk += "0"
        elif bit_freq[k] > 0:
            new_chunk += "1"
        else:
            new_chunk += choice(["0", "1"])

    return new_chunk

def calc_bit_acc(decoded_list, msg_str):
    original_msg = msg_str
    sum_acc = 0 
    success = False
    recon_msg  = compute_bitwise_majority(decoded_list) 
    if recon_msg == msg_str:
        success = True 
    sum_acc += sum(1 for a, b in zip(recon_msg,msg_str) if a == b)/float(len(msg_str))
    return sum_acc, success

def write_bit_acc(file_name, acc_dict, experiment_name, epoch, write_header=False):
    print(acc_dict)
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
    parser.add_argument("--msg_sets", type=int, nargs="+",
                     default=[0,1,2,3,4,5,6,7,8,9], required=False)
    args = parser.parse_args()

    completed_runs = ['30bit','16bit','9bit'] #'30bit','16bit','9bit'
    #to store the results 
    bit_acc_all = {}
    bit_std_all = {}
    word_acc_all = {}
    word_std_all = {}
    for run_name in completed_runs:
       
        if run_name not in bit_acc_all:
            bit_acc_all[run_name] = {}
            bit_acc_all[run_name]['no attack'] = []
            bit_acc_all[run_name]['attack'] = []

            bit_std_all[run_name] = {}
            bit_std_all[run_name]['no attack'] = []
            bit_std_all[run_name]['attack'] = []
        
        if run_name not in word_acc_all:
            word_acc_all[run_name] = {}
            word_acc_all[run_name]['no attack'] = []
            word_acc_all[run_name]['attack'] = []

            word_std_all[run_name] = {}
            word_std_all[run_name]['no attack'] = []
            word_std_all[run_name]['attack'] = []
        
        write_csv_header = True

        acc_all_attacks = {}
        acc_all_attacks["config"] = []
        acc_all_attacks["set"] = []
        acc_all_attacks["bit_acc"] = []
        acc_all_attacks["word_acc"] = []

        for msg_set in args.msg_sets:
            message = None
            current_run = os.path.join(args.runs_root, run_name)
            print(f'Run folder: {current_run}')
            options_file = os.path.join(current_run, 'options-and-config.pickle')
            train_options, hidden_config, noise_configs = utils.load_options(options_file)
            noise_configs.append(Identity())
            for noise_config in noise_configs:
                acc = {}
                acc["set"] = []
                acc["bit_acc"] = []
                acc["word_acc"] = []

                noise_config = [noise_config]
                decoded_msg_list = []
                print(noise_config[0])
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
                for image, _ in val_data:
                    step += 1
                    image = image.to(device)
                    if message is None:
                        print(f"Message length is {hidden_config.message_length}")
                        #message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], hidden_config.message_length))).to(device)
                        message, message_str = (read_message(msg_set,hidden_config.message_length))
                        message = message.to(device)

                    #losses, (encoded_images, noised_images, decoded_messages) = model.validate_on_batch([image, message[:image.shape[0]]])
                    losses, (encoded_images, noised_images, decoded_messages) = model.validate_on_batch(
                        [image, message.expand(image.shape[0],len(message_str))])

                    decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
                    for one_decoded_rounded in decoded_rounded:
                        decoded_rounded_str = ''
                        for digit in one_decoded_rounded:
                            if digit == 0:
                                decoded_rounded_str = decoded_rounded_str + "0"
                            else: 
                                decoded_rounded_str = decoded_rounded_str + "1"
                        decoded_msg_list.append(decoded_rounded_str)

                bit_acc, word_acc = calc_bit_acc(decoded_msg_list, message_str)
                acc["set"].append(msg_set)
                acc["bit_acc"].append(bit_acc)
                acc["word_acc"].append(word_acc)

                acc_all_attacks["config"].append(str(noise_config[0]))
                acc_all_attacks["set"].append(msg_set)
                acc_all_attacks["bit_acc"].append(bit_acc)
                acc_all_attacks["word_acc"].append(word_acc)   

                write_bit_acc(os.path.join(args.runs_root, f'validation_baseline_{run_name}.csv'), acc, 
                str(noise_config),
                checkpoint['epoch'],
                write_header=write_csv_header)
                write_csv_header = False
        
        df = pd.DataFrame(acc_all_attacks)
        df_attacks      = df.loc[df['config'] != 'Identity()']
        df_identity      = df.loc[df['config'] == 'Identity()']

        #Attack results
        baseline_attack_avg_ba = df_attacks['bit_acc'].mean() * 100.
        baseline_attack_std_ba = df_attacks['bit_acc'].std() * 100.
        baseline_attack_avg_wa = df_attacks['word_acc'].mean() * 100.
        baseline_attack_std_wa = df_attacks['word_acc'].std() * 100.

        #no attack results
        baseline_identity_avg_ba = df_identity['bit_acc'].mean() * 100.
        baseline_identity_std_ba = df_identity['bit_acc'].std() * 100.
        baseline_identity_avg_wa = df_identity['word_acc'].mean() * 100.
        baseline_identity_std_wa = df_identity['word_acc'].std() * 100.

        with open(os.path.join(args.runs_root, f'validation_baseline_{run_name}.csv'), 
                  'a', newline='') as csvfile:

            writer = csv.writer(csvfile)
            row_to_write = ['message length', 'epoch', 'avg_attack_ba','avg_attack_wa','avg_identity_ba','avg_identity_wa']
            writer.writerow(row_to_write)
            row_to_write = [hidden_config.message_length, checkpoint['epoch'], 
                            baseline_attack_avg_ba, baseline_attack_avg_wa,
                            baseline_identity_avg_ba, baseline_identity_avg_wa]
            writer.writerow(row_to_write)
        csvfile.close()
        
        #attack results
        bit_acc_all[run_name]["attack"].append(baseline_attack_avg_ba)
        bit_std_all[run_name]["attack"].append(baseline_attack_std_ba)

        word_acc_all[run_name]["attack"].append(baseline_attack_avg_wa)
        word_std_all[run_name]["attack"].append(baseline_attack_std_wa)

        #no attack
        bit_acc_all[run_name]["no attack"].append(baseline_identity_avg_ba)
        bit_std_all[run_name]["no attack"].append(baseline_identity_std_ba)

        word_acc_all[run_name]["no attack"].append(baseline_identity_avg_wa)
        word_std_all[run_name]["no attack"].append(baseline_identity_std_wa)

    msg_length = [30,16,9]
    fig1, ax1 = plt.subplots(figsize=(9,6))
    acc1 = []
    std1= []
    for key in bit_acc_all:
        acc1.append(bit_acc_all[key]["no attack"])
        std1.append(bit_std_all[key]["no attack"])
    acc1 = np.array(acc1).flatten()
    std1 = np.array(std1).flatten()
    ax1.plot(msg_length,acc1,'^-',markersize=25, linewidth=3)
    #ax1.fill_between(msg_length, acc1+std1, acc1-std1, facecolor='blue', alpha=0.1)
    #ax1.set_title("No attack")
    ax1.set_xlabel("Message Length (bits)")
    ax1.set_ylabel("Bit Accuracy (%)")
    ax = ax1
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(36)
    plt.tight_layout()
    fig1.savefig('./plots/baselines_ba_no_attack_hidden.png',bbox_inches='tight')   # save the figure to file


    fig2, ax2 = plt.subplots(figsize=(9,6))
    acc2 = []
    std2= []
    for key in bit_acc_all:
        acc2.append(bit_acc_all[key]["attack"][0])
        std2.append(bit_std_all[key]["attack"])
    acc2 = np.array(acc2).flatten()
    std2 = np.array(std2).flatten()

    ax2.plot(msg_length,acc2,'^-',markersize=25, linewidth=3)
    #ax2.fill_between(msg_length, acc2+std2, acc2-std2, facecolor='blue', alpha=0.1)
    #ax2.set_title("Attack")
    ax2.set_xlabel("Message Length (bits)")
    ax2.set_ylabel("Bit Accuracy (%)")
    ax = ax2
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(36)
    plt.tight_layout()
    fig2.savefig('./plots/baselines_ba_attack_hidden.png',bbox_inches='tight')

    fig3, ax3 = plt.subplots(figsize=(9,6))
    acc3 = []
    std3= []
    for key in word_acc_all:
        acc3.append(word_acc_all[key]["no attack"])
        std3.append(word_std_all[key]["no attack"])
    acc3 = np.array(acc3).flatten()
    std3 = np.array(std3).flatten()
    ax3.plot(msg_length,acc3,'^-',markersize=25, linewidth=3)
    #ax3.fill_between(msg_length, acc3+std3, acc3-std3, facecolor='blue', alpha=0.1)
    ax3.set_title("No attack")
    ax3.set_xlabel("Message Length (bits)",y=-0.05)
    ax3.set_ylabel("Word Accuracy (%)",x=-0.01)
    ax = ax3
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(36)
    plt.tight_layout()
    fig3.savefig('./plots/baselines_wa_no_attack_hidden.png',bbox_inches='tight')

    fig4, ax4 = plt.subplots(figsize=(9,6))
    acc4 = []
    std4= []
    for key in word_acc_all:
        acc4.append(word_acc_all[key]["attack"])
        std4.append(word_std_all[key]["attack"])
    acc4 = np.array(acc4).flatten()
    std4 = np.array(std4).flatten()
    ax4.plot(msg_length,acc4,'^-',markersize=25, linewidth=3)
    #ax4.fill_between(msg_length, acc4+std4, acc4-std4, facecolor='blue', alpha=0.1)
    ax4.set_title("Attack")
    ax4.set_xlabel("Message Length (bits)",y=-0.05)
    ax4.set_ylabel("Word Accuracy (%)",x=-0.01)
    ax = ax4
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(36)
    plt.tight_layout()
    fig4.savefig('./plots/baselines_wa_attack_hidden.png',bbox_inches='tight')


if __name__ == '__main__':
    main()  