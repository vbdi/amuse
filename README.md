## Code for AMUSE: Adaptive Multi-Segment Encoding for Dataset Watermarking codes
The repo provides the code for AMUSE: Adaptive Multi-Segment Encoding for Dataset Watermarking codes paper. 

AMUSE is accepted to ICME 2025. 

### Setup: 
```
#create a conda env 
#conda create -n amuse  python=3.8.13
#!conda activate amuse
pip install -r ./requirements.txt
```
### AMUSE
The implementation of AMUSE is provided in watermark_chunk_method.py

### Baseline+AMUSE results
run the following command
```
python eval_dataset.py 
```
It will run the code for 10 watermark messages. The average PSNR values over 10 messages will be printed at the end. The average bit accuracy results will be saved in the ./exp/validation_run_30_summary.csv.


### Subset attack
run the following command
```
python eval_dataset_subset.py 
```
It will run the code for 10 watermark messages. The average accuracy results for the subset attack expeirments will be saved in ./plots/.

### Refrences: 
AMUSE is built based on the following repo https://github.com/ando-khachatryan/HiDDeN/.


### Citation: 
```
@inproceedings{DBLP:conf/icmcs/AlvarB24,
  author       = {Saeed Ranjbar Alvar and Mohammad Akbari and David Ming Xuan Yue and Yong Zhang},
  title        = {AMUSE: Adaptive Multi-Segment Encoding for
Dataset Watermarking},
  booktitle    = {{IEEE} International Conference on Multimedia and Expo, {ICME} 2025},
  publisher    = {{IEEE}},
  year         = {2025},
}
```



