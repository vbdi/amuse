from random import sample, choice
from math import ceil
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
from functools import lru_cache
from glob import glob
from os.path import join
from shutil import move
from random import choice, sample, shuffle,random
import os
import time

class ExtractionFailure(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class DatasetWatermark:

    def __init__(self, message: str, guarantee: float) -> None:
        self.N: int = None
        self.chunks_per_message: int = None
        self.message: str = message
        self.total_unique_number_of_messages: int = None
        self.guarantee = guarantee
        self.actual_guarantee: float = None

        self.choose_nk()

    def get_chunk_combination_indicies(self):
        indicies = list(range(self.N))
        return list(combinations(indicies, self.chunks_per_message))

    def encode_ordering_bits(self, c: int):

        bits_required = self.total_unique_number_of_messages.bit_length()
        c_bin = bin(c)[2:]

        if len(c_bin) < bits_required:
            c_bin = "0" * (bits_required - len(c_bin)) + c_bin
        return c_bin

    def extract_ordering_bits(self, bitstring: int):
        bits_required = self.total_unique_number_of_messages.bit_length()

        order = bitstring[:bits_required]
        body = bitstring[bits_required:]

        return int(order, 2), body

    def choose_nk_david(self):

        def padding(l, n):

            return (n - l % n) % n

        optimal_pair = (float('inf'), 1)

        for n in range(1, len(self.message) + 1):
            for k in range(1, n):

                if k / n > self.guarantee:
                    continue

                pad_len = padding(len(self.message), n)

                if (n / k) > (len(self.message) + pad_len) / (self.choose(n, n - k).bit_length() + pad_len):
                    continue

                o_n, o_k = optimal_pair

                if (n / k) < (o_n / o_k):
                    optimal_pair = (n, k)

        n, k = optimal_pair

        if n == float('inf'):
            self.N = 1
            self.chunks_per_message = 1
            self.actual_guarantee = 0
            self.total_unique_number_of_messages = 1
        else:
            self.N = n
            self.chunks_per_message = n - k
            self.actual_guarantee = k / n
            self.total_unique_number_of_messages = self.choose(
                self.N, self.chunks_per_message)

    def choose_nk(self):

        def padding(l, n):

            return (n - l % n) % n

        optimal_pair = (float('inf'), 1)
        optimal_l = len(self.message)
        max_chunk = 100
        before_nk_selection = time.time()
        for n in range(1, max_chunk + 1):
            for k in range(1, n):

                if k / n > self.guarantee:
                    continue

                pad_len = padding(len(self.message), n)
                
                # (n-k)/n*L + ordering bits
                sub_message_length =  (n-k)* (len(self.message) + pad_len) / n + (self.choose(n, n - k).bit_length())
                if sub_message_length > len(self.message):
                    continue

                if sub_message_length<optimal_l:
                    optimal_pair = (n, k)
                    optimal_l = sub_message_length

        n, k = optimal_pair
        latency = time.time() - before_nk_selection
        print(f"latency for n, k selection is {latency*1000.0}")

        if n == float('inf'):
            self.N = 1
            self.chunks_per_message = 1
            self.actual_guarantee = 0
            self.total_unique_number_of_messages = 1
        else:
            self.N = n
            self.chunks_per_message = n - k
            self.actual_guarantee = k / n
            self.total_unique_number_of_messages = self.choose(
                self.N, self.chunks_per_message)

    def generate_watermark_chunks(self, bitstring: str):

        if self.N > len(bitstring):
            raise ValueError(
                "number of bits in message cannot exceed number of chunks")

        if self.N == 1:
            return [bitstring]

        bitstring = "0" * ((self.N - len(bitstring) %
                           self.N) % self.N) + bitstring

        chunk_len = len(bitstring) // self.N
        message_chunks = [bitstring[i: i + chunk_len]
                          for i in range(0, len(bitstring), chunk_len)]

        orderings = self.get_chunk_combination_indicies()

        embed_chunks = []
        for c, order in enumerate(orderings):
            embed_chunks.append(self.encode_ordering_bits(c) +
                                "".join(message_chunks[i] for i in order))

        return embed_chunks

    def compute_bitwise_majority(self, bitstrings: 'list[str]'):

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

    def split_submessage_body_to_chunks(self, submessage_body: str):
        embed_chunk_len = len(submessage_body) // self.chunks_per_message

        chunks = [submessage_body[i: i + embed_chunk_len]
                  for i in range(0, len(submessage_body), embed_chunk_len)]

        return chunks

    def decode_submessage(self, submessage: str):
        order, body = self.extract_ordering_bits(submessage)
        return (order, self.split_submessage_body_to_chunks(body))

    def recover_watermark_from_embed_chunks(
            self, submessages: 'list[str]', original_message_len: int):

        chunk_map = dict()

        if len(submessages) == 1 and len(submessages[0]) == original_message_len:
            return submessages[0]

        for chunk in submessages:
            orderings = self.get_chunk_combination_indicies()
            order, chunks = self.decode_submessage(chunk)

            # #don't append the entry with wrong order
            if not (order> (len(orderings) - 1)):
                #order = choice(np.arange(len(orderings))) #order_max -1 

                order_c = orderings[order]

                for i, c in zip(order_c, chunks):

                    if i not in chunk_map:
                        chunk_map[i] = []

                    chunk_map[i].append(c)

        message_bitstring = ""

        for i in range(0, self.N):

            if i not in chunk_map:
                raise ExtractionFailure("Cannot extract watermark, missing chunk %d" % i)

            message_bitstring += self.compute_bitwise_majority(chunk_map[i])

        message_bitstring = message_bitstring[len(
            message_bitstring) - len(self.message):]


        return message_bitstring



        actual = self.recover_watermark_from_embed_chunks_kmeans(
            leaked_submessages, original_message_len)
        return actual == self.message

    def can_extract_watermark(
            self, leaked_submessages: 'list[str]', original_message_len: int):

        actual = self.recover_watermark_from_embed_chunks(
            leaked_submessages, original_message_len)
        count = sum(1 for a, b in zip(actual, self.message) if a == b)
        count /= float(len(self.message))
        #print(count)
        #input("check the error")
        return  actual == self.message,count

    def calculate_guarantee_percentage(self, chunks: 'list[str]'):

        k = self.choose(self.N - 1, self.chunks_per_message)

        freq = dict()

        minFreq = float("-inf")

        for c in chunks:

            freq[c] = freq.get(c, 0) + 1

        chunks.sort(key=lambda r: freq[r], reverse=True)

        counts = list(map(lambda r: freq[r], chunks))

        return sum(counts[:k]) / len(chunks)

    @lru_cache
    def factorial(self, n):
        v = 1
        while n > 0:
            v *= n
            n = n - 1
        return v

    def choose(self, n, r):

        return self.factorial(n) // (self.factorial(r) * self.factorial(n - r))

    def run_watermarking_sim(
            self,
            messages: 'list[str]',
            message_length: int,
            guarantee: float):

        self.choose_nk(message_length, guarantee)

        self.run_sim(message_length, messages)

    def run_watermark_sim_full(self, image_path: str, message_length: int):

        amount = len(glob(join(image_path, "0/*")))
        embed_message_len = self.watermark_images(message_length, amount)

        out_files = glob("src/ssl_watermarking/output/imgs/*.png")

        for f in out_files:

            move(f, "src/ssl_watermarking/input")

        messages = self.extract_watermarks(embed_message_len)

        self.run_watermarking_sim(messages, message_length)

    def guarantee_plot(self, message_length: int):

        def embed_ratio(p: int):
            r = self.N.bit_length()

            k = ceil(p / 100 * self.N)

            return (r + (1 - k / self.N) * (message_length + (self.N -
                    message_length % self.N) % self.N)) / message_length

        xs = []
        ys = []

        for p in range(1, 100 + 1, 1):

            if p != 1 and p % 10 != 0:
                continue

            xs.append(p)
            ys.append(embed_ratio(p) * 100)

        plt.figure(figsize=(7, 7))
        plt.scatter(xs, ys, s=24)
        plt.grid()
        plt.title("Embed/Original Ratio vs Guarantee %")
        plt.xlabel("Guarantee %")
        plt.ylabel("Embed/Original Ratio")
        plt.xticks(np.arange(0, max(xs) + 2, 10))
        plt.yticks(np.arange(0, 120, 10))

        txt = """
        Message Length: %d bits
        Chunks: %d
        """ % (
            message_length, self.N)

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        # place a text box in upper left in axes coords
        # plt.text(0.60, 0.90, txt, transform=plt.gca().transAxes, fontsize=9,
        #          verticalalignment='top', bbox=props)

        plt.savefig("guarantee.png")

    def generate_messages(self, amount: int):
        unique_chunks = self.generate_watermark_chunks(self.message)

        unique_chunks *= (amount // len(unique_chunks)) + 1

        unique_chunks = unique_chunks[:amount]

        return unique_chunks

    def run_sim(self, messages: 'list[str]', original_message_len: int):

        amount = len(messages)
        print("Total Amount of Samples: %d" % amount)
        print("Total number of unique chunks: %d" % self.N)
        print("Number of chunks per sub message: %d" % self.chunks_per_message)
        print("Guarantee Percentage: > %f %%" %
              (self.guarantee * 100))

        trials = 10000

        amounts_x = []
        rate_y = []
        max_leak_percentage = 50
        for leak_percentage in range(1, max_leak_percentage, 1):

            success = 0
            total = 0
            leak_percentage = leak_percentage / 100

            for i in range(trials):

                leaked = sample(messages, ceil(amount * leak_percentage))

                if self.can_extract_watermark(leaked, original_message_len):
                    success += 1

                total += 1

            amounts_x.append(leak_percentage * 100)
            rate_y.append(success / total)

            print(
                "%d samples (~%f %%) -> Successful Extraction Rate: %f" %
                (ceil(
                    amount *
                    leak_percentage),
                    ceil(
                    amount *
                    leak_percentage) /
                    amount *
                    100,
                    success /
                    total))

        # plt.figure(figsize=(12,12))
        plt.figure(figsize=(7, 7))
        plt.tight_layout()
        plt.plot(amounts_x, rate_y, label="Simulation")
        plt.grid()
        plt.title("Watermark Extraction Accuracy")

        bit_reduction = (len(self.message) -
                         len(messages[0])) / len(self.message) * 100

        txt = """
        Chunks: %d
        Chunks per message: %d
        Original message: %d bits
        Embedded message: %d bits
        Bit reduction: %.2f %%
        Trials: %d
        Total samples: %d
        """ % (self.N,
               self.chunks_per_message,
               len(self.message),
               len(messages[0]),
               bit_reduction,
               trials,
               amount)

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        plt.text(0.60, 0.90, txt, transform=plt.gca().transAxes, fontsize=9,
                 verticalalignment='top', bbox=props)

        plt.xlabel("Samples leaked %")
        plt.ylabel("Successful Extraction Rate")

        plt.xticks(
            np.arange(
                0, max(
                    max_leak_percentage, self.guarantee * 100) + 10, 5))

        plt.yticks(np.arange(max(0, min(rate_y) - 0.05), 1.05, 0.05))
        plt.axvline(x=self.guarantee * 100, color="red",
                    label="Expected Extraction Guarantee")
        plt.axvline(x=self.actual_guarantee * 100, color="green",
                    label="Actual Extraction Guarantee")
        plt.legend()
        plt.show()
        plt.savefig("wm.png")

    def sample_sim(self, amount: int):

        self.choose_nk()

        unique_chunks = self.generate_messages(amount)

        self.run_sim(unique_chunks, len(self.message))

def prepare_message(message_length,n_sets, n_samples=100):
    for set in np.arange(n_sets):
        original_message = ''
        for l in np.arange(message_length):
            r = random() > 0.5
            if r:
                original_message = original_message+'1'
            else:
                original_message = original_message+'0'
        

        # if not os.path.exists(f'ssl_watermarking/users/{message_length}/{set}/'):
        #     os.makedirs(f'ssl_watermarking/users/{message_length}/{set}/')

        # with open(f"ssl_watermarking/users/{message_length}/{set}/baseline.txt", "w") as f:
        #     f.write("\n".join([original_message] * n_samples))

        #runtime for message encoding can be obtained here
        for t in [20, 40, 60, 80, 100]:
            before_encoding = time.time()
            dwm = DatasetWatermark(original_message, t/100)
            messages = dwm.generate_messages(n_samples)
            latency_encoding = time.time() - before_encoding
            print(f"Latency for encoding is {latency_encoding*1000.0} seconds")
            print(dwm.N, dwm.chunks_per_message)

            # with open(f'ssl_watermarking/users/{message_length}/{set}/message_threshold_{t}.txt', "w") as f:
            #     f.write("\n".join(messages))


if __name__ == '__main__':

    message_length = 300
    n_sets = 10
    n_samples = 9144
    prepare_message(message_length,n_sets,n_samples)

        