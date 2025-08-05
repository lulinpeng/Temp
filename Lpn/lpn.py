import os
import numpy as np
import math
from Crypto.Cipher import AES
import logging
import json
import argparse

class Lpn:
    def __init__(self):
        logging.basicConfig(format='%(asctime)s.%(msecs)03d [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        logging.getLogger().setLevel(logging.INFO)
        self.logger = logging.getLogger()
        self.BYTE_BITS = 8 # byte bits
        self.k = 1024 # secret key bits
        self.alpha = 0.4 * 1 / np.sqrt(self.k) # noise rate
        self.kappa = 32 # seed bytes
        self.n = 1024
        self.t = 15 # for repetition coding
        self.meta_bit_len = self.BYTE_BITS
        self.s = self.gen_noise(self.k) # random secret key
        self.e = self.gen_noise(self.n) # noise vector
        self.expansion_ratio = self.k * self.t + 1
        
        self.seed = os.urandom(self.kappa) # random seed
        self.gen_coeff_matrix() # coefficient matrix A
        
        self.logger.debug(f'A = {self.A}, {self.A.shape}, sum = {self.A.sum()}')
        self.logger.debug(f's = {self.s}, {self.s.shape}, sum = {self.s.sum()}')
        self.logger.debug(f'e = {self.e}, {self.e.shape}, sum = {self.e.sum()}')
        self.b = (self.s @ self.A % 2 + self.e) % 2 # b = A s + e
        self.logger.debug(f'b = {self.b}, {self.b.shape}, sum = {self.b.sum()}')
        return
    
    def load_secret_key(self, sk_hex:str):
        self.decompress_sk(sk_hex)
        return
    
    def load_public_key(self, pk_hex:str):
        seed_hex_len = self.kappa * 2
        b_hex_len = self.n // 4
        seed_hex, b_hex = pk_hex[:seed_hex_len], pk_hex[seed_hex_len: seed_hex_len+b_hex_len]
        self.seed = self.hex_to_bytes(seed_hex)
        b_bits = self.hex_to_bits(b_hex)
        self.b = np.array(b_bits, dtype=np.uint8)
        return
        
    def hex_to_bytes(self, hex_str:str):
        bits_str = self.hex_to_bits_str(hex_str)
        bits = self.bits_str_to_bits(bits_str)
        bytes = self.bits_to_bytes(bits)
        return bytes
    
    def hex_to_bits(self, hex_str:str):
        bits_str = self.hex_to_bits_str(hex_str)
        bits = self.bits_str_to_bits(bits_str)
        return bits
    
    def hex_to_bits_str(self, hex_str:str):
        table = {'0': '0000', '1': '0001', '2': '0010', '3': '0011', '4': '0100', '5': '0101', '6': '0110', '7': '0111', 
                 '8': '1000', '9': '1001', 'a': '1010', 'b': '1011', 'c': '1100', 'd': '1101', 'e': '1110', 'f': '1111'}
        bits_str = ''.join([table[h] for h in hex_str])
        return bits_str
    
    def bits_str_to_bits(self, bits_str:str):
        ''' convert bit string into a list of bit each represented by an integer'''
        return [int(bit) for bit in bits_str]
    
    def bits_to_hex(self, bits:list):
        bits_str = ''.join([str(bit) for bit in bits])
        return self.bits_str_to_hex(bits_str)
    
    def bits_str_to_hex(self, bits:str):
        table = {'0000': '0', '0001': '1', '0010': '2', '0011': '3', '0100': '4', '0101': '5', '0110': '6', '0111': '7', 
                 '1000': '8', '1001': '9', '1010': 'a', '1011': 'b', '1100': 'c', '1101': 'd', '1110': 'e', '1111': 'f'}
        hex_str = ''
        for i in range(0, len(bits), 4):
            key = bits[i:i+4]
            hex_str += (table[key] if key in table else '')
        return hex_str
    
    def prg(self, num_bits:int):
        cipher = AES.new(self.seed, AES.MODE_CTR, nonce=b'')
        num_bytes = (num_bits + 7) // 8
        rand_bytes = cipher.encrypt(b'\x00' * num_bytes)
        bits = []
        for byte in rand_bytes:
            bits += [(byte >> i) & 1 for i in range(8)]
        return bits[:num_bits]

    def bytes_to_bits(self, data:bytes):
        bits = []
        for byte in data:
            bits += [(byte >> i) & 1 for i in range(8)]
        return bits
    
    def bits_to_bytes(self, bits:list):
        n = len(bits) // self.BYTE_BITS
        t = b''
        for i in range(n):
            start = i * self.BYTE_BITS
            end = (i + 1) * self.BYTE_BITS
            bits_str = ''.join(str(bit) for bit in bits[start:end])
            t += int(bits_str, 2).to_bytes(1, 'big')
        return t
    
    def decompress_sk(self, sk_hex:str):
        sk_hex = sk_hex.lower()
        self.logger.debug(f'len(sk_hex) = {len(sk_hex)}')
        sk_bits = bin(int(sk_hex, 16))[2:].zfill(len(sk_hex) * 4)
        self.logger.debug(sk_bits)
        meta = int(sk_bits[:self.meta_bit_len], 2)
        self.logger.debug(f'meta = {meta}')
        idx_bit_len = math.ceil(math.log2(self.k))
        assert(meta == ((len(sk_bits) - self.meta_bit_len) // idx_bit_len))
        s = [0] * self.k
        for i in range(meta):
            start = self.meta_bit_len + i*idx_bit_len
            end = self.meta_bit_len + (i+1)*idx_bit_len
            idx = int(sk_bits[start:end], 2)
            s[idx] = 1
        self.s = np.array(s, dtype=np.uint8)
        self.logger.debug(f's = {self.s}, {self.s.shape}, sum = {self.s.sum()}')
        return

    def compress_sk(self):
        '''meta | idxes | padding'''
        if self.s is None:
            return None
        idx_bit_len = math.ceil(math.log2(self.k))
        meta_bits = bin(int(self.s.sum()))[2:].zfill(self.meta_bit_len)
        idxes = self.s.nonzero()[0].tolist()
        self.logger.debug(f'idxes = {idxes}')
        idxes_bits = ''.join([bin(idx)[2:].zfill(idx_bit_len) for idx in idxes])
        compressed_sk_bits = meta_bits + idxes_bits
        self.logger.debug(f'compressed sk bits: {compressed_sk_bits}')
        padding_bit_len = (self.BYTE_BITS - len(compressed_sk_bits) % self.BYTE_BITS) % self.BYTE_BITS
        self.logger.debug(f'padding_bit_len = {padding_bit_len}')
        compressed_sk_bits += '0' * padding_bit_len
        self.logger.debug(f'len(compressed_sk_bits) = {len(compressed_sk_bits)}')
        return self.bits_str_to_hex(compressed_sk_bits)
    
    def save_params(self, outfile:str=None):
        outfile = 'params_lpn.txt' if outfile is None else outfile
        brief = self.brief()
        with open(outfile, 'w') as f:
            f.write(json.dumps(brief))
        return
    
    def brief(self):
        seed_hex = self.seed.hex()
        self.logger.debug(f'sum = {self.s.sum()}')
        # s_hex = ''.join([str(s) for s in self.s.tolist()])
        s_hex = self.compress_sk()
        self.decompress_sk(s_hex)
        self.logger.debug(f's = {self.s}')
        self.logger.debug(f'b = {self.b}, {self.b.shape}, sum = {self.b.sum()}')
        b_hex = self.bits_to_bytes(self.b.tolist()).hex()
        pk_hex = seed_hex + b_hex
        return {'n':self.n, 'k':self.k, 'alpha':float(self.alpha), 'kappa':self.kappa, 'expansion_ratio':self.expansion_ratio, 'sk':s_hex, 'pk':pk_hex, 't':self.t}
    
    def gen_rand_bits(self, n:int, m:int=None):
        total = n if m is None else n * m
        bits = np.random.randint(0, 2, total, dtype=np.uint8)
        self.logger.debug(f'bits = {bits}, {type(bits)}')
        bits = bits if m is None else np.array(bits, dtype=np.uint8).reshape(n, m)
        return bits
    
    def gen_noise(self, n:int, m:int=None):
        total = n if m is None else n * m
        noise = (np.random.rand(total) < self.alpha).astype(np.uint8)
        noise = noise if m is None else np.array(noise, dtype=np.uint8).reshape(n, m) 
        return noise
    
    def encode(self, msg:bytes):
        msg_bits = self.bytes_to_bits(msg)
        self.logger.info(f'msg bits: {msg_bits} {msg}')
        encoded_msg_bits = []
        for bit in msg_bits:
            encoded_msg_bits += [bit] * self.t
        self.logger.info(f'encoded msg bits: {encoded_msg_bits}')
        return np.array(encoded_msg_bits, dtype=np.uint8).astype(np.uint8)
    
    def decode(self, d:list):
        self.logger.debug(f'type(d) = {type(d)}, {d}')
        n = len(d) // self.t
        if n * self.t != len(d):
            error_msg = f'decode: {len(d)} should be divided by {self.t}'
            self.logger.debug(error_msg)
            raise BaseException(error_msg)
        dd = []
        for i in range(n):
            s, t = i * self.t, (i+1)*self.t
            self.logger.debug(f'{d[s:t]}, {sum(d[s:t])/self.t}')
            dd.append(str(int(sum(d[s:t])/self.t > 0.5)))
        return ''.join(dd)
    
    def encrypt(self, msg:bytes, outfile:str=None):
        outfile = 'cipher_lpn.txt' if outfile is None else outfile
        mm = self.encode(msg)
        self.logger.info(f'mm: {mm.shape}, sum = {mm.sum()}')
        t = mm.shape[0]
        S = self.gen_noise(self.n, t)
        E = self.gen_noise(self.k, t)
        self.logger.info(f'A: {self.A.shape}, sum = {self.A.sum()}')
        self.logger.info(f'S: {S.shape}, sum = {S.sum()}')
        self.logger.info(f'E: {E.shape}, sum = {E.sum()}')
        C1 = (self.A @ S % 2 + E) % 2 # C1 = A S + E
        self.logger.info(f'C1: {C1.shape}, sum = {C1.sum()}')
        self.logger.info(f'b: {self.b.shape}, sum = {self.b.sum()}')
        C2 = self.b @ S + mm  # C2 = b S + m'
        self.logger.info(f'C2: {C2.shape}, sum = {C2.sum()}')
        C1_bits = C1.flatten().tolist()
        C2_bits = C2.flatten().tolist()
        self.logger.info(f'{len(C1_bits)}, {len(C2_bits)}')
        C_bits = C1_bits + C2_bits
        C_hex = self.bits_to_hex(C_bits)
        with open(outfile, 'w') as f:
            f.write(C_hex)
        print(f'cipher file: {outfile}')
        # self.logger.debug(f's.shape = {self.s.shape}, E.shape = {E.shape}')
        # e0 = self.s @ E % 2
        # e1 = self.e @ S % 2
        # self.e0 = e0
        # self.e1 = e1
        # self.logger.debug(f'e0 = {e0}, {e0.shape}, sum = {e0.sum()}')
        # self.logger.debug(f'e1 = {e1}, {e1.shape}, sum = {e1.sum()}')
        # e = (e0 + e1) % 2
        # self.logger.debug(f'e = e0 + e1 = {e}, {e.shape}, sum = {e.sum()}')
        return C_hex
    
    def decrypt_from_file(self, infile:str):
        with open(infile) as f:
            cipher_hex_str = f.read()
        C_bits = self.hex_to_bits(cipher_hex_str)
        l = len(C_bits) // ((self.k + 1) * self.t)
        print(f'len(cipher_hex_str) = {len(cipher_hex_str)}, len(C_bits)  = {len(C_bits)}, k = {self.k}, t = {self.t}, l = {l}')
        return
    
    def decrypt(self, cipher:tuple):
        C1, C2 = cipher
        d = (self.s @ C1 % 2 + C2) % 2
        self.logger.debug(f'd = {d}')
        d = d.tolist()
        dd = self.decode(d)
        self.logger.debug(f'dd = {dd}')
        return dd
    
    def gen_sk(self):
        return
        
    def gen_coeff_matrix(self):
        """seed -> A"""
        total_bits = self.n * self.k
        a = self.prg(total_bits)
        self.A = np.array(a, dtype=np.uint8).reshape(self.k, self.n)
        return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LPN asymmetric encryption')
    subparsers = parser.add_subparsers( dest="command", title="available commands", metavar="command")
    parser_encrypt = subparsers.add_parser("encrypt", help="encrypt a message", description="encrypt a message")
    parser_encrypt.add_argument('--hexmsg', type=str, help='message to be encoded', required=True)
    parser_encrypt.add_argument('--pk', type=str, help='hex string of the public key', required=True)
    parser_encrypt.add_argument('--outfile', type=str, default=None, help='path of cipher')
    
    parser_decrypt = subparsers.add_parser("decrypt", help="encode file into QR Code images", description="encode file into QR Code images")
    parser_decrypt.add_argument('--infile', type=str, help='ciphertext file', required=True)
    parser_decrypt.add_argument('--sk', type=str, help='hex string of the secret key', required=True)

    args = parser.parse_args()
    
    if not hasattr(args, "command") or args.command is None:
        parser.print_help()
    if args.command == 'encrypt':
        print(f'+++++ encrypt +++++')
        print(f'message = {args.hexmsg}, pk = {args.pk}\n')
        lpn = Lpn()
        lpn.load_public_key(args.pk)
        msg_bytes = lpn.hex_to_bytes(args.hexmsg)
        cipher = lpn.encrypt(msg_bytes, outfile=args.outfile)
    elif args.command == 'decrypt':
        print(f'+++++ decrypt +++++')
        lpn = Lpn()
        lpn.load_secret_key(args.sk)
        lpn.decrypt_from_file(args.infile)
