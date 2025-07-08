import argparse
import gc
import numpy as np
import torch
from torch import nn
from transformers import T5EncoderModel, T5Tokenizer
from transformers.utils import logging as hf_logging
import re
import esm
import sys
from datetime import datetime
from eprsalib import eprsacfg as ecfg

class EpRSANN(nn.Module):
    def __init__(self, IN_SIZE, WING, H1_SIZE=64, H2_SIZE=16, DROPOUT=0.1):
        """Initialize the EpRSANN model architecture."""
        super().__init__()
        self.cnn = nn.Conv1d(IN_SIZE, IN_SIZE, WING * 2 + 1, padding='valid', groups=IN_SIZE)
        self.linear = nn.Sequential(
            nn.Linear(IN_SIZE, H1_SIZE),
            nn.Dropout(p=DROPOUT),
            nn.ReLU(),
            nn.Linear(H1_SIZE, H2_SIZE),
            nn.Dropout(p=DROPOUT),
            nn.ReLU(),
            nn.Linear(H2_SIZE, 1)
        )

    def forward(self, x):
        """Run forward pass."""
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        x = torch.flatten(x)
        return x

class EpRSA:
    def __init__(self, input_file, output_file, batch_size=15000, verbosity=2):
        """Initialize EpRSA with paths, constants, and models."""
        try:
            # Set command line parameters
            self.input_file = input_file
            self.output_file = output_file
            self.batch_size = batch_size
            self.verbosity = verbosity

            # Set model paths and constants
            self.PROT_T5_MODEL = f"{ecfg.EPRSA_PLM_DIR}/prot_t5_xl_uniref50/"
            self.ESM2_MODEL = f"{ecfg.EPRSA_PLM_DIR}/esm2_t33_650M_UR50D.pt"
            self.EPRSA_MODEL = ecfg.EPRSA_MODEL
            self.EMB_SIZE = 1280 + 1024
            self.WING = 15

            # Load models
            hf_logging.set_verbosity_error() # Suppress hf warning
            self.model_t5 = T5EncoderModel.from_pretrained(self.PROT_T5_MODEL)
            self.tokenizer = T5Tokenizer.from_pretrained(self.PROT_T5_MODEL)
            self.model_esm2, self.alphabet2 = esm.pretrained.load_model_and_alphabet(self.ESM2_MODEL)
            self.model_eprsa = EpRSANN(self.EMB_SIZE, self.WING)
            self.model_eprsa.load_state_dict(torch.load(self.EPRSA_MODEL)["state_dict"])
            self.model_eprsa.eval()
            self.model_eprsa.float()
        except Exception as e:
            handle_error(f"Error while loading the models: {e}", verbosity)

        if self.verbosity >= 2:
            print("All models loaded.")

    def run_batches(self):
        """Main loop."""
        # Read sequences
        try:
            seq_ids, sequences = self._read_fasta(self.input_file)
        except Exception as e:
            handle_error(f"Error reading FASTA: {e}", self.verbosity)

        if self.verbosity >= 2:
            print(f"{len(seq_ids)} sequences loaded, starting predictions.")

        # Reset output file
        open(self.output_file, 'w').close()

        # Process batches
        try:
            for i, (batch_ids, batch_seqs) in enumerate(self._generate_batches(seq_ids, sequences), 1):
                self._predict(batch_ids, batch_seqs, i)
        except Exception as e:
            handle_error(f"Error during prediction: {e}", self.verbosity)

        if self.verbosity >= 2:
            print(f"Predictions for {len(seq_ids)} sequences done. You can find the output in the file {self.output_file}")

    def _read_fasta(self, filename):
        """Parse a multi-FASTA file."""
        seq_ids, sequences = [], []
        with open(filename) as reader:
            current_id, current_seq = None, ""
            for line in reader:
                line = line.strip()
                if line.startswith(">"):
                    if current_id is not None:
                        seq_ids.append(current_id)
                        sequences.append(current_seq)
                    current_id = line[1:].split()[0]
                    current_seq = ""
                else:
                    current_seq += line
            if current_id is not None:
                seq_ids.append(current_id)
                sequences.append(current_seq)
        return seq_ids, sequences

    def _generate_batches(self, seq_ids, sequences):
        """Yield batches of sequences without exceeding the maximum number of padded residues."""
        batch_ids = []
        batch_seqs = []
        max_len = 0

        for seq_id, seq in zip(seq_ids, sequences):
            proposed_max_len = max(max_len, len(seq))
            proposed_batch_size = proposed_max_len * (len(batch_seqs) + 1)

            if proposed_batch_size <= self.batch_size:
                # Add to current batch
                batch_ids.append(seq_id)
                batch_seqs.append(seq)
                max_len = proposed_max_len
            else:
                # Yield current batch
                yield batch_ids, batch_seqs
                # Start new batch
                batch_ids = [seq_id]
                batch_seqs = [seq]
                max_len = len(seq)
                
        # Yield final batch
        if batch_ids:
            yield batch_ids, batch_seqs

    def _predict(self, seq_ids, sequences, batch_num):
        """Run the full prediction pipeline and write results to the file."""
        if self.verbosity >= 2:
            print(f"Batch {batch_num} ({len(seq_ids)} sequences) . . .", end="", flush=True)

        # Embed sequences
        protT5 = self._embed_prot_t5(sequences)
        esm2 = self._embed_esm(sequences, seq_ids)

        # Concatenate
        Xs = self._concatenate_padded(protT5, esm2)
        del esm2, protT5
        gc.collect()
        
        # Predict
        pred = self.model_eprsa(Xs)
        del Xs
        gc.collect()

        # Write results to output file
        with open(self.output_file, 'a') as writer:
            pred_idx = 0
            for seq_id, seq in zip(seq_ids, sequences):
                for idx, res in enumerate(seq, 1):
                    p = float(pred[pred_idx])
                    c = 'Exposed' if p >= 0.2 else 'Buried'
                    print(seq_id, str(idx), res, str(round(p, 2)), c, sep='\t', file=writer, flush=True)
                    pred_idx += 1
        del pred
        gc.collect()

        if self.verbosity >= 2:
            print(" Done!")

    def _embed_prot_t5(self, sequences):
        """Compute ProtT5 embeddings for a list of sequences."""
        if len(sequences) == 0:
            return
        # Replace uncommon residues and prepare inputs
        seqs = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequences]
        ids = self.tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding="longest")
        input_ids = torch.tensor(ids['input_ids'])
        attention_mask = torch.tensor(ids['attention_mask'])

        # Run model inference
        with torch.no_grad():
            embedding_repr = self.model_t5(input_ids=input_ids, attention_mask=attention_mask)

        # Collect variable-length embeddings
        lengths = [len(sequence) for sequence in sequences]
        ret = [embedding_repr.last_hidden_state[i, :lengths[i]].detach().cpu().numpy() for i in range(len(sequences))]
        return ret

    def _embed_esm(self, sequences, seq_ids):
        """Compute ESM2 embeddings for a list of sequences."""
        if len(sequences) == 0:
            return

        # Handle sequences longer than 1022 tokens by splitting
        _sequences, _seq_ids, map_splits = [], [], []
        for fasta, id in zip(sequences, seq_ids):
            n_splits = int(np.ceil(len(fasta) / 1022))
            len_splits = int(np.ceil(len(fasta) / n_splits))
            seq_splits = [fasta[len_splits * i:len_splits * (i + 1)] for i in range(n_splits)]
            for i, seq in enumerate(seq_splits):
                _sequences.append(seq)
                _seq_ids.append(id + '_' + str(i))
            map_splits.append(n_splits)

        # Prepare batch and run model
        self.model_esm2.eval()
        batch_converter = self.alphabet2.get_batch_converter()
        data = list(zip(_seq_ids, _sequences))
        _, _, batch_tokens = batch_converter(data)
        batch_lens = (batch_tokens != self.alphabet2.padding_idx).sum(1)

        with torch.no_grad():
            results = self.model_esm2(batch_tokens, repr_layers=[33], return_contacts=False)

        # Reconstruct full embeddings for original sequences
        token_representations = results["representations"][33]
        ret, idx, offset, temp = [], 0, -1, []
        for i, tokens_len in enumerate(batch_lens):
            temp.append(token_representations[i, 1:tokens_len - 1].detach().cpu().numpy())
            if i == map_splits[idx] + offset:
                ret.append(np.concatenate(temp))
                temp = []
                offset += map_splits[idx]
                idx += 1
        return ret

    def _concatenate_padded(self, protT5, esm2):
        """Combine ProtT5 and ESM2 embeddings and apply padding."""
        Xs = []
        Xs_padder = np.zeros((self.WING, self.EMB_SIZE))
        for a, b in zip(protT5, esm2):
            embs = np.concatenate([Xs_padder, np.concatenate([a, b], 1), Xs_padder])
            for i in range(self.WING, len(embs) - self.WING):
                Xs.append(np.transpose(embs[i - self.WING:i + self.WING + 1]))
        return torch.from_numpy(np.array(Xs)).float()

def handle_error(message, verbosity):
    """Handle errors according to verbosity level."""
    if verbosity == 0:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("error-log.txt", "a") as log:
            log.write(f"[{timestamp}] {message}\n")
    else:
        print(message, file=sys.stderr)
    sys.exit(1)

def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="Run E-pRSA on a multifasta file.")
    parser.add_argument("input_file", help="Input FASTA file with sequences.")
    parser.add_argument("output_file", help="Output file to write results.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=15000,
        help=("Maximum number of padded residues to process in a batch (default: 15000). This is calculated as the number of sequences in a batch times the length of the longest sequence in that batch. Higher values improve speed but require more RAM. Suggested values based on available RAM: 2,000 for 8 GB, 4,000 for 16 GB, 8,000 for 32 GB, 15,000 for 64 GB, 30,000 for 128 GB. Monitor memory usage on first run to adjust if needed.")
    )
    parser.add_argument("--verbosity", type=int, default=2, choices=[0,1,2], help="Verbosity level (Default: 2): 0=silent (log errors to local file), 1=errors only, 2=progress messages.")
    return parser.parse_args()

def main():
    """Main function."""
    try:
        args = parse_args()
        runner = EpRSA(args.input_file, args.output_file, args.batch_size, args.verbosity)
        runner.run_batches()
    except Exception as e:
        handle_error(f"Fatal error: {e}", args.verbosity)


if __name__ == "__main__":
    main()
