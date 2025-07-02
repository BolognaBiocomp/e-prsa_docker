import argparse
import gc
import numpy as np
import torch
from torch import nn
from transformers import T5EncoderModel, T5Tokenizer
import re
import esm
import sys

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
    def __init__(self, input_file, output_file, batch_size=15000):
        """Initialize EpRSA with paths, constants, and models."""
        # Set input/output/batch parameters
        self.input_file = input_file
        self.output_file = output_file
        self.batch_size = batch_size

        # Set hardcoded model paths and constants
        self.PROT_T5_MODEL = "/mnt/fat3/data/manfredi/plm/prot_t5_xl_uniref50/"
        self.ESM2_MODEL = "/mnt/fat3/data/manfredi/plm/esm2_t33_650M_UR50D.pt"
        self.EPRSA_MODEL = "/mnt/mini3/work/manfredi/E-pRSA/deeprex2.model.ckpt"
        self.EMB_SIZE = 1280 + 1024
        self.WING = 15

        # Load models
        self.model_t5 = T5EncoderModel.from_pretrained(self.PROT_T5_MODEL)
        self.tokenizer = T5Tokenizer.from_pretrained(self.PROT_T5_MODEL)
        self.model_esm2, self.alphabet2 = esm.pretrained.load_model_and_alphabet(self.ESM2_MODEL)
        self.model_eprsa = EpRSANN(self.EMB_SIZE, self.WING)
        self.model_eprsa.load_state_dict(torch.load(self.EPRSA_MODEL)["state_dict"])
        self.model_eprsa.eval()
        self.model_eprsa.float()

    def run_batches(self):
        """Main loop to batch and process input sequences."""
        try:
            # Read input sequences and IDs from the FASTA file
            seq_ids, sequences = self._read_fasta(self.input_file)
        except Exception as e:
            print(f"Error reading FASTA: {e}", file=sys.stderr)
            sys.exit(1)

        # Overwrite output file
        with open(self.output_file, 'w') as writer:
            pass

        # Initialize counters and buffers
        i = 0
        _sequences = []
        _seq_ids = []
        count = 0
        max_len = 0

        while True:
            # Accumulate sequences in the current batch until it reaches size limit
            if (i < len(sequences)) and ((count + 1) * max(len(sequences[i]), max_len) <= self.batch_size):
                _sequences.append(sequences[i])
                _seq_ids.append(seq_ids[i])
                count += 1
                max_len = max(len(sequences[i]), max_len)
                i += 1
            else:
                # Perform prediction and output writing for current batch
                try:
                    self._predict(_seq_ids, _sequences)
                except Exception as e:
                    print(f"Error during prediction: {e}", file=sys.stderr)
                    sys.exit(1)
                # Reset batch buffers
                _sequences = []
                _seq_ids = []
                count = 0
                max_len = 0
                if i >= len(sequences):
                    break

    def _read_fasta(self, filename):
        """Parse a multi-FASTA file, handling multiline sequences."""
        sequences, seq_ids = [], []
        with open(filename) as reader:
            current_seq, current_id = [], None
            for line in reader:
                line = line.strip()
                if line.startswith(">"):
                    if current_id is not None:
                        sequences.append("".join(current_seq))
                    current_id = line[1:].split()[0]
                    seq_ids.append(current_id)
                    current_seq = []
                else:
                    current_seq.append(line)
            if current_id is not None:
                sequences.append("".join(current_seq))
        return seq_ids, sequences

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
        """Combine ProtT5 and ESM2 embeddings and apply wing padding."""
        Xs = []
        Xs_padder = np.zeros((self.WING, self.EMB_SIZE))
        for a, b in zip(protT5, esm2):
            embs = np.concatenate([Xs_padder, np.concatenate([a, b], 1), Xs_padder])
            for i in range(self.WING, len(embs) - self.WING):
                Xs.append(np.transpose(embs[i - self.WING:i + self.WING + 1]))
        return torch.from_numpy(np.array(Xs)).float()

    def _predict(self, seq_ids, sequences):
        """Run the full prediction pipeline and write results to file."""
        # Embed sequences
        protT5 = self._embed_prot_t5(sequences)
        esm2 = self._embed_esm(sequences, seq_ids)

        # Concatenate and pass to neural network
        Xs = self._concatenate_padded(protT5, esm2)
        del esm2, protT5
        gc.collect()

        pred = self.model_eprsa(Xs)
        del Xs
        gc.collect()

        # Threshold predictions into exposed/buried classes
        class_pred = ['Exposed' if p >= 0.2 else 'Buried' for p in pred]

        # Prepare final output entries
        ids_list, res_list, idx_list = [], [], []
        for seq_id, sequence in zip(seq_ids, sequences):
            for idx, res in enumerate(sequence, 1):
                ids_list.append(seq_id)
                res_list.append(res)
                idx_list.append(idx)

        # Write results to output file
        with open(self.output_file, 'a') as writer:
            for id, idx, r, c, p in zip(ids_list, idx_list, res_list, class_pred, pred):
                print(id, str(idx), r, str(round(float(p), 2)), c, sep='\t', file=writer, flush=True)

        del pred, class_pred, ids_list, res_list, idx_list, seq_id, sequence, res, writer, id, idx, r, c, p
        gc.collect()


def main():
    """Parse arguments and run E-pRSA pipeline."""
    parser = argparse.ArgumentParser(description="Run E-pRSA on a multifasta file.")
    parser.add_argument("input_file", help="Input FASTA file with sequences.")
    parser.add_argument("output_file", help="Output file to write results.")
    parser.add_argument("--batch_size", type=int, default=15000, help="Maximum sequence batch size. Default: 15000")
    args = parser.parse_args()

    try:
        runner = EpRSA(args.input_file, args.output_file, args.batch_size)
        runner.run_batches()
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
