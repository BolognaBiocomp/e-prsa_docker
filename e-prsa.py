import sys
import gc
import numpy as np
import torch
from torch import nn
from transformers import T5EncoderModel, T5Tokenizer
import re
import esm
from collections import OrderedDict

BATCH_SIZE = 15000
INPUT_FILE, OUTPUT_FILE = sys.argv[1], sys.argv[2]
PROT_T5_MODEL = "/srv/data/biocomp/plm/prot_t5_xl_uniref50/"
ESM2_MODEL = "/srv/data/biocomp/plm/esm2/esm2_t33_650M_UR50D.pt"
ESM1B_MODEL = "/srv/data/biocomp/plm/esm1b/esm1b_t33_650M_UR50S.pt"
DEEPREX2_MODEL = "/home/biocomp/deeprex2/deeprex2.model.ckpt"
ISPREDSEQ_MODEL = "/home/biocomp/deeprex2/ispred-seq.model.pth"
EMB_SIZE = 1280+1024
WING = 15

class DeepRExNN(nn.Module):
	def __init__(self, IN_SIZE, WING, H1_SIZE = 64, H2_SIZE = 16, DROPOUT = 0.1):
		super().__init__()
		#Define nn layers
		self.cnn = nn.Conv1d(
			IN_SIZE,
			IN_SIZE,
			WING*2+1,
			padding='valid',
			groups=IN_SIZE
		)
		self.linear = nn.Sequential(
			nn.Linear(IN_SIZE,H1_SIZE),
			nn.Dropout(p=DROPOUT),
			nn.ReLU(),
			nn.Linear(H1_SIZE,H2_SIZE),
			nn.Dropout(p=DROPOUT),
			nn.ReLU(),
			nn.Linear(H2_SIZE,1)
		)

	def forward(self, x):
		x = self.cnn(x)
		x = torch.flatten(x,start_dim=1)
		x = self.linear(x)
		x = torch.flatten(x)
		return x

class IspredSeqNN(nn.Module):
	def __init__(self, INPUT_SIZE, WING, H1_SIZE = 128, H2_SIZE = 32, DROPOUT=0.5):
		super(IspredSeqNN, self).__init__()
		self.cnn = nn.Conv1d(INPUT_SIZE,INPUT_SIZE,WING*2+1,padding='valid',groups=INPUT_SIZE)
		self.l1 = nn.Linear(INPUT_SIZE,H1_SIZE)
		self.l2 = nn.Linear(H1_SIZE,H2_SIZE)
		self.l3 = nn.Linear(H2_SIZE,1)
		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()
		self.dropout = nn.Dropout(p=DROPOUT)

	def forward(self, x):
		x = self.cnn(x)
		x = torch.flatten(x,start_dim=1)
		
		x = self.l1(x)
		x = self.dropout(x)
		x = self.relu(x)
		
		x = self.l2(x)
		x = self.dropout(x)
		x = self.relu(x)
		
		x = self.l3(x)
		x = torch.flatten(x)
		return x

def embed_prot_t5(sequences, model, tokenizer):
	if len(sequences) == 0:
		return
	
	#Compute output
	seqs = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequences]
	ids = tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding="longest")
	input_ids = torch.tensor(ids['input_ids'])
	attention_mask = torch.tensor(ids['attention_mask'])
	with torch.no_grad():
		embedding_repr = model(input_ids=input_ids,attention_mask=attention_mask)
	
	#Construct list with embeddings
	lengths = [len(sequence) for sequence in sequences]
	ret = []
	for i in range(len(sequences)):
		emb = embedding_repr.last_hidden_state[i,:lengths[i]]
		ret.append(emb.detach().cpu().numpy())
	return ret

def embed_esm(sequences, seq_ids, model, alphabet):
	if len(sequences) == 0:
		return
	
	#Separate sequences longer than 1022
	_sequences, _seq_ids, map_splits = [], [], []
	for fasta,id in zip(sequences, seq_ids):
		n_splits = int(np.ceil(len(fasta)/1022))
		len_splits = int(np.ceil(len(fasta)/n_splits))
		seq_splits = [fasta[len_splits*i:len_splits*(i+1)] for i in range(n_splits)]
		for i,seq in enumerate(seq_splits):
			_sequences.append(seq)
			_seq_ids.append(id+'_'+str(i))
		map_splits.append(n_splits)
	
	#Compute outputs
	model.eval()
	batch_converter = alphabet.get_batch_converter()
	data = list(zip(_seq_ids, _sequences))
	batch_labels, batch_strs, batch_tokens = batch_converter(data)
	batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
	with torch.no_grad():
		results = model(batch_tokens, repr_layers=[33], return_contacts=False)
	token_representations = results["representations"][33]
	
	#Construct list with embeddings, putting together split sequences
	ret = []
	idx = 0
	offset = -1
	temp = []
	for i, tokens_len in enumerate(batch_lens):
		temp.append(token_representations[i, 1 : tokens_len - 1].detach().cpu().numpy())
		if i == map_splits[idx] + offset:
			ret.append(np.concatenate(temp))
			temp = []
			offset += map_splits[idx]
			idx += 1
	return ret

def concatenate_padded(protT5, esm2, WING, EMB_SIZE):
	Xs = []
	Xs_padder = np.zeros((WING,EMB_SIZE))
	
	for a,b in zip(protT5, esm2):
		embs = np.concatenate([Xs_padder,np.concatenate([a,b],1),Xs_padder])
		for i in range(WING, len(embs)-WING):
			Xs.append(np.transpose(embs[i-WING:i+WING+1]))
	Xs = torch.from_numpy(np.array(Xs)).float() #double()
	#print(sys.getsizeof(Xs) + Xs.element_size() * Xs.nelement())
	return Xs

try:
	#READ SEQUENCES
	sequences = []
	seq_ids = []

	with open(INPUT_FILE) as reader:
		for line in reader:
			seq_ids.append(line[1:].strip().split()[0])
			sequences.append(reader.readline().strip())

	#LOAD MODELS
	model_t5 = T5EncoderModel.from_pretrained(PROT_T5_MODEL)
	tokenizer = T5Tokenizer.from_pretrained(PROT_T5_MODEL)

	model_esm2, alphabet2 = esm.pretrained.load_model_and_alphabet(ESM2_MODEL)
	
	model_esm1b, alphabet1b = esm.pretrained.load_model_and_alphabet(ESM1B_MODEL)

	model_dr2 = DeepRExNN(EMB_SIZE, WING)
	model_dr2.load_state_dict(torch.load(DEEPREX2_MODEL)["state_dict"])
	model_dr2.eval()
	model_dr2.float() #double()

	model_is = IspredSeqNN(EMB_SIZE, WING)
	model_is.load_state_dict(torch.load(ISPREDSEQ_MODEL))
	model_is.eval()
	model_is.float() #double()
	sigmoid = nn.Sigmoid()

	#PROCESS BATCHES
	i = 0
	_sequences = []
	_seq_ids = []
	count = 0
	max_len = 0
	while True:
		if (i < len(sequences)) and ((count + 1) * max(len(sequences[i]), max_len) <= BATCH_SIZE):
			_sequences.append(sequences[i])
			_seq_ids.append(seq_ids[i])
			count += 1
			max_len = max(len(sequences[i]), max_len)
			i+=1
		else:
			#GET EMBEDDINGS AND PREDICTIONS
			protT5 = embed_prot_t5(_sequences, model_t5, tokenizer)
			esm2 = embed_esm(_sequences, _seq_ids, model_esm2, alphabet2)
			Xs = concatenate_padded(protT5, esm2, WING, EMB_SIZE)
			del esm2
			gc.collect()
			pred = model_dr2(Xs)
			del Xs
			gc.collect()
			
			esm1b = embed_esm(_sequences, _seq_ids, model_esm1b, alphabet1b)
			Xs = concatenate_padded(esm1b, protT5, WING, EMB_SIZE)
			del esm1b
			del protT5
			gc.collect()
			pred_is = sigmoid(model_is(Xs))
			del Xs
			gc.collect()
			
			class_pred = ['Exposed' if p>=0.2 else 'Buried' for p in pred]
			class_pred_is = ['IS' if (p>=0.2 and p2>=0.5) else 'N' for p,p2 in zip(pred,pred_is)]


			#WRITE RESULTS
			ids_list = []
			res_list = []
			idx_list = []
			for seq_id, sequence in zip(_seq_ids, _sequences):
				for idx, res in enumerate(sequence, 1):
					ids_list.append(seq_id)
					res_list.append(res)
					idx_list.append(idx)

			with open(OUTPUT_FILE,'a') as writer:
				for id, idx, r, c, p, cis in zip(ids_list, idx_list, res_list, class_pred, pred, class_pred_is):
					print(
						id,
						str(idx),
						r,
						str(round(float(p),2)),
						c,
						cis,
						sep='\t',
						file=writer
					)

			_sequences = []
			_seq_ids = []
			count = 0
			max_len = 0
			del pred, pred_is, class_pred, class_pred_is, ids_list, res_list, idx_list, seq_id, sequence, res, writer, id, idx, r, c, p, cis
			gc.collect()
			if i >= len(sequences):
				break
except:
#	raise
	sys.exit(1)
sys.exit(0)
