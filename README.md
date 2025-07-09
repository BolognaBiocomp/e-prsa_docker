# E-pRSA

Predictor of residue solvent accessibility.

E-pRSA is a deep learning-based tool that predicts the relative solvent accessibility (RSA) of each residue in a protein sequence using protein language models (PLMs) such as ProtT5 and ESM2.

---

## Requirements

E-pRSA requires Docker Engine to be installed.

To install Docker Engine on a Debian-based system, follow the official instructions:  
[https://docs.docker.com/engine/install/debian/](https://docs.docker.com/engine/install/debian/)

**Hardware requirements**:
- At least 4 CPU cores
- At least 48 GB RAM  
  (necessary to load large pre-trained protein language models)

> **Important:** After installing Docker, make sure your current user has permission to run Docker commands without `sudo`.  
> You can do this by adding your user to the `docker` group:

```bash
sudo usermod -aG docker $USER
```

Then, run:
```
newgrp docker
```
to apply changes to group configuration.

---

## Installation

### 1. Create and activate a conda environment

```bash
conda create -n eprsa-docker python
conda activate eprsa-docker
```

### 2. Install required Python packages

```bash
pip install docker absl-py
```

### 3. Clone the repository and move into the project directory

```bash
git clone https://github.com/MatteoManfredi/e-prsa_docker.git
cd e-prsa_docker
```

### 4. Build the Docker image

```bash
docker build -t eprsa:1.0 .
```

### 5. Download pre-trained models (ProtT5 and ESM2)

```bash
cd ~
wget https://e-prsa.biocomp.unibo.it/main/download_plms/e-prsa-plms.tar.gz
tar xvzf e-prsa-plms.tar.gz
```

This will create a `e-prsa-plms/` directory containing the required models. Please be aware that the models require approximately 40 GB of free disk space.

---

## Run E-pRSA

To run the predictor, use the `run_docker.py` script in the root directory. You must provide:

- a FASTA input file
- an output file path
- the path to the pre-trained models directory

### Example

```bash
cd e-prsa_docker
python run_docker.py \
  --fasta_file=example-data/Q96KB5.fasta \
  --output_file=example-data/Q96KB5.tsv \
  --plm_dir=${HOME}/e-prsa-plms
```

### Optional: Adjusting batch size for performance

You can customize the number of padded residues processed per batch using the `--batch_size` option:

    python run_docker.py \
      --fasta_file=example-data/Q96KB5.fasta \
      --output_file=example-data/Q96KB5.tsv \
      --plm_dir=${HOME}/e-prsa-plms \
      --batch_size=8000

**Default value:** `--batch_size=1000`

The batch size is computed as:

    batch_size = number_of_sequences × length_of_longest_sequence_in_batch

Larger values improve execution speed but require more memory.

Use the table below as a reference for tuning `--batch_size` based on your available RAM:

| Available RAM | Recommended `--batch_size` |
|---------------|-----------------------------|
| 8 GB          | 2000                        |
| 16 GB         | 4000                        |
| 32 GB         | 8000                        |
| 64 GB         | 15000                       |
| 128 GB        | 30000                       |

> 💡 *Tip:* Monitor memory usage during the first run and adjust accordingly.



---

## Output Format

The output is a tab-separated values (TSV) file with one row per residue.

Example:

```
sp|Q96KB5|TOPK_HUMAN	1	M	0.41	Exposed
sp|Q96KB5|TOPK_HUMAN	2	E	0.53	Exposed
sp|Q96KB5|TOPK_HUMAN	3	G	0.55	Exposed
sp|Q96KB5|TOPK_HUMAN	4	I	0.37	Exposed
sp|Q96KB5|TOPK_HUMAN	5	S	0.49	Exposed
sp|Q96KB5|TOPK_HUMAN	6	N	0.59	Exposed
sp|Q96KB5|TOPK_HUMAN	7	F	0.29	Exposed
```

### Column description

| Column        | Description                                        |
|---------------|----------------------------------------------------|
| **ID**        | Protein accession (from FASTA header)              |
| **Position**  | Residue index (1-based)                            |
| **Residue**   | Amino acid (1-letter code)                         |
| **RSA Score** | Predicted relative solvent accessibility (0 to 1) |
| **Label**     | Discrete label: `Exposed` or `Buried`             |

---

## Citation

If you use **E-pRSA** in your work, please cite:

> Manfredi M, Savojardo C, Martelli PL, Casadio R. E-pRSA: Embeddings Improve the Prediction of Residue Relative Solvent Accessibility in Protein Sequence. J Mol Biol. 2024 Sep 1;436(17):168494. doi: 10.1016/j.jmb.2024.168494. 

---

## Contact

For questions or issues, please contact:

**Biocomputing Group – University of Bologna**  
📧 [matteo.manfredi4@unibo.it](mailto:matteo.manfredi4@unibo.it)  
🌐 [https://biocomp.unibo.it](https://biocomp.unibo.it)

---

## License

This project is distributed under the **MIT License**.  
See the [`LICENSE`](./LICENSE) file for details.
