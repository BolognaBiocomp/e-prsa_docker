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
git clone https://github.com/BolognaBiocomp/e-prsa_docker
cd e-prsa_docker
```

### 4. Build the Docker image

```bash
docker build -t eprsa:1.0 .
```

### 5. Download pre-trained models (ProtT5 and ESM2)

```bash
cd ~
wget https://coconat.biocomp.unibo.it/static/data/coconat-plms.tar.gz
tar xvzf coconat-plms.tar.gz
```

This will create a `coconat-plms/` directory containing the required models.

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
  --plm_dir=${HOME}/coconat-plms
```

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

> _E-pRSA: deep learning-based residue solvent accessibility prediction using protein language models_  
> [Biocomputing Group – University of Bologna](https://biocomp.unibo.it)  
> DOI or preprint coming soon.

---

## Contact

For questions or issues, please contact:

**Biocomputing Group – University of Bologna**  
📧 [biocomp@unibo.it](mailto:biocomp@unibo.it)  
🌐 [https://biocomp.unibo.it](https://biocomp.unibo.it)

---

## License

This project is distributed under the **MIT License**.  
See the [`LICENSE`](./LICENSE) file for details.
