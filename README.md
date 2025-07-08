# E-pRSA
Predictor of residue solvent accessibility

## Requirements

E-pRSA requires Docker Engine to be installed. Please, follow this instructions
for the installation of Docker Engine on a Debian system:

https://docs.docker.com/engine/install/debian/

E-pRSA requires the loading of large pre-trained protein language models (ProtT5 and ESM2)
in memory. We suggest to run E-pRSA on a machine with at least 4 CPU cores and 48GB of RAM.

## Installation

Create a conda environment with Python 3:

```
conda create -n eprsa-docker python
conda activate eprsa-docker
```

Install dependencies using pip:

```
pip install docker absl-py
```

Clone this repo and cd into the package dir:

```
git clone https://github.com/BolognaBiocomp/e-prsa_docker
cd e-prsa_docker
```

Build the Docker image:

```
docker build -t eprsa:1.0 .
```

Download the ESM and ProtT5 pLMs (e.g. on ${HOME}):

```
cd
wget https://coconat.biocomp.unibo.it/static/data/coconat-plms.tar.gz
tar xvzf coconat-plms.tar.gz
```

## Run E-pRSA

To run the program use the run_docker.py script inside the
e-prsa_docker root directory, providing a FASTA file an output file, and the path
where ESM2 and ProtT5 pLMs are stored, as follows:

```
cd e-prsa_docker
python run_docker.py --fasta_file=example-data/example.fasta \
--output_file=example-data/example.tsv --plm_dir=${HOME}/coconat-plms
```

The output file (example-data/example.tsv) looks like the following:

```
sp|Q96KB5|TOPK_HUMAN	1	M	0.41	Exposed
sp|Q96KB5|TOPK_HUMAN	2	E	0.53	Exposed
sp|Q96KB5|TOPK_HUMAN	3	G	0.55	Exposed
sp|Q96KB5|TOPK_HUMAN	4	I	0.37	Exposed
sp|Q96KB5|TOPK_HUMAN	5	S	0.49	Exposed
sp|Q96KB5|TOPK_HUMAN	6	N	0.59	Exposed
sp|Q96KB5|TOPK_HUMAN	7	F	0.29	Exposed

```

You have one row for each residues, and columns are defined as follows:

* ID: protein accession, as reported in the input FASTA file
* ...
