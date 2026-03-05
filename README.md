# Reproducibility Instructions

This repository contains the scripts and configurations to reproduce the main results presented in our paper.

install opacus:(version 1.5.2)
pip install git+https://github.com/pytorch/opacus.git@53b3c25432bb75dd92b22131d02fcdc39c8dbe5f

## 🔧 Script–Result Mapping

| Script Name         | Corresponding Figure/Table in Paper     | Description                                      |
|---------------------|------------------------------------------|--------------------------------------------------|
| `run.sh`            | **Figure 2**                             | Runs the main experiment pipeline for baseline results visualized in Figure 2. |
| `run_account.sh`    | **Table 2**                              | Executes experiments for communication/computation overhead statistics shown in Table 2. |
| `run_account1.sh`   | **Table 5**                              | Runs setting 1 of the ablation study contributing to Table 5. |
| `run_account2.sh`   | **Table 5**                              | Runs setting 2 of the ablation study contributing to Table 5. |

## 📌 Usage

Make the shell scripts executable and run them as follows:

```bash
chmod +x run.sh run_account.sh run_account1.sh run_account2.sh
./run.sh
./run_account.sh
./run_account1.sh
./run_account2.sh
