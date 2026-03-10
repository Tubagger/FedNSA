# Reproducibility Instructions

This repository contains the scripts and configurations to reproduce the main results presented in our paper.

install opacus:(version 1.5.2)
pip install opacus==1.5.2

## 🔧 Script–Result Mapping

| Script Name         | Corresponding Figure/Table in Paper     | Description                                      |
|---------------------|------------------------------------------|--------------------------------------------------|
| `run.sh`            | **Figure 2**                             | Runs the main experiment pipeline for baseline results visualized in Figure 2. |
| `run_account.sh`    | **Table 3,4,5**                              | Executes experiments for communication/computation overhead statistics shown in Table 3,4,5. |
| `run_account1.sh`   | **Table 6**                              | Runs setting 1 of the ablation study contributing to Table 6. |
| `run_account2.sh`   | **Table 7**                              | Runs setting 2 of the ablation study contributing to Table 7. |

## 📌 Usage

Make the shell scripts executable and run them as follows:

```bash
chmod +x run.sh run_account.sh run_account1.sh run_account2.sh
./run.sh
./run_account.sh
./run_account1.sh
./run_account2.sh
