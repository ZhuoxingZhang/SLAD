# Introduction
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;This repository contains various artifacts, such as source code, experimental results, and other materials, that supplement our work on **Substructure-aware Log Anomaly Detection**. This work proposes a novel framework for code file anomaly detection from system logs, SLAD. It first introduces a Monte Carlo tree search strategy tailored specifically for log anomaly detection to discover representative substructures. Then, SLAD incorporates a substructure distillation way to enhance the efficiency of anomaly inference based on the representative substructures. By utilizing the distilled substructure representations, each node in the inference phase can get the approximate substructure information through attention mechanism, and then find the key substructure information of nodes via soft pruning. Finally, SLAD uses an MLP with softmax to predict labels. The framework of SLAD is listed below.
![SLAD framework diagram](https://github.com/ZhuoxingZhang/SLAD/blob/main/Artifact/slad-framework.png)
# Preliminaries: Getting ready for experiments
> 1. Download the datasets: Forum & Halo & Novel
>> The datasets for our experiments are placed [here](https://drive.google.com/drive/folders/11blVvVF?usp=drive_link). The datasets are in a .rar format. Before conducting experiments, please unzip the datasets and place them in the same directory with the source code.
> 2. Download other necessary files for experiments
>> Please visit the directory <kbd>Artifact/necessary/</kbd> and download all of them. Please make sure to place them in the same directory with the source code.
> 3. Software requirements
>>python 3.10+; pytorch 2.1.2+; torch-geometric 2.4.0+
# How to run code from the command line
1. Clone the repository:
   ```bash
   git clone https://github.com/ZhuoxingZhang/SLAD.git
   ```
2. Navigate to the project directory:
   ```bash
   cd <your_project_directory>
   ```
3. Run code from the command line for experiments:
   ```bash
   python3 --bs <batch size> --n_epoch <epoch number> --representation_num <number of representative substructures> --dataset <Forum/Halo> slad_framework.py
   ```

   

