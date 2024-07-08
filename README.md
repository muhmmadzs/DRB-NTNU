# DRB-NTNU
Deep Autoencoder Model for Bridge Damage Detection


This repository contains a code for damage assessment using a deep autoencoder model. This method comes from a soon-to-be-published paper. The paper's title is [Numerical benchmark for road bridge damage detection from passing vehicles responses applied to four data-driven methods](https://link.springer.com/article/10.1007/s43452-024-01001-9)


# Core Focus

This repository contains implementation for Section 4 of the aforementioned paper. In this section, the spotlight is on the deep autoencoder model, its architecture, and its application in damage detection. This model is adept at learning the inherent patterns of undamaged bridge structures and subsequently detecting deviations or anomalies that signify possible damage.

# Methodology

Once the deep autoencoder is trained on baseline (undamaged) data, it attempts to reproduce any new data fed to it. Any significant discrepancies between the reproduced data and the original are potential indicators of damage. To quantify these discrepancies or deviations, the Kullback-Leibler (KL) divergence is employed. The KL divergence serves as a statistical measure to gauge the difference between two probability distributions. In this context, it quantifies the divergence between the model's predictions and the actual data, giving us a 'damage index'. A higher damage index suggests a higher likelihood of damage.

# Usage

For Training Run the "AE_B15_P00.py" File. 
For the Damage index Locate the python file "Plotting_log_128.py" in the folder directory "/output/FEM Bridge/B15/PA00/"

