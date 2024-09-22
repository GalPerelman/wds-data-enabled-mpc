# Data Enabled Predictive Control for Water Distribution Systems Optimization
by Gal Perelman and Avi Ostfeld

## Abstract
Recent developments in control theory coupled with the growing availability of real-time data have paved the way for improved data-driven control methodologies. This study explores the application of Data-Enabled Predictive Control (DeePC) algorithm (Coulson et al., 2018) to optimize the operation of water distribution systems (WDS). WDSs are characterized by inherent uncertainties and complex nonlinear dynamics. Hence, classic control strategies that involve physical model-based or state-space methods are often hard to implement and infeasible to scale. The DeePC method suggests a paradigm shift by utilizing a data-driven approach. The method employs a finite set of input-output samples (control settings, and measured data) to learn an unknown system's behavior and derive optimal policies, effectively bypassing the need for an explicit mathematical model of the system. In this study, DeePC is applied to two WDS control applications of pressure management and chlorine disinfection scheduling, demonstrating superior performance compared to standard control strategies.

## Usage
You can download a copy of all the files in this repository by cloning the
[git](https://git-scm.com/) repository:

    git clone https://github.com/GalPerelman/wds-data-enabled-mpc.git

or [download a zip archive](https://github.com/GalPerelman/wds-data-enabled-mpc/archive/master.zip).

## Installations
Create a python virtual environment `python -m venv <venv_name>`<br>
Activate the new venv by source venv/bin/activate (for mac) or venv/Scripts/activate.bat (for windows)<br>
Install packages in requirements.txt by calling  `pip install -r requirements.txt`