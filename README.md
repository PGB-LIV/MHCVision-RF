# MHCVision-RF
A predicition tool for a probability of MHC I binding and T cell recognition
### **Introduction**
MHCVision-RF is an assemble pipeline of [MHCVision](https://github.com/PGB-LIV/MHCVision) and the immunogenicity prediction model using the Random Forest. A probability score produced from MHCVision-RF is computed from a true MHC binding probabillity and an immunogegenic probability. The current version is feasible for MHC-peptide binding prediction using NetMHCpan (version >= 4.0) or MHCflurry. 
### **Client software requirement**
1. The model requires Python 3 ( >= 3.7) and the following python packages:
```
pandas (>= 1.1.2)
numpy (>= 1.19.1)
scipy (>=1.5.2)
scikit-learn (>=0.23.2)
```
 For python installing packages, please see [here](https://packaging.python.org/tutorials/installing-packages/#use-pip-for-installing)

 If your system has both Python 2 and Python 3, please ensure that Python 3 is being used when following these instructions.

2. Standalone BLAST ([version 2.7.1](https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/2.7.1/)) 

 For BLAST installation, please see [here](https://www.ncbi.nlm.nih.gov/books/NBK279690/)
 ### **How to install?**
 1. Clone this repository
```
git clone https://github.com/PGB-LIV/MHCVision-RF
```
For other methods for cloning a GitHub repository, please see  [here](https://help.github.com/articles/cloning-a-repository/)

2. Install the latest version of 'pip' and 'setuptools' packages for Python 3 if your system does not already have them
```
python -m ensurepip --default-pip
pip install setuptools
```
For more information, please see [here](https://packaging.python.org/tutorials/installing-packages/#install-pip-setuptools-and-wheel)

### **Usage**
```
usage: mhcvision-rf.py [options] input_file.csv -o/--output output_file.csv
options:
-a, --allele   REQUIRED: type the allele name i.e. HLA-A0101, which are supported in the "supplied_alleles.txt"
-t, --tool     REQUIRED: Specify the MHC-peptide prediction tool you used, type NetMHCpan or MHCflurry
-i, --input    REQUIRED: specify the input filename, the input file must be in ".CSV" format (comma-separated values), the column headers must contain 'Peptide', 'IC50'
-o, --output   Optional: specify the output filename 
-h, --help     Print the usage information
```

### **Sample scripts**
You can use input_sample.csv as the input file
```
python mhcvision-rf.py -a HLA-A0201 -t NetMHCpan -i input_sample.csv
```
