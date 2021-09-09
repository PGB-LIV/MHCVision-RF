# import modules
import os
import sys
import pandas as pd
import MHCVision-RF.Models.mhcvision_pred

argv = sys.argv
path = 'support_data/'
# load supported alleles
supported_allele_netpan = []
supported_allele_flurry = []
with open(path+'supplied_alleles_NetMHCpan.txt', 'rt') as fhla:
    for line in fhla:
        supported_allele_netpan.append(line.strip())
with open(path+'supplied_alleles_MHCflurry.txt', 'rt') as fhla:
    for line in fhla:
        supported_allele_flurry.append(line.strip())
"""
check errors
"""
# check if the user provided valid arguments
def check_valid_argument(arg):
    invalid_flag = False
    if '-a' not in arg and '--allele' not in arg:
        invalid_flag = True
        print('Error: -a/--allele argument is required')
    if '-t' not in arg and '--tool' not in arg:
        invalid_flag = True
        print('Error: -t/--tool argument is required')
    if '-i' not in arg and '--input' not in arg:
        invalid_flag = True
        print('Error: -i/--input argument is required')
    # print detected error
    if invalid_flag == True:
        print('\nPlease see help information below:')
    return invalid_flag

# print help statement
def print_help_():
    print('usage: mhcvision-rf.py [options] input_file.csv -o/--output output_file.csv\n'
          '-a, --allele   REQUIRED: type the allele name i.e. HLA-A0101, which are supported in the "supplied_alleles.txt"\n'
          '-t, --tool     REQUIRED: Specify the MHC-peptide prediction tool you used, type NetMHCpan or MHCflurry\n'
          '-i, --input    REQUIRED: specify the input filename,\n'
          '               the input file must be in ".CSV" format (comma-separated values)'
          ', the column headers must contain "Peptide", "IC50"\n'
          '-o, --output   Optional: specify the output filename\n'
          '-h, --help     Print the usage information')

# extract argument values
def extract_required_arg(arg):
    if '-a' in arg:
        allele_loc = arg.index('-a')
    else:
        allele_loc = arg.index('--allele')
    if '-t' in arg:
        tool_loc = arg.index('-t')
    else:
        tool_loc = arg.index('--tool')
    if '-i' in arg:
        input_loc = arg.index('-i')
    else:
        input_loc = arg.index('--input')
    if '-o' in arg or '--output' in arg:
        out_file = arg[-1]
    else:
        out_file = 'output_' + arg[input_loc+1]
    return arg[allele_loc+1], arg[tool_loc+1], arg[input_loc+1], out_file

# check the input table file format and allele name
def check_input_arg(hla, prediction, file):
    invalid_flag = False
    # 1.check input prediction tools, they must be NetMHCpan or MHCflurry
    if prediction not in ['NetMHCpan', 'MHCflurry']:
        invalid_flag = True
        print('Error: '+ prediction +' has not been supported in this version, it must be NetMHCpan or MHCflurry')
    # 2.check input allele
    supported_allele = []
    if prediction == 'NetMHCpan':
        supported_allele = supported_allele_netpan
    if prediction == 'MHCflurry':
        supported_allele = supported_allele_flurry
    if hla not in supported_allele:
        invalid_flag = True
        print('Error: '+ hla + ' is not in "supplied_alleles_'+prediction+'.txt"')
    # 3.check input file format
    df = pd.read_csv(file, sep=',')
    row1 = list(df.iloc[0,:])
    if len(row1) <= 1:
        invalid_flag = True
        print('Error: The input file must be .CSV format, the columns are separated by ","')
    header = df.columns
    # 4.check the present of IC50 column
    if 'IC50' not in header:
        invalid_flag = True
        print('Error: "IC50" column can not be found')
    # 5.check values in IC50 column
    ic50_col = list(df.loc[:, 'IC50'])
    res = [n for n in ic50_col if (isinstance(n, str))]
    if res:
        invalid_flag = True
        print('Error: "IC50" column must be numbers')
    # print detected error
    if invalid_flag:
        print('\nPlease see help information below:')
    # 6. check the present of 'Peptide' coulumn
    if 'Peptide' not in header:
        invalid_flag = True
        print('Error: "Peptide" column can not be found')

    return invalid_flag


def run_models(allele, prediction_tool, input_file):
    """
    run the RF model
    """
    df_input = pd.read_csv(input_file)
    peptides = df_input.loc[:,'Peptide']
    with open('peptide.txt', 'wt') as fpep: # create an input file for rf_pred.py
        for i in range(len(peptides)):
            fpep.write(peptides[i]+'\n')
    MHCVision-RF.Models.rf_pred.immune_pred()  ## return peptide and immune prob
    """
    run MHCVision
    """
     # loaded hla parameter ranges based on which prediction tool
    df_parameter_range = pd.read_csv(path+'parameter_range_'+prediction_tool+'.csv')
    hla_list = df_parameter_range.iloc[:,0]
    hla_parameter_range = {}
    for y in range(len(hla_list)):
        a2_range = list(df_parameter_range.iloc[y,1:3]) # [min,max]
        b2_range = list(df_parameter_range.iloc[y,3:])
        values = a2_range+b2_range
        hla_parameter_range[hla_list[y]] = values
    data = MHCVision-RF.Models.mhcvision_pred.convert_score_for_beta(input_file)  # converted IC50
    print('The parameter estimation is running...')
    model = MHCVision-RF.Models.mhcvision_pred.BMM(data, allele, hla_parameter_range, input_file,)
    model.initialisation()
    model.termination()
    est_fdr = MHCVision-RF.Models.mhcvision_pred.FDR('beta_parameter.csv', data, allele, input_file)
    print('The FDR/PEP are calculating...')
    est_fdr.write_output()  ## return peptides and statistic values
    # warn users if the estimation goes to all true or all false
    df_est = pd.read_csv('beta_parameter.csv')
    est_w1 = df_est.iloc[0, 1]
    est_w2 = df_est.iloc[1, 1]
    # write warning file if the estimation get a single distribution
    if model.checking_estimated_parameters() == True:
        with open('warning.txt', 'wt') as fwarning:
            if est_w1 > est_w2:
                fwarning.write('Note: The input data is estimated to all binding peptides.\n')
            else:
                fwarning.write('Note: The input data is estimated to all non-binding peptides.\n')
    return


"""
run models and calculate a final probability
"""
# check for -h/--help argument
if '-h' in argv or '--help' in argv or check_valid_argument(argv) == True:
    print_help_()
# if everything has been checked out, run the models
else:
    # extract all arguments
    allele, prediction_tool, input_file, output_file = extract_required_arg(argv)
    # check input argument values
    if check_input_arg(allele, prediction_tool, input_file) == True:
         print_help_()
    # if all argument values are correct, make the estimation
    else:
        run_models(allele, prediction_tool, input_file)

    # calculate final probability
    immune_pred = pd.read_csv('Immune_pred.csv').sort_values(by='Peptide')
    fdr_pred = pd.read_csv('FDR_pred.csv').sort_values(by='Peptide')
    immune_prob = immune_pred.loc[:, 'Immunogenic probability']
    true_binding_prob = fdr_pred.loc[:, 'True probability (1-PEP)']
    final_prob = immune_prob*true_binding_prob
    fdr_pred['Final probability'] = final_prob
    # write final output
    fdr_pred.to_csv(output_file)
    # delete all intermediate files
    os.remove('beta_parameter.csv')
    os.remove('peptide.txt')
    os.remove('Immune_pred.csv')
    os.remove('FDR_pred.csv')
    print('Done! Wrote output to ' + output_file)
