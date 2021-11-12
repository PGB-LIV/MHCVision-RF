import os
import pandas as pd
import numpy as np

"""
convert peptide sequence to numerical values of 42 features
"""
prop_aa_num = {'EISD860102': {'A': 0.0, 'V': 0.48, 'I': 1.2, 'L': 1.0, 'M': 1.9, 'F': 1.1, 'Y': 1.8, 'W': 1.6, 'S': 0.73, 'T': 1.5, 'N': 1.3, 'Q': 1.9, 'C': 0.17, 'G': 0.0, 'P': 0.18, 'R': 10.0, 'H': 0.99, 'K': 5.7, 'D': 1.9, 'E': 3.0},
'OOBM770102': {'A': -1.404, 'V': -1.254, 'I': -1.189, 'L': -1.315, 'M': -1.303, 'F': -1.135, 'Y': -1.03, 'W': -1.03, 'S': -1.297, 'T': -1.252, 'N': -1.178, 'Q': -1.116, 'C': -1.365, 'G': -1.364, 'P': -1.236, 'R': -0.921, 'H': -1.215, 'K': -1.074, 'D': -1.162, 'E': -1.163},
'ZIMJ680104': {'A': 6.0, 'V': 5.96, 'I': 6.02, 'L': 5.98, 'M': 5.74, 'F': 5.48, 'Y': 5.66, 'W': 5.89, 'S': 5.68, 'T': 5.66, 'N': 5.41, 'Q': 5.65, 'C': 5.05, 'G': 5.97, 'P': 6.3, 'R': 10.76, 'H': 7.59, 'K': 9.74, 'D': 2.77, 'E': 3.22},
'OOBM770103': {'A': -0.491, 'V': -0.728, 'I': -0.762, 'L': -0.65, 'M': -0.659, 'F': -0.729, 'Y': -0.656, 'W': -0.839, 'S': -0.455, 'T': -0.515, 'N': -0.382, 'Q': -0.405, 'C': -0.67, 'G': -0.534, 'P': -0.463, 'R': -0.554, 'H': -0.54, 'K': -0.3, 'D': -0.356, 'E': -0.371},
'KRIW790102': {'A': 0.28, 'V': 0.22, 'I': 0.12, 'L': 0.16, 'M': 0.08, 'F': 0.1, 'Y': 0.25, 'W': 0.15, 'S': 0.27, 'T': 0.26, 'N': 0.31, 'Q': 0.39, 'C': 0.11, 'G': 0.28, 'P': 0.46, 'R': 0.34, 'H': 0.23, 'K': 0.59, 'D': 0.33, 'E': 0.37},
'FASG760101': {'A': 89.09, 'V': 117.15, 'I': 131.17, 'L': 131.17, 'M': 149.21, 'F': 165.19, 'Y': 181.19, 'W': 204.24, 'S': 105.09, 'T': 119.12, 'N': 132.12, 'Q': 146.15, 'C': 121.15, 'G': 75.07, 'P': 115.13, 'R': 174.2, 'H': 155.16, 'K': 146.19, 'D': 133.1, 'E': 147.13},
'KRIW710101': {'A': 4.6, 'V': 3.4, 'I': 2.6, 'L': 3.25, 'M': 1.4, 'F': 3.2, 'Y': 4.35, 'W': 4.0, 'S': 5.25, 'T': 4.8, 'N': 5.9, 'Q': 6.1, 'C': -1.0, 'G': 7.6, 'P': 7.0, 'R': 6.5, 'H': 4.5, 'K': 7.9, 'D': 5.7, 'E': 5.6},
'FAUJ880103': {'A': 1.0, 'V': 3.0, 'I': 4.0, 'L': 4.0, 'M': 4.43, 'F': 5.89, 'Y': 6.47, 'W': 8.08, 'S': 1.6, 'T': 2.6, 'N': 2.95, 'Q': 3.95, 'C': 2.43, 'G': 0.0, 'P': 2.72, 'R': 6.13, 'H': 4.66, 'K': 4.77, 'D': 2.78, 'E': 3.78},
'PRAM900101': {'A': -6.7, 'V': -10.9, 'I': -13.0, 'L': -11.7, 'M': -14.2, 'F': -15.5, 'Y': 2.9, 'W': -7.9, 'S': -2.5, 'T': -5.0, 'N': 20.1, 'Q': 17.2, 'C': -8.4, 'G': -4.2, 'P': 0.8, 'R': 51.5, 'H': 12.6, 'K': 36.8, 'D': 38.5, 'E': 34.3},
'BLAS910101': {'A': 0.616, 'V': 0.825, 'I': 0.943, 'L': 0.943, 'M': 0.738, 'F': 1.0, 'Y': 0.88, 'W': 0.878, 'S': 0.359, 'T': 0.45, 'N': 0.2366, 'Q': 0.251, 'C': 0.68, 'G': 0.501, 'P': 0.711, 'R': 0.0, 'H': 0.165, 'K': 0.283, 'D': 0.028, 'E': 0.043},
'GRAR740102': {'A': 8.1, 'V': 5.9, 'I': 5.2, 'L': 4.9, 'M': 5.7, 'F': 5.2, 'Y': 6.2, 'W': 5.4, 'S': 9.2, 'T': 8.6, 'N': 11.6, 'Q': 10.5, 'C': 5.5, 'G': 9.0, 'P': 8.0, 'R': 10.5, 'H': 10.4, 'K': 11.3, 'D': 13.0, 'E': 12.3},
'HUTJ700103': {'A': 154.33, 'V': 207.6, 'I': 233.21, 'L': 232.3, 'M': 202.65, 'F': 204.74, 'Y': 229.15, 'W': 237.01, 'S': 174.06, 'T': 205.8, 'N': 207.9, 'Q': 235.51, 'C': 219.79, 'G': 127.9, 'P': 179.93, 'R': 341.01, 'H': 242.54, 'K': 300.46, 'D': 194.91, 'E': 223.16},
'DAWD720101': {'A': 2.5, 'V': 5.0, 'I': 5.5, 'L': 5.5, 'M': 6.0, 'F': 6.5, 'Y': 7.0, 'W': 7.0, 'S': 3.0, 'T': 5.0, 'N': 5.0, 'Q': 6.0, 'C': 3.0, 'G': 0.5, 'P': 5.5, 'R': 7.5, 'H': 6.0, 'K': 7.0, 'D': 2.5, 'E': 5.0},
'EISD840101': {'A': 0.25, 'V': 0.54, 'I': 0.73, 'L': 0.53, 'M': 0.26, 'F': 0.61, 'Y': 0.02, 'W': 0.37, 'S': -0.26, 'T': -0.18, 'N': -0.64, 'Q': -0.69, 'C': 0.04, 'G': 0.16, 'P': -0.07, 'R': -1.76, 'H': -0.4, 'K': -1.1, 'D': -0.72, 'E': -0.62},
'EISD860103': {'A': 0.0, 'V': 0.84, 'I': 0.99, 'L': 0.89, 'M': 0.94, 'F': 0.92, 'Y': -0.93, 'W': 0.67, 'S': -0.67, 'T': 0.09, 'N': -0.86, 'Q': -1.0, 'C': 0.76, 'G': 0.0, 'P': 0.22, 'R': -0.96, 'H': -0.75, 'K': -0.99, 'D': -0.98, 'E': -0.89},
'GOLD730101': {'A': 0.75, 'V': 1.7, 'I': 2.95, 'L': 2.4, 'M': 1.3, 'F': 2.65, 'Y': 2.85, 'W': 3.0, 'S': 0.0, 'T': 0.45, 'N': 0.69, 'Q': 0.59, 'C': 1.0, 'G': 0.0, 'P': 2.6, 'R': 0.75, 'H': 0.0, 'K': 1.5, 'D': 0.0, 'E': 0.0},
'ZIMJ680103': {'A': 0.0, 'V': 0.13, 'I': 0.13, 'L': 0.13, 'M': 1.43, 'F': 0.35, 'Y': 1.61, 'W': 2.1, 'S': 1.67, 'T': 1.66, 'N': 3.38, 'Q': 3.53, 'C': 1.48, 'G': 0.0, 'P': 1.58, 'R': 52.0, 'H': 51.6, 'K': 49.5, 'D': 49.7, 'E': 49.9}}

aa_features = ['EISD860102', 'OOBM770102', 'ZIMJ680104', 'OOBM770103', 'KRIW790102', 'FASG760101', 'KRIW710101',
               'FAUJ880103', 'PRAM900101', 'BLAS910101', 'GRAR740102', 'HUTJ700103', 'DAWD720101', 'EISD840101',
               'EISD860103', 'GOLD730101', 'ZIMJ680103']

optimal_features = ['p6_EISD860102', 'p2_OOBM770102', 'p6_OOBM770102', 'p7_OOBM770102', 'p2_ZIMJ680104', 'p9_ZIMJ680104',
                    'p3_OOBM770103', 'p8_OOBM770103', 'p2_KRIW790102', 'p9_FASG760101', 'p4_KRIW710101', 'p5_KRIW710101',
                    'p3_FAUJ880103', 'p6_FAUJ880103', 'p9_FAUJ880103', 'p1_PRAM900101', 'p5_PRAM900101', 'p7_PRAM900101',
                    'p9_PRAM900101', 'p3_BLAS910101', 'p7_BLAS910101', 'p2_GRAR740102', 'p1_HUTJ700103', 'p8_HUTJ700103',
                    'sum_EISD860102', 'sum_OOBM770102', 'sum_ZIMJ680104', 'sum_OOBM770103', 'sum_KRIW790102', 'sum_FASG760101',
                    'sum_KRIW710101', 'sum_FAUJ880103', 'sum_DAWD720101', 'sum_EISD840101', 'sum_EISD860103', 'sum_GOLD730101',
                    'sum_PRAM900101', 'sum_BLAS910101', 'sum_GRAR740102', 'sum_ZIMJ680103', 'sum_HUTJ700103']


def convert_aa_to_num(pep_seq):
    pep_seq = pep_seq.upper()
    pep_9mer = pep_seq[:9]
    df_convert = pd.DataFrame()
    loc_p = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9']
    # convert aa in the input peptide to number
    for i in range(len(pep_9mer)):
        prop_num = []
        for p in prop_aa_num:
            aa_num = prop_aa_num[p][pep_9mer[i]]
            prop_num.append(aa_num)
        df_convert[loc_p[i]] = prop_num
    df_convert.index = aa_features  # columns = aa and sum, rows = features
    # compute sum of numbers for all aa in the peptide of each feature
    sum_pep_feat = []
    for j in range(len(df_convert)):
        feat_j = np.sum(df_convert.iloc[j, :])
        sum_pep_feat.append(feat_j)
    df_convert['sum'] = sum_pep_feat
    selected_feat = []
    for k in range(len(optimal_features)):
        feat = optimal_features[k]  # [0]=column, [1]=row
        feat = str(feat).split('_')
        selected_feat.append(feat)
    value_input_pep = {'peptides': pep_seq}  # return a value of a selected feature, {feature:value,...}
    for sf in selected_feat:
        c = sf[0]
        r = sf[1]
        value = df_convert.loc[r,c]
        value_input_pep[c+'_'+r] = value
    # convert dict to data frame, the first col is peptide, the rest are optimum features
    df_pep_to_num = pd.DataFrame([value_input_pep])
    return df_pep_to_num


def blast_search(pep,num_aln):
    db = './BLAST_db/homo_sapien'
    f = open('pep.txt','wt')
    f.write('>'+pep+'\n')
    f.write(pep+'\n')
    f.close()
    command = 'blastp -task blastp-short -db ' + db + ' -query pep.txt -out blastout.csv ' \
                '-outfmt "10 qseq sseq sacc pident evalue bitscore score nident mismatch" ' \
                '-evalue 10000000 -word_size 2 -num_alignments '+str(num_aln)
    os.system(command)
    df_blast = pd.read_csv('blastout.csv', header=None,
                           names=['qseq','sseq','sacc','pident','evalue','bitscore','score','nident','mismatch'])
    return df_blast


def blast_score(pep, num_aln):
    pep_len = len(pep)
    df_blast = blast_search(pep,num_aln)
    while len(df_blast) == 0:
        blast_search(pep, num_aln)
    else:
        df_blast = df_blast.assign(len_qseq=np.zeros(len(df_blast)))
        df_blast = df_blast.assign(len_sseq=np.zeros(len(df_blast)))
        df_blast['gap_qseq'] = ['no'] * len(df_blast)
        df_blast['gap_sseq'] = ['no'] * len(df_blast)
        # restrict with no gap and nine mers of qseq and sseq
        for i in range(len(df_blast)):
            df_blast.loc[i, 'len_qseq'] = int(len(str(df_blast.loc[i, 'qseq'])))
            df_blast.loc[i, 'len_sseq'] = int(len(str(df_blast.loc[i, 'sseq'])))
            if '-' in str(df_blast.loc[i, 'qseq']):
                df_blast.loc[i, 'gap_qseq'] = 'yes'
            else:
                df_blast.loc[i, 'gap_qseq'] = 'no'
            if '-' in str(df_blast.loc[i, 'sseq']):
                df_blast.loc[i, 'gap_sseq'] = 'yes'
            else:
                df_blast.loc[i, 'gap_sseq'] = 'no'
        no_gap = df_blast[(df_blast['gap_qseq'] == 'no') & (df_blast['gap_sseq'] == 'no')]
        len_no_gap = no_gap[(no_gap['len_qseq'] == pep_len) & (no_gap['len_sseq'] == pep_len)]
        return len_no_gap


def concat_numerical_data(seq,num_aln):
    len_no_gap = blast_score(seq, num_aln)
    best_score = np.max(len_no_gap['score'])
    while len(len_no_gap) == 0:
        # increase align number
        num_aln += 500
        len_no_gap = blast_score(seq, num_aln)
        if len(len_no_gap) != 0:
            best_score = np.max(len_no_gap['score'])
            break
    # label aaindex for nine residues
    df_out = convert_aa_to_num(seq)
    df_out['blast_score'] = best_score
    return df_out


def generate_matrix(input_file):
    df_input = pd.read_csv(input_file,sep='\n',header=None)
    peptides = df_input.iloc[:,0]
    num_aln = 500
    # do the first peptide
    df_matrix = concat_numerical_data(peptides[0], num_aln)
    os.remove('pep.txt')
    os.remove('blastout.csv')
    # do the rest (the second and so on)
    for j in range(1, len(peptides)):
        df_next = concat_numerical_data(peptides[j], num_aln)
        df_matrix = pd.concat([df_matrix, df_next])
        os.remove('pep.txt')
        os.remove('blastout.csv')
    return df_matrix
