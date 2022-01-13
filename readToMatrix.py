import numpy as np
import re
def readToMatrix(filename,name):#
    f=open(filename,"rt")
# pssm
    if name in ['pssm','psfm','pssmAndLabels','psfmAndLabels']:
        pssm=[]
        for j,line in enumerate(f.readlines()):
            if j > 2:
                line=line.strip()
                overall_vec = re.split(r" +",line)
                if len(overall_vec)<44:
                    break
                else:
                    pssm.append(overall_vec[:42])
        pssm=np.array(pssm)
        if name == 'pssm':
            return pssm[:, 2:22].astype(np.float)  # pssm1
        elif name == 'psfm':
            return pssm[:, 22:42].astype(np.float)  # pssm2
        elif name == 'pssmAndLabels':
            return pssm[:, 2:22].astype(np.float), pssm[:, 1]  # pssm1 labels
        elif name == 'psfmAndLabels':
            return pssm[:, 22:42].astype(np.float), pssm[:, 1]  # pssm2 labels
#  hmm
    elif name in['hmm','hmmAndLabels']:
        aminos = ['A ', 'C ', 'D ', 'E ', 'F ', 'G ', 'H ', 'I ', 'K ', 'L ', 'M ', 'N ', 'P ', 'Q ', 'R ', 'S ', 'T ',
              'V ', 'W ', 'Y ']
        hmm = []
        for line in f.readlines():
            line = line.strip()
            if line[:2] in aminos:
                currentList = re.split(r'\s+', line)
                hmm.append(currentList)
        hmm = np.array(hmm)
        if name=='hmm':
            return hmm[:, 2:22]
        else :
            return hmm[:,2:22],hmm[:,0]


def autoNorm(matrix,name):
    if name=="pssm":
        matrix=matrix.astype(np.float)
        matrix = 1 / (1 + np.exp(0 - matrix))
    elif name=="psfm":
        matrix=matrix.astype(np.float)
        matrix = matrix / 100
    elif name=="hmm":
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i][j] == '*':
                    matrix[i][j] = 0
                else:
                    a = 0.001 * eval(matrix[i][j])
                    a = 0 - a
                    matrix[i][j] = 2 ** a
        matrix = matrix.astype(float)
    return matrix






