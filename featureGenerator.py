import numpy as np
from cmath import *
#aac
def get20Aminos(matrix):
    vector=matrix.sum(axis=0)/matrix.shape[0]
    return vector
#pssm-composition
def getRPT(matrix,labels):
    aminos = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
    ls=[]
    reMatrix=np.zeros((20,matrix.shape[1]))
    for j,c in enumerate(aminos):
        reMatrix[j,:]=matrix[labels==c,:].sum(axis=0)
    return reMatrix.flatten()
#dp-pssm
def getDP_PSSM(matrix,a):
    nrow=matrix.shape[0]
    ncol=matrix.shape[1]
    #
    matrix = (matrix - np.mean(matrix)) / np.std(matrix)
    #
    ls=[]
    #
    for i in range(ncol):#
        posnum=0
        negnum=0
        posvalue=0
        negvalue=0
        for j in range(nrow):#
            if matrix[j,i]>=0:
                posnum+=1
                posvalue+=matrix[j,i]
            else:
                negnum+=1
                negvalue+=matrix[j,i]
        if posnum==0:
            ls.append(0)
            ls.append(negvalue/(nrow-posnum))
        elif negnum==0:
            ls.append(posvalue / posnum)
            ls.append(0)
        else:
            ls.append(posvalue / posnum)
            ls.append(negvalue / negnum)
    #
    for z in range(1,a+1):
        for i in range(ncol):#
            NDP = 0
            NDN=0
            NDPvalue = 0
            NDNvalue = 0
            for j in range(nrow-z):#
                n=matrix[j][i]-matrix[j+z][i]
                if n>=0:
                    NDP+=1
                    NDPvalue+=n**2
                else:
                    NDN+=1
                    NDNvalue-=n**2
            ls.append(NDPvalue/NDP)
            ls.append(NDNvalue/NDN)
    ls=np.array(ls)
    return ls
















