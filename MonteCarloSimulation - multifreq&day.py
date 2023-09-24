import numpy as np
import math
from itertools import chain
import scipy.stats as ss


s0=1.51
r=0.08/100
q=0.025
T=138/365
sigma=0.38
onedayfreq=1
stepNo=int(139/365*252)*onedayfreq
simNo=10000
paidIRR=0.33
spread=0.47
deltaT=T/(stepNo-1)
df=np.exp(-(r+spread)*deltaT)
X0 = np.zeros((simNo,1))
increments = ss.norm.rvs(loc=(r-q - sigma**2/2)*deltaT, scale=np.sqrt(deltaT)*sigma, size=(simNo,stepNo-1))
X = np.concatenate((X0,increments), axis=1).cumsum(1)
S = s0 * np.exp(X)
# print(S)
# S=[[1.51,1.22343271,1.08075051, 0.9609744, 0.7521414 ],\
#  [1.51,1.66147791,1.83109161, 1.99872643, 2.30452198],\
#  [1.51,1.6575743,  1.7866781,  1.91275282, 1.98343603],\
#  [1.51,1.48351019, 1.44310339, 1.35410765, 1.77200051],\
#  [1.51,1.74296688, 1.36040951, 1.18865053, 1.03568281],\
#  [1.51,1.5190136,  1.44904886 ,1.56777364, 1.32501458]]
stockmatrix=np.matrix(S)#把stockList转换成矩阵
# print(stockmatrix)

def strikePi(stepi,paidIRR):#strikeP(t)的函数
    strikePi=2.3*(1+0.08)**(deltaT*stepi+5.5-T)-paidIRR
    return strikePi


def KnockInPoint(stockListi):#用于每条path上的knockinpoint在第几个step的判断,stockListi代表第i条路径
    knockinpoint=stepNo*2
    stockclosing=[]
    for i in range(0,stepNo,onedayfreq):
        stockclosing.append(stockListi[i])
    for i in range(9,int(stepNo/onedayfreq)):
        a=stockclosing[i-9:i+1]
        if max(a)<1.61:
            knockinpoint=i*onedayfreq
            break
    return knockinpoint
knockinpointList=[]
for stockListi in S:#
    knockinpointList.append(KnockInPoint(stockListi))
# print(knockinpointList)


def EV(stockListi):#针对某条path，计算每个step上的payoff，并返回一条payoffList
    EVpayoffList=[]
    knockinpointi=KnockInPoint(stockListi)
    for stepi in range(0,stepNo):
        strikepi=strikePi(stepi,paidIRR)
        if stepi==stepNo-1:
            EVpayoff=np.maximum(strikepi-stockListi[stepi],0)
        elif stepi>knockinpointi:
            EVpayoff=np.maximum(strikepi-stockListi[stepi],0)#如果对应step已经knock in，则可以考虑put的行权
        else:
            EVpayoff=0#如果还没有Knock in，则就是股价，就不考虑行权与否
        EVpayoffList.append(EVpayoff)
    return EVpayoffList
EVpayoffList=[]
for stockListi in S:
    EVpayoffList.append(EV(stockListi))
EVmatrix=np.matrix(EVpayoffList)#转换成EVpayoff矩阵
# print(EVmatrix)

value=np.zeros_like(stockmatrix)
value[:,stepNo-1]=EVmatrix[:,-1]
# print(value)
# print(strikePi(stepi,paidIRR))
for stepi in range(stepNo-2,-1,-1):
    EVstepi = EVmatrix[:,stepi]#取出第stepi步的所有EV exercise value
    # print(EVstepi)
    stockstepi=stockmatrix[:,stepi]#取出第stepi步的所有S
    HVstepi=value[:,stepi+1]*df#取出第stepi步的所有holding value
    # print(HVstepi)
    good_paths=EVmatrix[:,stepi]>0#挑出EV在第stepi上不为0的path
    # print(good_paths)
    x=list(chain.from_iterable(stockstepi[good_paths].tolist()))
    y=list(chain.from_iterable(HVstepi[good_paths].tolist()))
    if x!=[]:
        rg=np.polyfit(x,y,2)
        EHV=np.polyval(rg,list(chain.from_iterable(stockstepi.tolist())))#把真实的S代入进去,包括那些EVpayoff=0的path
        for simNoi in range(0,simNo):
            if good_paths[simNoi]==False:
                EHV[simNoi]=0#对于EVpayoff=0的点，EHV设为0
    else:
        EHV=np.zeros_like(EVstepi)#x为空集代表这个stepi上的所有path的EV均为0
    # print(EHV)
    for simNoi in range(0,simNo):
        if good_paths[simNoi]==False:
            value[simNoi,stepi]=HVstepi[simNoi]
        elif EHV[simNoi]>=EVstepi[simNoi]:
            value[simNoi,stepi]=HVstepi[simNoi]
        else:
            value[simNoi,stepi]=EVstepi[simNoi]
    # print(value)
# print(value[:,0])
print(np.std(value[:,0]))
print(np.mean(value[:,0]))   