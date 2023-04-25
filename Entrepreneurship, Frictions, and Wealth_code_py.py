'原代码为Fortran,为增强可读性，将其转换为py.fortran代码有部分错误，且和论文中介绍模型有一定区别。已在注释中标注'
import pandas as pd
import numpy as np
import matplotlib as plt
'parameter values'

da1, da2 = 40, 220 # # asset levels for each grid 原始值40,220
da = da1 + da2 # # asset levels
dr = 2 # # 企业能力类型
dy = 5 # # income realizations
dk = da # # k level

'preferences and technology'
bet = 0.867
gam = 1.5
delt = 0.06 # capital deprec.

eta = 1.0 # altruism toward children
ni = 0.88 # decr returns
alph = 0.33 # capital share in non entr sector
abig = 1 # constant in nonentr prod fn 非生产部门C-D函数中
'degree of decreasing returns to entrep. inv.'


'grid for a'
mina = 0.0
maxa = 1700  #maxa=1700原始值

'entrepreneurs'
mink = 0.05 # minimum investment level
maxk = maxa # maximum investment level

'aging'
pyou = 0.9778 # prob. staying young
pold = 0.911 # prob. staying old (not dying)

'government parameters'
replrate = 0.4 # repl. rate for pensions
tauc = 0.0 # 0.11 # consumption tax
taua = 0.0 # capital income tax

'cricri'
'INTEGER, PARAMETER :: indtaua=0 ! 0 if taua=0, 1 if taua>0'
indtaua = 1 # 0 if taua=0, 1 if taua>0
tauls = 0.0 # lump sum tax
exem = 500 # 20.0 # estate taxes exemption level
taub = 0.0 # tax rate on estates
gfrac = 0.0 # frac gov exp / gdp
debtfrac = 0.10 # frac gov debt /total capital

'enforcement'
'投资k后可保留的份额'
eff = 0.75 # prop k kept when defaulting

'transition matrix nonzero elements'
nyoung = dr * dy * da # # possible states for young w or entr
"""
这里dr表示经商才能，dy表示收入冲击，da表示个体期初的资产
"""
noe = da * dr # # possible states for old entr
"""
老年个体无法供给劳动，没有dy
"""
nstates = 2 * nyoung + noe + da # # states (total)
nonzero = 2 * (dy * dr + dr + 1) * nyoung + (dy * dr + dr + 1) * noe + (dy * dr + 1) * da
sizeM = 2 * nyoung + noe + da # dimension of trans mat M

'number of iterations on iterakhat for which we want to save value functions'
nite = 6

'*****************  VARIABLES'
'indexes'
i = 0
i2 = 0
j = 0
j1 = 0
j2 = 0
jj = 0
l = 0
ll = 0
OpenStatus = 0
imax = [0] # one-dimensional integer array with length of 1
indanet = [0] # one-dimensional integer array with length of 1
imaxmat = [[0,0],[0,0]] # two-dimensional integer array with size of 2x2

# convergence criteria

#epsihat = abs(epsihaty + epsihato) # abs((kyhat-newkyhat)+(kohat-newkohat))=0
                                    # iterate on val funs and bc until it's zero.

# equilibrium risk-free interest rate and wages
"""
rbar = 0.0
rbarmin = 0.0
rbarmax = 0.0
rimplied = 0.0
wage = 0.0
wageimplied = 0.0

# pensions
transf = 0.0
"""
# grids
a = np.zeros(da)     # grid for assets
anet = np.zeros(da)  # grid for net assets
k = np.zeros(dk)     # grid for k
r = np.zeros(dr)     # grid for entrepr. ability
y = np.zeros(dy)     # grid for worker ability
Pyr = np.zeros((dr*dy,dr*dy))  # joint distr. of y and r
Pyrtr = np.zeros((dr*dy,dr*dy)) # transition matrix for Pyr
Pr = np.zeros((dr,dr))  # p(r'|r)
Prtr = np.zeros((dr,dr)) # transition matrix for Pr
Py = np.zeros((dy,dy))   # p(y'|y)
Pytr = np.zeros((dy,dy)) # transition matrix for Py
invyr = np.zeros(dy*dr)  # invariant distr of y and r
invy = np.zeros(dy)      # invariant distr of y
invr = np.zeros(dr)      # inv dist for r


# taxes estimated using Gouveia Strauss tau=b-b*(s*y**p+1)**(-1/p)
# stax depends on income normalization
#btaxw, staxw, ptaxw, staxwbase, avgywsim = 0.0, 0.0, 0.0, 0.0, 0.0
#btaxe, staxe, ptaxe, staxebase, avgyesim = 0.0, 0.0, 0.0, 0.0, 0.0

# value functions
# young
Vy = np.zeros((da, dy, dr))   # young dy da分别表示劳动和经商的两种冲击
Vye = np.zeros((da, dy, dr))  # young that is entrep for this period
Vyw = np.zeros((da, dy, dr))  # young that is worker for this period
# old
Voee = np.zeros((da, dr))     # old entrepreneur staying entrep
Vow = np.zeros(da)            # old, retired, worker
Voe = np.zeros((da, dr))      # old entrepreneur
# defaulted worker
Vwkeff = np.zeros((dk, dy, dr))
Vokeff = np.zeros((dk))

# descendants
Vynet = np.zeros((da))
EVnewbw = np.zeros((da))
EVnewbe = np.zeros((da, dr))
EVy = np.zeros((da, dy, dr))

# policy functions
# young
apolye = np.zeros((da, dy, dr), dtype=int)
kpolye = np.zeros((da, dy, dr), dtype=int)
apolyw = np.zeros((da, dy, dr), dtype=int)
apoly = np.zeros((da, dy, dr), dtype=int)
kpoly = np.zeros((da, dy, dr), dtype=int)

# old
apolow = np.zeros((da))
apolownet = np.zeros((da))
apoloe = np.zeros((da, dr))
kpoloe = np.zeros((da, dr))
apoloenet = np.zeros((da,dr))

# exogenous investment limits
kyhat = np.zeros((da, dy, dr), dtype=int)
kohat = np.zeros((da, dr), dtype=int)
newkyhat = np.zeros((da, dy, dr))
newkohat = np.zeros((da, dr))

# transition matrix
colM = np.zeros((nonzero))
rowM = np.zeros((nonzero))
valM = np.zeros((nonzero))

# invariant distribution
invm = np.zeros((nstates))
invm1 = np.zeros((nstates))
prgrid = np.zeros((da))
prgridyw = np.zeros((da))
prgridye = np.zeros((da))
prgridoe = np.zeros((da))
prgridow = np.zeros((da))
invlevk = np.zeros((nstates-da))
invrk = np.zeros((nstates-da))
invpolk = np.zeros((nstates-da))

# other variables
totL = toteffL = 0.0
"""
totayw = totaye = totaow = totaoe = 0.0
totk = inck = 0.0
k2gdp = totkcorp = gdp = 0.0
ykshare = totke = 0.0
yktotsh = 0.0
totentr = totret = 0

beq2gdp = 0.0
"""
ifswitchew = np.zeros((nstates), dtype=int)
ifswitchwe = np.zeros((nstates), dtype=int)
#propewswitch = propweswitch = 0.0

#government revenues
vectaxcw = np.zeros(nstates)
vectaxce = np.zeros(nstates)
vectaxl = np.zeros(nstates)
vectaxe = np.zeros(nstates)
vectaxa = np.zeros(nstates)
vectaxbw = np.zeros(nstates)
vectaxbe = np.zeros(nstates)
vectotincw = np.zeros(nstates)
vecwe2yw = np.zeros(nstates)
vecwe2ye = np.zeros(nstates)

tottaxcw = 0.0
tottaxce = 0.0
tottaxe = 0.0
tottaxa = 0.0
tottaxbw = 0.0
tottaxbe = 0.0
tottaxl = 0.0
totincw = 0.0
govdebt = 0.0
govbal = 0.0
govbal1 = 0.0
govbalold = 0.0
govbalinf = 0.0
govbalsup = 0.0
taubal = 0.0
taubal1 = 0.0
taubalold = 0.0
taubalinf = 0.0
taubalsup = 0.0

#temporary objects

ytaxw = 0.0
taxw = 0.0
ytaxo = 0.0


ytaxe = np.zeros(dk)
taxe = np.zeros(dk)
ucons = np.zeros(da)
cs = np.zeros(da)
uconsold = np.zeros((da, da))
uconsolde = np.zeros((da, da, dr, dk))
uconsw = np.zeros((da, da, dy))
uconse = np.zeros((da, da, dr-1, dk))
uconsl = np.zeros(dk)
csl = np.zeros(dk)
newVy = np.zeros((da, dy, dr))
newVyw = np.zeros((da, dy, dr))
newVye = np.zeros((da, dy, dr))
newVoe = np.zeros((da,dr))
newVow = np.zeros(da)
Vowtemp = np.zeros(da)
Vywtemp = np.zeros(da)
Vyetemp = np.zeros((dk, da))
Veetemp = np.zeros((dk, da))
Voeetemp = np.zeros((dk, da))
sumrowM = np.zeros(sizeM)
ahere = 0.0
khere = 0.0
rhere = 0.0
entinchere = 0.0
winchere = 0.0
#19.11
# define arrays for eigenvalues and eigenvectors
eigvaly = np.zeros(dy)
eigvalcy = np.zeros(dy, dtype=complex)
eigvalyr = np.zeros(dy*dr)
eigvalcyr = np.zeros(dy*dr, dtype=complex)
eigvecy = np.zeros((dy, dy))
eigveccy = np.zeros((dy, dy), dtype=complex)
eigvecyr = np.zeros((dy*dr, dy*dr))
eigveccyr = np.zeros((dy*dr, dy*dr), dtype=complex)
eigvalr = np.zeros(dr)
eigvalcr = np.zeros(dr, dtype=complex)
eigvecr = np.zeros((dr, dr))
eigveccr = np.zeros((dr, dr), dtype=complex)

# define integer and character variables
counter = 0
crow = 0
count1 = 0
count2 = 0
iterar = 0
iterarmax = 0
itera = 0
iteragov = 0
iteragovmax = 0
iterakhat = 0
fname = ''
penalty = 0.0
relax = 0.0
#19.13
#relaxation  parameter for gov. distance first  two starting pts for taul in algo
relaxgov = 0.1
pertgov = 0.001
fundiff = np.zeros(50)
funtota = np.zeros(50)
funtotk = np.zeros(50)
funrbar = np.zeros(50)
we2ye = 0.0  #wealth to income ratio for entrep and workers
we2yw = 0.0
we2ywmedian = 0.0
we2yemedian = 0.0
#gridabreak = 0.0  #cutoff for two types of grid
bracket = 0
noneedrbarmax = 0 #noneedrbarmax is a switch indicator
fundiffnow = 0.0
fundiffmin = 0.0
fundiffmax = 0.0

# define val funs that we save when iterating on iterakhat to speed up computation time
# value functions
Vyer = np.zeros((da, dy, dr, nite))
Vywr = np.zeros((da, dy, dr, nite))
Vyr = np.zeros((da, dy, dr, nite))
Voeer = np.zeros((da, dr, nite))
Vowr = np.zeros((da, nite))
Voer = np.zeros((da, dr, nite))

counterinvm = 0
incomei = 0.0
w2iwi = 0.0
w2iei = 0.0
kborrowed = 0.0

# convenient constants used to compute w2's
ykshare1 = 0.0
totkborr = 0.0
totyshe = 0.0
invkborr = np.zeros(nstates - da)
invyshe = np.zeros(nstates - da)

# the following used in the median w/y computation
nentr = 0
nywork = 0
whoise = np.zeros(nstates, dtype=int)
vecwe2yeonly = None
invmeonly = None
vecwe2ywonly = None
invmywonly = None


# INTERFACE SUBROUTINES

def linspace(xmin, xmax, npoints):
    # Returns npoints evenly spaced numbers over the range (xmin, xmax).
    lspace = np.linspace(xmin, xmax, npoints)
    return lspace

"""
def interplin(x, y, z):
    # Given (x, y) pairs, and points z where you want to interpolate,
    # returns the interpolated points v.
    n = len(z)
    v = np.interp(z, x, y)
    return v
"""

def interplin(l, x, y, n, z, v):
    for i in range(n):
        k = np.argmax(-np.abs(z[i]-x))
        ind = k
        diff = z[i]-x[ind]
        if (np.abs(diff) < 1e-04):
            v[i] = y[ind]
        else:
            if (diff < 0):
                v[i] = y[ind-1] + (z[i]-x[ind-1])/(x[ind]-x[ind-1])*(y[ind]-y[ind-1])
            else:
                v[i] = y[ind] + (z[i]-x[ind])/(x[ind+1]-x[ind])*(y[ind+1]-y[ind])

def checkrow1(A):
    # This function takes a matrix as input and checks if each row sums to one.
    # It returns nothing, but raises an error if the sum of any row is not 1.
    if not np.allclose(np.sum(A, axis=1), 1):
        raise ValueError("checkrow1: Rows of the input matrix do not sum to one.")

def quantilweighted(series, weights, qprop):
    lvec = len(series)
    seriesord = np.argsort(series)
    weightord = weights[seriesord]
    csum = np.cumsum(weightord) - 0.5* weightord  # compute the cumulative sum of weights
    cuth = np.argwhere(csum > qprop)[0][0] if np.any(csum > qprop) else lvec  # find where the required quantile falls
    if cuth == 1:   # if the required quantile is below the first, then set it to the first
        quant = series[0]
    elif qprop >= csum[lvec-1]:  # above the last
        quant = seriesord[lvec-1]
    else:  # interpolate between the closest gridpoints
        quant = seriesord[cuth-1] + (qprop - csum[cuth-1]) / (csum[cuth] - csum[cuth-1]) * (seriesord[cuth] - seriesord[cuth-1])
    return quant


# INITIALIZE STUFF

relax = 0.0 # relaxation parameter for val. funcs (weight on OLD iteration)
relaxgov = 0.0 # relax par for tau. weight on OLD iteration
pertgov = 0.005
epsimin = 4e-05
epsinvmin = 1e-06
epsirmin = 1e-04
epsigovmin = 1e-04
penalty = -1e+7
iterarmax = 50
iteragovmax = 20

invm = 1.0 / nstates

# initialize val fns saved to see if we can speed up



# Gouveia Strauss parameters
btaxw = 0.32
staxw = 0.2439
ptaxw = 0.8179
btaxe = 0.2562
staxe = 0.0
ptaxe = 1.4
# way to impose proportional taxation?
# staxw = 1e8
# staxe = 1e8
# btaxe = 0.25
# btaxw = 0.18

avgywsim = 1.33
avgyesim = 1.33
# stax is estimated on inc/25000
# so staxest*(y/25000)**p = staxbar*(45/25)**p * (y/45000)**p
staxwbase = staxw * (45 / 25) ** ptaxw

#note: basically we are normalizing to avg inc w also for e!
staxebase = staxe * ((45/25)**ptaxe)
staxw = staxwbase * (avgywsim ** (-ptaxw))
staxe = staxebase * (avgyesim ** (-ptaxe))


# Load stuff
fname = "C:\\Users\\Administrator\\Desktop\\yentr2"
# reading income and transitions
with open(fname, "r") as f:
    # Read the first line into variable y
    y = f.readline().strip()
    # Split the rest of the lines into a numpy array Py
    Py = np.loadtxt(f)

# Convert y into a numpy array of floats
y = np.array(y.split(), dtype=float)
#(y)
# we'll normalize the submatrix below
Py /= Py.sum(axis=1)[:, np.newaxis]

"""
def checkrow1(Pr, dr):
    # Function implementation
    pass
"""
#checkrow1(Pr, dr)

gridabreak = 3.0
# Entrepreneurial ability. ALWAYS SET THE FIRST ONE TO ZERO!
r = np.array([0.0, 0.514])
Pr = np.array([[0.964, 0.036], [0.206, 0.794]])

def checkrow1(Pr, dr):
    # Check that rows of Pr sum to 1
    rowsums = np.sum(Pr, axis=1)
    for i in range(dr):
        if abs(rowsums[i] - 1.0) > 1e-6:
            print("WARNING: Row", i+1, "of Pr does not sum to 1.")

#compute invariant distr for Pr
Prtr = Pr.T
eigvalcr, eigveccr = np.linalg.eig(Prtr)
eigvalr = eigvalcr.real
eigvecr = eigveccr.real
imax = np.argmax(eigvalr)
invr = eigvecr[:, imax]
invr /= np.sum(invr)



# Make up joint distribution for y and r. y is outside
Pyr = np.zeros((dy*dr, dy*dr))

for j in range(dy):       # y today
    for jj in range(dy):  # y tomorrow
        Pyr[(j*dr):((j+1)*dr), (jj*dr):((jj+1)*dr)] = Py[j,jj] * Pr

for i in range(dy*dr):
    Pyr[i,:] = Pyr[i,:] / np.sum(Pyr[i,:])

# compute inv dist for joint of y and r
Pyrtr = Pyr.T
eigvalcyr, eigveccyr = np.linalg.eig(Pyrtr)
eigvalyr = eigvalcyr.real
eigvecyr = eigveccyr.real
imax = np.argmax(eigvalyr)
invyr = eigvecyr[:, imax]
invyr = invyr / np.sum(invyr)

# compute inv dist for y
Py = np.array(Py)
Pytr = Py.T
eigvalcy, eigveccy = np.linalg.eig(Pytr)
eigvaly = eigvalcy.real
eigvecy = eigveccy.real
imax = np.argmax(eigvaly)
print(imax)
invy = eigvecy[:, imax] #invy表示y的不变分布
invy = invy / np.sum(invy) #概率分布，如果np.sum(invy)不为1的话




fname = "cacca"
with open(fname, "w") as f:
    f.write("***************************************\n")
    f.write(f"da={da}  dk={dk}  maxa={maxa}\n")
    f.write(f"da1={da1}  da2={da2}  gridabreak={gridabreak}\n")
    f.write(f"bet={bet}  gam={gam}  eta={eta}\n")
    f.write(f"ni={ni}  theta={r[1]}  eff={eff}\n")#这里把2修改成了1
    f.write(f"delt={delt}\n")
    f.write(f"alph={alph}  abig={abig}\n")
    f.write(f"repl rate={replrate}  tauc={tauc}  taua={taua}\n")
    f.write(f"tauls={tauls}  exem={exem}  taub={taub}\n")
    f.write(f"btaxw={btaxw}  staxw={staxw}  ptaxw={ptaxw}\n")
    f.write(f"btaxe={btaxe}  staxe={staxe}  ptaxe={ptaxe}\n")
    f.write(f"gfrac={gfrac}  debtfrac={debtfrac}\n")
    f.write("***************************************\n")

# START COMPUTING
'这里原代码的划分方式是0-da1平缓，da1-da陡'

a[0:da1] = np.linspace(mina, gridabreak, da1) #mina=0,maxa=,gridabreak = 3.0
a[da1:da] = np.linspace(0.0, np.sqrt(maxa - gridabreak), da2) ** 2 + gridabreak
"""尝试更加平缓的打点
acutoff=maxa*da1/da
a[0:da1] = np.linspace(mina, acutoff, da1) #mina=0,maxa=,gridabreak = 3.0
a[da1:da] = np.linspace(0.0, np.sqrt(maxa - acutoff), da2) ** 2 + acutoff
"""

#k表示投资水平
k[:da1] = np.linspace(mink, gridabreak, da1)
k[da1:da] = np.linspace(0.0, np.sqrt(maxk - gridabreak), da2) ** 2 + gridabreak

rbarmin=0.063
rbarmax=0.067


anet = a
anet = np.where(anet > exem, (a - exem) * (1 - taub) + exem, anet)  #exem:遗产税免税水平 #收了遗产税后的财富水平

# loop for government TBD
# *********************
iteragov = 1
taubal = 0.1415
# initialize bounds at very large values
taubalinf = 0.0
govbalinf = -10
taubalsup = 1.0
govbalsup = 10
epsigov = 1.0

# DO WHILE loop for government TBD
while (epsigov > epsigovmin) and (iteragov <= iteragovmax): #epsigov初始值为1，收敛值控制为1e-4,；迭代次数为20
    # loop for rbar
    # init stuff for rbar loop INSIDE govt bc
    epsir = 10
    if iteragov > 1:
        rbarmin = rbar - 0.002
        rbarmax = rbar + 0.002
    #fundiff =
    fundiff=np.zeros(50)
    iterar = 1
    bracket = 1
    noneedrbarmax = 0
    # loop on equilibrium interest rate
    while (epsir > epsirmin) and (iterar <= iterarmax): #epsir收敛值为1e-4，初始值为10，迭代次数为50
        # false position method. first we must compute the value of fundiff in the extremes
        # (bracket=1 and 2), then we start interpolating the extremes
        if bracket == 1: #bracket的含义
            rbar = rbarmin
        elif (bracket == 2) and (noneedrbarmax == 0):
            rbar = rbarmax
        else:
            # this should be the zero for the linear interpolation
            rbar = rbarmin - (rbarmax - rbarmin) * fundiffmin / (fundiffmax - fundiffmin)
        wage = (1 - alph) * abig * ((rbar + delt) / (alph * abig)) ** (alph / (alph - 1))
        'transf表示养老金'
        transf = replrate * wage * np.dot(y, invy) #共收取的养老金


        'we now compute U(c) since it does not depend on V and borrowing constr'
        '老年退休个体的效用'
        'ytaxe：当期个人所得'
        '老年退休个体的消费效用'
        uconsold = np.zeros((da, da))    # old retired.   rows=a, column=a'
        for i in range(da):  # 固定今天的资产a
            ytaxo = transf + (1 - indtaua) * rbar * a[i]
            taxo = (btaxw - btaxw * (staxw * ytaxo ** ptaxw + 1) ** (-1 / ptaxw)) * ytaxo \
                   + indtaua * rbar * a[i] * taua
            cs = (a[i] * (1 + rbar) + transf - a - tauls - taxo) / (1 + tauc)
            '当期的资产是a[i]，养老金是p，下期的资产是a，计算得到当期消费cs'
            'tauls表示总额税，设为0； taxo tauc表示消费税'
            ucons=np.where(cs>0,(cs**(1-gam))/(1-gam),penalty)
            'penalty = -1e+7'
            uconsold[i,:]=ucons

        '老年企业家的效用'
        uconsolde = np.zeros((da, da, dr, dk))  # old entrepreneur staying entr (da,da',dr,dk)
        for i in range(da):  # today's assets
            for j in range(dr):  # today's r r表示投资能力
                ytaxe = r[j] * k ** ni - delt * k - rbar * (k - a[i]) #当期收入
                #those with r=0 can have negative income. set taxes to zero in such case
                taxe = np.where(ytaxe > 0.0, (btaxe - btaxe * (staxe * ytaxe ** ptaxe + 1) ** (-1 / ptaxe)) * ytaxe,
                                0.0)
                for jj in range(da):  # tomorrow's a'
                    csl = (ytaxe - taxe + a[i] - a[jj] - tauls) / (1 + tauc)
                    uconsl = np.where(csl > 0.0, (csl ** (1 - gam)) / (1 - gam), penalty)
                    uconsolde[i, jj, j, :] = uconsl
        '年轻工人的效用'
        uconsw = np.zeros((da, da, dy))   #young worker  (da,da',dy) note: does not depend on r!
        for i in range(da):
            for j in range(dy):
                ytaxw = wage * y[j] + (1 - indtaua) * rbar * a[i]
                taxw = (btaxw - btaxw * (staxw * ytaxw ** ptaxw + 1) ** (-1 / ptaxw)) * ytaxw + indtaua * rbar * a[
                    i] * taua + taubal * ytaxw
                cs = ((1 + rbar) * a[i] + wage * y[j] - taxw - a - tauls) / (1 + tauc)
                ucons = np.where(cs > 0, cs ** (1 - gam) / (1 - gam), penalty)
                uconsw[i, :, j] = ucons


        '年轻企业家的效用'
        uconse = np.zeros((da, da, dr,dk))  # young e     (da,da',dr,dk)
        for i in range(da):  # today's assets
            for j in range(1, dr):  # today's r ! changed from dy=1,dr
                # nnnnn j1 does not appear to be used here
                # for j1 in range(dy):    # today's y
                ytaxe = r[j] * k ** ni - delt * k - rbar * (k - a[i])
                taxe = (btaxe - btaxe * (staxe * ytaxe ** ptaxe + 1) ** (-1 / ptaxe)) \
                       * ytaxe  # cricri +taubal*ytaxe
                for jj in range(da):  # tomorrow's a'
                    csl = (ytaxe - taxe + a[i] - a[jj] - tauls) / (1 + tauc)
                    uconsl = np.where(csl > 0.0,
                                      (csl ** (1.0 - gam)) / (1.0 - gam),
                                      penalty)
                    uconse[i, jj, j-1, :] = uconsl

        epsi = 10
        # investment limit iniat. start over at the max k for any given rbar.
        # later on we will check if there is a more efficient way to do it
        #kyhat = np.full((da, dk, dy), dk) fortran原代码不懂，但显然这里是错的
        #kohat = np.full((da, dy), dk)

        # and impose that people with r=0 cannot borrow (it would be true anyway)
        # might as well save computations
        """
        kyhat = np.zeros((da, dy, dr), dtype=int)
        kohat = np.zeros((da, dr), dtype=int)
        """
        kyhat[:, :, 0] = 0
        kohat[:, 0] = 0

        iterakhat = 1
        epsihat = 1


        #值函数的计算方法有问题
        while epsihat > 0:  # loop on endogenous borrowing constraints 内生借贷约束
            itera = 0
            epsihat = 0
            # start value function iterations, given khat for young and old
            epsi = 10
            if iterakhat <= nite and iterar != 0:
                'iterakhat表示迭代次数,nite=6最多迭代6次就跳出循环'
                'da,dy,dr'
                """
                Vyer = np.zeros((da, dy, dr, nite)) 年轻企业家
                Vywr = np.zeros((da, dy, dr, nite)) 年轻工人
                Vyr = np.zeros((da, dy, dr, nite)) 年轻人
                Voeer = np.zeros((da, dr, nite))  老年企业家
                Vowr = np.zeros((da, nite))  老年工人
                Voer = np.zeros((da, dr, nite))  老年人
                """
                Vye = Vyer[:, :, :, iterakhat]
                Vyw = Vywr[:, :, :, iterakhat]
                Vy = Vyr[:, :, :, iterakhat]

                Voee = Voeer[:, :, iterakhat]
                Vow = Vowr[:, iterakhat]
                Voe = Voer[:, :, iterakhat]
            while epsi > epsimin:  # loop on value funcs for given b.c epsi=10,epsimin=4e-05
                itera = itera + 1
                # compute expected value of V of NEWBORNS
                # only NEWBORN ENTREPRENEUR, INHERITS parent's r'
                for i in range(dy):  # tomorrow
                    for j in range(dr):
                        '这里Vy的dy和dr都给定，只有da变化'
                        'Vyi表示一组dy和dr下不同的da的值函数'
                        '计算新生儿的值函数的期望'
                        Vyi = Vy[:, i, j]
                        #Vynet = np.interp(anet, a, Vyi)
                        Vynet = np.interp(Vyi, a, anet) #表示收遗产税后，后代获得的a的格点
                        "这里的插值有问题，"
                        #print(Vynet)
                        EVnewbw += invyr[(i) * dr + j] * Vynet
                        #给定的r和y对应的概率*Vynet
                        EVnewbe[:, j] += invy[i] * Vynet
                #print(EVnewbw)

                # loop for OLD WORKER
                for i in range(da):  # today's assets
                    Vowtemp = uconsold[i, :] + bet * pold * Vow + eta * bet * (1.0 - pold) * EVnewbw
                    #pold下一期仍然是老人的概率
                    imax = np.argmax(Vowtemp)#给定当期资产ai，老工人的效用最大值
                    newVow[i] = Vowtemp[imax]
                    apolow[i] = imax
                #print(apolow)

                # loop for old entrepreneur
                # entrepreneur staying entrepreneur
                for i in range(da):  # today's assets
                    for j in range(dr):  # today's r
                        for jj in range(da):  # tomorrow's a'
                            Voeetemp[:, jj] = uconsolde[i, jj, j, :] + \
                                              bet * pold * np.dot(Voe[jj, :], Pr[j, :]) + \
                                              eta * bet * (1.0 - pold) * np.dot(EVnewbe[jj, :], Pr[j, :])
                            # impose kohat
                            if kohat[i, j] < dk:
                                # ** when cannot borrow (kohat=last at which one can borrow)
                                hcy=int(kohat[i, j] + 1)
                                Voeetemp[hcy:, jj] = penalty


                        imaxmat = np.unravel_index(np.argmax(Voeetemp), Voeetemp.shape)
                        Voee[i, j] = Voeetemp[imaxmat]
                        if Voee[i, j] > newVow[i]:
                            newVoe[i,j] = Voee[i, j]
                            apoloe[i,j] = imaxmat[1]
                            kpoloe[i,j] = imaxmat[0]
                        else:
                            #print(i)
                            #print(len(newVow))
                            #print(len(newVoe))
                            newVoe[i,j] = newVow[i]
                            apoloe[i,j] = apolow[i]
                            kpoloe[i,j] = -1
                for i in range(dy):  # today
                    for j in range(dr):
                        for i2 in range(dy):  # tomorrow
                            for j2 in range(dr):
                                #print(EVy[:, i, j])
                                #print(Pyr[(i - 1) * dr + j, (i2 - 1) * dr + j2])
                                #print(Vy[:, i2, j2])
                                EVy[:, i, j] += Pyr[(i - 1) * dr + j, (i2 - 1) * dr + j2] * Vy[:, i2, j2]


                # val fn for young that is a worker for the period
                for i in range(da):  # today
                    for j in range(dy):  # today's y
                        for jj in range(dr):  # todays' r
                            Vywtemp = uconsw[i, :, j] + bet * pyou * EVy[:, j, jj] + \
                                      bet * (1.0 - pyou) * newVow
                            imax = np.argmax(Vywtemp)
                            newVyw[i, j, jj] = Vywtemp[imax]
                            apolyw[i, j, jj] = imax


                # NOTE THAT THE ** YOUNG** GUYS WITH ZERO ENTR ABILITY
                # ALWAYS CHOOSE TO BE WORKERS, BECAUSE THIS WAY THEY GET
                # THE WAGE. LET'S EXPLOIT THIS
                # val fn for young that is a ENTR for the period
                # LET US COMPUTE IT ONLY FOR GUYS WITH POSITIVE R
                # young entrepreneur decisions
                for i in range(da):  # today's assets
                    for j in range(1, dr):  # today's r ! changed from dy=1,dr
                        for j1 in range(dy):  # today's y
                            for jj in range(da):  # tomorrow's a'
                                # changed index in uncose for j nnnnn
                                Vyetemp[:, jj] = uconse[i, jj, j - 1, :] + \
                                                 bet * pyou * EVy[jj, j1, j] + \
                                                 bet * (1.0 - pyou) * np.dot(newVoe[jj, :], Pr[j, :])
                                # impose kyhat
                                if kyhat[i, j1, j] < dk:
                                    hzy=int(1 + kyhat[i, j1, j])
                                    Vyetemp[hzy:, jj] = penalty

                            imaxmat = np.unravel_index(np.argmax(Vyetemp), Vyetemp.shape)
                            newVye[i, j1, j] = Vyetemp[imaxmat]
                            apolye[i, j1, j] = imaxmat[1]
                            kpolye[i, j1, j] = imaxmat[0]
                            # YOUNG decide if e or w
                            if newVye[i, j1, j] > newVyw[i, j1, j]:
                                newVy[i, j1, j] = newVye[i, j1, j]
                                apoly[i, j1, j] = apolye[i, j1, j]
                                kpoly[i, j1, j] = kpolye[i, j1, j]
                            else:
                                newVy[i, j1, j] = newVyw[i, j1, j]
                                apoly[i, j1, j] = apolyw[i, j1, j]
                                kpoly[i, j1, j] = -1
                # now use the fact that YOUNG guys with r=0 always choose
                # to be workers
                newVye[:, :, 0] = newVyw[:, :, 0]
                apolye[:, :, 0] = apolyw[:, :, 0]
                kpolye[:, :, 0] = -1
                newVy[:, :, 0] = newVyw[:, :, 0]
                apoly[:, :, 0] = apolyw[:, :, 0]
                kpoly[:, :, 0] = -1

                epsi = np.max(np.abs(newVow - Vow)) + np.max(np.abs(newVoe - Voe)) + \
                       np.max(np.abs(newVye - Vye)) + np.max(np.abs(newVyw - Vyw)) + \
                       np.max(np.abs(newVy - Vy))
                #print(itera, epsi)
                Vow = (1.0 - relax) * newVow + relax * Vow
                Voe = (1.0 - relax) * newVoe + relax * Voe
                Vyw = (1.0 - relax) * newVyw + relax * Vyw
                Vye = (1.0 - relax) * newVye + relax * Vye
                Vy = (1.0 - relax) * newVy + relax * Vy

            # 现在检查是否借钱运行的诱惑
            # 老工人在违约后开始的价值函数（具有资产k*eff）
            interplin(da, a, Vow, dk, eff * k, Vokeff)
            for i in range (da):
                for j in range(1, dr):
                    count1 = 1
                    count2 = 1
                    while (count1 > 0):
                        if (Voe[i,j] < Vokeff[count2]):
                            newkohat[i][j]= count2 - 1
                            count1 = 0
                        count2 = count2 + 1
                        if (count2 == dk):
                            newkohat[i][j] = dk-1
                            count1 = 0

            # 对于r=0的人，我们知道不会借钱
            newkohat[:, 1] = 0
            epsihato = np.max(np.abs(newkohat - kohat))
            print("MAXVAL(newkohat-kohat)", np.max(newkohat - kohat))
            kohat = newkohat

            # 违约后开始的年轻工人的价值函数（具有资产k*eff）
            for j1 in range(dy):
                for j in range(dr):
                    interplin(da, a, Vyw[:, j1, j], dk, eff * k, Vwkeff[:, j1, j])

            for i in range(da):
                for j1 in range(dy):
                    for j in range(1, dr):
                        count1 = 1
                        count2 = 1
                        while (count1 > 0):
                            if (Vye[i, j1, j] < Vwkeff[count2, j1, j]):
                                newkyhat[i][j1][j]= count2 - 1
                                count1 = 0
                            count2 = count2 + 1
                            if (count2 == (dk)):
                                newkyhat[i][j1][j] = dk-1
                                count1 = 0

            # impose no borrowing for people with r=0, which we know would be true
            newkyhat[:, :, 0] = 0
            epsihaty = np.max(np.abs(newkyhat - kyhat))
            print("MAXVAL(newkyhat-kyhat)", np.max(newkyhat - kyhat))
            kyhat = newkyhat
            epsihat = epsihato + epsihaty
            print("iterakhat", iterakhat, " epsihato", epsihato, " epsihaty", epsihaty)
            if (iterakhat <= nite):
                Vyer[:, :, :, iterakhat] = Vye
                Vywr[:, :, :, iterakhat] = Vyw
                Vyr[:, :, :, iterakhat] = Vy
                Voeer[:, :, iterakhat] = Voee
                Vowr[:, iterakhat] = Vow
                Voer[:, :, iterakhat] = Voe
            iterakhat = iterakhat + 1

        # constructing the TRANSITION MATRIX in sparse form
        # sparse=row indices, col indices, values
        #
        # order of the transition matrix:
        # y workers-        r - y - assets
        # y entrepreneurs-  r - y - assets
        # old entr          r -     assets
        # old                       assets
        #
        """
        rowM = 0
        colM = 0
        valM = 0.0
        """
        # ***************** assets net of bequest taxes

        for i in range(da):
            if (a[int(apolow[i])] <= exem):
                apolownet[i] = apolow[i]
            else:
                indanet = np.argmin(np.abs(a - (1 - taub) * (a[apolow[i]] - exem) - exem))
                apolownet[i] = indanet

        for i in range(da):
            for j in range(dr):
                if (a[int(apoloe[i, j])] <= exem):
                    apoloenet[i, j] = apoloe[i, j]
                else:
                    indanet = np.argmin(np.abs(a - (1 - taub) * (a[int(apoloe[i, j] - exem)]) - exem))
                    apoloenet[i, j] = indanet

        counter = nonzero - (da * (dy * dr + 1))   # number of elements for the old

        # block transition oldw - ?
        for i in range(da):  # asset today
            crow = sizeM - da + i  # row of M

            # if reborn
            for ll in range(dr):  # r tomorrow
                for jj in range(dy):  # y tomorrow

                    rowM[int(counter)] = crow

                    colM[counter] = (ll) * dy * da + (jj) * da + apolownet[i]

                    valM[counter] = invyr[(jj) * dr + ll] * (1.0 - pold)

                    counter = counter + 1

            # if remains old

            rowM[counter] = crow

            colM[counter] = 2 * nyoung + noe + apolow[i]

            valM[counter] = pold

            counter = counter + 1
        #block transition old entr
        counter = 2 * (dy * dr + dr + 1) * nyoung
        for l in range(dr):  # r today
            for i in range(da):  # a today
                crow = 2 * nyoung + (l - 1) * da + i
                # if reborn
                for ll in range(dr):  # r', the one the children inherit
                    for jj in range(dy):  # child's y, drawn from invariant distrbn
                        rowM[counter] = crow
                        # if remain entr, this is col
                        colM[counter] = (ll) * dy * da + (jj) * da + apoloenet[i, l]
                        # if become work, add some elements to col
                        if kpoloe[i, l] > 0:
                            colM[counter] += nyoung
                        valM[counter] = (1.0 - pold) * invy[jj] * Pr[l, ll]
                        counter += 1

                # if remains old
                if kpoloe[i, l] == -1:  # becomes worker
                    rowM[counter] = crow
                    colM[counter] = 2 * nyoung + noe + apoloe[i, l]
                    valM[counter] = pold
                    counter += 1
                    for ll in range( dr):
                        rowM[counter] = crow
                        colM[counter] = 2 * nyoung + (ll ) * da + apoloe[i , l ]
                        valM[counter] = 0.0
                        counter += 1
                else:  # remains entrepreneur
                    rowM[counter] = crow
                    colM[counter] = 2 * nyoung + noe + apoloe[i , l ]
                    valM[counter] = 0.0
                    counter += 1
                    for ll in range(dr):
                        rowM[counter] = crow
                        colM[counter] = 2 * nyoung + (ll) * da + apoloe[i, l]
                        valM[counter] = Pr[l, ll] * pold
                        counter += 1
        #block transition young work-
        counter = 0
        for l in range(dr):  # today
            for j in range(dy):  # today
                for i in range(da):  # today
                    crow = (l) * da * dy + (j) * da + i
                    #   if next period remains young
                    for ll in range(dr):  # tomorrow
                        for jj in range(dy):
                            rowM[counter] = crow
                            # if remain work, this is col
                            colM[counter] = (ll) * dy * da + (jj) * da + apoly[i, j, l]
                            #checkcolM(1)
                            # if become entr, add some elements to col
                            if kpoly[i, j, l] > 0:
                                colM[counter] = colM[counter] + nyoung
                                #checkcolM(2)
                            valM[counter] = pyou * Pyr[(j) * dr + l, (jj) * dr + ll]
                            counter = counter + 1
                    #   if next period becomes old
                    if kpoly[i, j, l] == -1:  # if remains worker
                        rowM[counter] = crow
                        colM[counter] = 2 * nyoung + noe + apoly[i, j, l]
                        #checkcolM(3)
                        valM[counter] = (1 - pyou)
                        counter = counter + 1
                        for ll in range(dr):
                            rowM[counter] = crow
                            colM[counter] = 2 * nyoung + (ll) * da + apoly[i, j, l]
                            #checkcolM(4)
                            valM[counter] = 0.0
                            counter = counter + 1
                    else:  # if becomes entr
                        rowM[counter] = crow
                        colM[counter] = 2 * nyoung + noe + apoly[i, j, l]
                        #checkcolM(5)
                        valM[counter] = 0.0
                        counter = counter + 1
                        for ll in range(dr):
                            rowM[counter] = crow
                            colM[counter] = 2 * nyoung + (ll) * da + apoly[i, j, l]
                            #checkcolM(6)
                            valM[counter] = Pr[l, ll] * (1 - pyou)
                            counter = counter + 1

        #block young entr
        if counter != nyoung * (dy * dr + dr + 1):
            print("counter does not match!!")
        for l in range(dr):
            for j in range(dy):
                for i in range(da):
                    crow = nyoung + (l) * da * dy + (j) * da + i
                    for ll in range( dr):
                        for jj in range(dy):
                            rowM[counter] = crow
                            colM[counter] = (ll) * dy * da + (jj) * da + apoly[i, j, l]
                            if kpoly[i, j, l] > 0:
                                colM[counter] += nyoung
                            valM[counter] = pyou * Pyr[(j) * dr + l, (jj ) * dr + ll]
                            counter += 1
                    #if next period becomes old
                    if kpoly[i, j, l] == -1:  # if becomes worker
                        rowM[counter] = crow
                        colM[counter] = 2 * nyoung + noe + apoly[i, j, l]
                        valM[counter] = 1 - pyou
                        counter += 1
                        for ll in range(dr):
                            rowM[counter] = crow
                            colM[counter] = 2 * nyoung + (ll) * da + apoly[i, j, l]
                            valM[counter] = 0.0
                            counter += 1
                    else:
                        rowM[counter] = crow
                        colM[counter] = 2 * nyoung + noe + apoly[i, j, l]
                        valM[counter] = 0.0
                        counter += 1
                        for ll in range(dr):
                            rowM[counter] = crow
                            colM[counter] = 2 * nyoung + (ll) * da + apoly[i, j, l]
                            valM[counter] = Pr[l, ll] * (1 - pyou)
                            counter += 1

        # CALL checksumrowM()
        # Commented out as this is not necessary in Python

        # compute invariant distribution
        itera = 0
        epsinv = 10
        while epsinv > epsinvmin:
            invm = np.zeros(nstates)
            invm1 = invm
            # this do loop is the product M'*invm, M sparse
            # to transpose M, we simply use colM instead of rowM
            # invm= new inv distr
            # invm1= old inv distr
            for i in range(nonzero):
                hcy=int(colM[i] - 1)
                hzy=int(rowM[i] - 1)
                invm[hcy] += invm1[hzy] * valM[i]
            epsinv = np.max(np.abs(invm - invm1))
            itera += 1

        invm = invm / np.sum(invm)
        print("invdistr computed")
        #print(np.shape(invm))

        # invariant distribution of a on young workers
        prgridyw = np.zeros(da)
        for i2 in range(dy * dr):
            #print(np.shape(invm[(i2 * da):(i2 * da + da)]))
            prgridyw =prgridyw+ invm[(i2 * da):(i2 * da + da)]

        # invariant distribution of a on young entr
        prgridye = np.zeros(da)
        for i2 in range(dy * dr):
            prgridye += invm[(nyoung + (i2 * da)):(nyoung + (i2 * da + da))]

        # invariant distribution of a on old entr
        prgridoe = np.zeros(da)
        for i2 in range(dr):
            prgridoe += invm[(2 * nyoung + (i2 * da)):(2 * nyoung + (i2 * da + da))]

        # compute total number of entrepreneurs (which should also be a fraction,
        # since we normalized total population to be 1)
        totentr = np.sum(prgridye) + np.sum(prgridoe)

        # compute number of workers in the corporate sector
        totL = np.sum(prgridyw)

        # compute number of retirees
        totret = 1 - totentr - totL

        # invariant distribution of a on old workers
        prgridow = invm[nstates - da:]

        # total invariant distr
        prgrid = prgridyw + prgridye + prgridoe + prgridow

        invpolk = np.zeros(nstates-da)  # k pol fn corresponding to each element of invm
        invlevk = np.zeros(nstates-da)  # k level ...
        invrk = np.zeros(nstates-da)  # return corresponding to each element of invm (except old work)

        invkborr = np.zeros(nstates-da)  # amount borrowed by e, kbor=k-a
        invyshe = np.zeros(nstates-da)  # shadow y for e
        ifswitchew =np.zeros(nstates)  # 1 if switch from e to non e (w or ret)
        ifswitchwe = 0  # 1 if switch from w to e
        totaleffL = 0.0  # total efficiency units (.neq.totL because entr choice depends on y).
        counter = 0



        for l in range(dr):  # young workers
            invrk[(l) * da * dy:l * da * dy] = r[l]
            for j in range(dy):
                for i in range( da ):
                    invpolk[counter] = kpoly[i , j , l ]
                    if kpoly[i , j , l ] > 0:
                        invlevk[counter ] = k[kpoly[i, j, l]-1]
                        invkborr[counter ] = k[kpoly[i, j, l] - 1] - a[i]
                        invyshe[counter] = y[j] * wage
                        ifswitchwe[counter] = 1
                    else:
                        toteffL += y[j] * invm[(l) * da * dy + (j) * da + i]
                    counter += 1

        for l in range(dr):  # young entrepreneurs
            invrk[nyoung + (l) * da * dy:nyoung + l * da * dy] = r[l]
            for j in range(dy):
                for i in range(da):
                    invpolk[counter] = kpoly[i, j, l]
                    if kpoly[i, j, l] > 0:
                        invlevk[counter] = k[kpoly[i, j, l] - 1]
                        invkborr[counter] = k[kpoly[i, j, l] - 1] - a[i]
                        invyshe[counter] = y[j] * wage
                    else:
                        ifswitchew[counter] = 1
                        toteffL += y[j] * invm[nyoung + (l) * da * dy + (j) * da + i]
                    counter += 1

        for l in range(dr):  # old entrepreneurs
            invrk[(2 * nyoung + (l) * da):(2 * nyoung + l * da)] = r[l]
            for i in range(da):
                invpolk[counter] = kpoloe[i, l]
                if kpoloe[i, l] > 0:
                    invlevk[counter] = k[int(kpoloe[i, l] - 1)]
                    invkborr[counter] = k[int(kpoloe[i, l] - 1)] - a[i]
                    invyshe[counter] = transf
                else:
                    ifswitchew[counter] = 1
                counter += 1

        #capital EMPLOYED by entreprepr
        print(np.shape(invlevk),np.shape(invm[:(nstates - da)]))
        totk = np.dot(invlevk, invm[:(nstates - da)])
        inck = np.dot(invrk * (invlevk ** ni), invm[:(nstates - da)])
        tota = np.dot(a, prgrid)
        totayw = np.dot(a, prgridyw)
        totaye = np.dot(a, prgridye)
        totaow = np.dot(a, prgridow)
        totaoe = np.dot(a, prgridoe)

        totkcorp = tota / (1 + debtfrac) - totk
        rimplied = abig * alph * (totkcorp / toteffL) ** (alph - 1) - delt
        wageimplied = abig * (1 - alph) * ((rimplied + delt) / (abig * alph)) ** (alph / (alph - 1))
        govdebt = debtfrac * (totkcorp + totk)
        gdp = wage * toteffL + (rbar + delt) * totkcorp + inck
        beq2gdp = np.dot(a, (prgridoe + prgridow)) * (1 - pold) / gdp
        k2gdp = (totk + totkcorp) / gdp
        totke = np.dot(a, prgridye) + np.dot(a, prgridoe)
        ykshare1 = inck / gdp
        totkborr = np.dot(invkborr, invm[:nstates - da])
        totyshe = np.dot(invyshe, invm[:nstates - da])
        ykshare = (inck - rbar * totkborr) / gdp
        yktotsh = (inck + (rbar + delt) * totkcorp) / gdp
        propewswitch = np.dot(ifswitchew, invm) / (np.sum(prgridye) + np.sum(prgridoe))
        propweswitch = np.dot(ifswitchwe, invm) / np.sum(prgridyw)
        #printinvdistr()

        fundiffnow = rbar - rimplied
        fundiff[iterar] = fundiffnow
        funtota[iterar] = tota
        funtotk[iterar] = totk
        funrbar[iterar] = rbar

        fname = "cacca"
        OpenStatus = 0
        try:
            with open(fname, mode="a") as f:
                f.write("***********************************\n")
                f.write(f"iterar={iterar} iteragov={iteragov}\n")
                f.write(f"RIMPLIED={rimplied} rbar={rbar}\n")
                f.write(f"rbarmin={rbarmin} rbarmax={rbarmax}\n")
                f.write(f"wageimplied={wageimplied}\n")
                f.write(f"fundiffnow={fundiffnow}\n")
                f.write(f"fundiffmin={fundiffmin} fundiffmax={fundiffmax}\n")
                f.write(f"k2gdp={k2gdp} gdp={gdp}\n")
                f.write(f"taubal={taubal} govbal={govbal} (previous)\n")
                f.write(f"tota={tota} totkcorp={totkcorp} totk={totk}\n")
                f.write(f"totayw={totayw} totaye={totaye}\n")
                f.write(f"totaow={totaow} totaoe={totaoe}\n")
                f.write(f"tottaxl={tottaxl} tottaxe={tottaxe}\n")
                f.write(f"totaxa={tottaxa}\n")
                f.write(f"tottaxcw={tottaxcw} totaxce={tottaxce}\n")
                f.write(f"totaxbe={tottaxbe} totaxbw={tottaxbw}\n")
                f.write(f"totincw={totincw} (previous govbal) {govbal}\n")
                f.write("\n\n")
        except OSError:
            print('problems opening', fname)
            OpenStatus = 1

        print("taubal=", taubal, "   k2gdp=", k2gdp)
        print("rbarmin=", rbarmin, "   rbarmax=", rbarmax)
        print("rbar=", rbar, "   rimplied=", rimplied)
        print("fundiffnow=", fundiffnow)
        print("fundiffmin=", fundiffmin, "fundiffmax=", fundiffmax)
        print("tota=", tota, "   totkcorp=", totkcorp)
        print("totayw=", totayw, "  totaye=", totaye)
        print("totaow=", totaow, "  totaoe=", totaoe)
        print("iterar=", iterar, "  iteragov=", iteragov)
        print("**********************")
        print(" ")

        # using bisection algorithm to update rbar
        # bisection algorithm
        if bracket == 1:
            fundiffmin = fundiffnow
            bracket = 2
            if fundiffmin > 0.0:
                print("***************************")
                print("rbarmin gives a positive fundiff=", fundiffnow)
                print("trying another rbarmin")
                print("***************************")
                fname = "cacca"
                with open(fname, "a") as f:
                    f.write("***************************\n")
                    f.write("rbarmin gives a positive fundiff=" + str(fundiffnow) + "\n")
                    f.write("trying another rbarmin\n")
                    f.write("***************************\n")
                bracket = 1
                noneedrbarmax = 1
                fundiffmax = fundiffnow
                rbarmax = rbarmin
                rbarmin = rbarmin - 0.005
            if fundiffmin <= 0.0 and noneedrbarmax == 1:
                bracket = 0
        elif bracket == 2:
            fundiffmax = fundiffnow
            bracket = 0
            if fundiffmax < 0.0:
                print("***************************")
                print("rbarmax gives a negative fundiff=", fundiffnow)
                print("trying another rbarmax")
                print("***************************")
                fname = "cacca"
                with open(fname, "a") as f:
                    f.write("***************************\n")
                    f.write("rbarmax gives a positive fundiff=" + str(fundiffnow) + "\n")
                    f.write("trying another rbarmax\n")
                    f.write("***************************\n")
                bracket = 2
                rbarmin = rbarmax
                rbarmax = rbarmax + 0.005
        else:
            # convergence criterion
            epsir = min(abs(fundiffnow), abs(rbarmax - rbarmin))
            if fundiffnow > 0.0:
                rbarmax = rbar
                fundiffmax = fundiffnow
            else:
                rbarmin = rbar
                fundiffmin = fundiffnow

        #printtotfun()
        iterar = iterar + 1
        print("iterar on rbar", iterar)


    # compute tax revenues
    """
    vectaxcw = [0.0] * n
    vectaxce = [0.0] * n
    vectaxl = [0.0] * n
    vectaxe = [0.0] * n
    vectaxa = [0.0] * n
    vectaxbw = [0.0] * n
    vectaxbe = [0.0] * n
    

    # compute average income workers
    vectotincw = [0.0] * n
    vecwe2yw = [0.0] * n
    vecwe2ye = [0.0] * n
    whoise = [0] * n
    counter = 0
    """
    counter = 0
    for l in range(dr): #young workers
        rhere = r[l]
        for j in range(dy):
            for i in range(da):
                ahere = a[apoly[i,j,l] - 1]
                if kpoly[i,j,l] > 0:
                    khere = k[kpoly[i,j,l] - 1]
                    entinchere = rhere * khere ** ni - delt * khere - rbar * (khere - a[i])
                    vectaxe[counter] = (btaxe - btaxe * (staxe * entinchere ** ptaxe + 1) ** (-1 / ptaxe)) * entinchere
                    vectaxce[counter] = tauc * (entinchere - vectaxe[counter] + a[i] - tauls - ahere) / (1 + tauc)
                    vecwe2ye[counter] = a[i] / entinchere
                    whoise[counter] = 1
                else:
                    winchere = wage * y[j] + (1 - indtaua) * rbar * a[i]
                    vectaxl[counter] = (btaxw - btaxw * (staxw * winchere ** ptaxw + 1) ** (
                                -1 / ptaxw)) * winchere + taubal * winchere
                    vectaxa[counter] = taua * rbar * a[i]
                    vectaxcw[counter] = tauc * ((1 + rbar) * a[i] + wage * y[j] - vectaxl[counter] - vectaxa[
                        counter] - ahere - tauls) / (1 + tauc)
                    vectotincw[counter] = wage * y[j] + rbar * a[i]
                    vecwe2yw[counter] = a[i] / winchere
                counter += 1

    for l in range(dr, 2 * dr):
        rhere = r[l - dr]
        for j in range(dy):
            for i in range(da):
                ahere = a[apoly[i][j][l - dr] - 1]
                if kpoly[i][j][l - dr] > 0:
                    khere = k[k[i][j][l - dr] - 1]
                    entinchere = rhere * khere ** ni - delt * khere - rbar * (khere - a[i])
                    vectaxe[counter] = (btaxe - btaxe * (staxe * entinchere ** ptaxe + 1) ** (-1 / ptaxe)) * entinchere
                    vectaxce[counter] = tauc * (entinchere - vectaxe[counter] + a[i] - tauls - ahere) / (1 + tauc)
                    vecwe2ye[counter] = a[i] / entinchere
                    whoise[counter] = 1

    for l in range(dr):  # old entrepreneurs
        rhere = r[l]
        for i in range(da):
            ahere = a[int(apoloe[i,l] - 1)]
            if kpoloe[i][l] > 0:
                khere = k[kpoloe[i][l] - 1]
                entinchere = rhere * khere ** ni - delt * khere - rbar * (khere - a[i])
                vectaxe[counter] = (btaxe - btaxe * (staxe * entinchere ** ptaxe + 1) ** (
                            -1 / ptaxe)) * entinchere  # cricri +taubal*entinchere
                vectaxce[counter] = tauc * (entinchere - vectaxe[counter] + a[i] - tauls - ahere) / (1 + tauc)
                vectaxbe[counter] = max(0.0, (ahere - exem)) * taub
                vecwe2ye[counter] = a[i] / entinchere
                whoise[counter] = 1
            else:
                winchere = transf + (1 - indtaua) * rbar * a[i]
                vectaxl[counter] = (btaxw - btaxw * (staxw * winchere ** ptaxw + 1) ** (
                            -1 / ptaxw)) * winchere + taubal * winchere
                vectaxa[counter] = taua * rbar * a[i]
                vectaxcw[counter] = tauc * (
                            a[i] * (1 + rbar) + transf - ahere - tauls - vectaxl[counter] - vectaxa[counter]) / (
                                                1 + tauc)
                vectaxbw[counter] = max(0.0, (ahere - exem)) * taub
                vectotincw[counter] = transf + rbar * a[i]
                whoise[counter] = 2
            counter += 1

    for i in range(da):  # old retirees
        ahere = a[int(apolow[i] - 1)]
        winchere = transf + (1 - indtaua) * rbar * a[i]
        vectaxl[counter] = (btaxw - btaxw * (staxw * winchere ** ptaxw + 1) ** (
                    -1 / ptaxw)) * winchere + taubal * winchere
        vectaxa[counter] = taua * rbar * a[i]
        vectaxcw[counter] = tauc * (
                    a[i] * (1 + rbar) + transf - ahere - tauls - vectaxl[counter] - vectaxa[counter]) / (1 + tauc)
        vectaxbw[counter] = max(0.0, (ahere - exem)) * taub
        vectotincw[counter] = transf + rbar * a[i]
        whoise[counter] = 2
        counter += 1

    tottaxl = np.dot(invm, vectaxl)
    tottaxe = np.dot(invm, vectaxe)
    tottaxa = np.dot(invm, vectaxa)
    tottaxcw = np.dot(invm, vectaxcw)
    tottaxce = np.dot(invm, vectaxce)
    tottaxbw = np.dot(invm, vectaxbw) * (1 - pold)
    tottaxbe = np.dot(invm, vectaxbe) * (1 - pold)
    totincw = np.dot(invm, vectotincw) / (1 - totentr)
    we2yw = np.dot(invm, vecwe2yw) / totL
    we2ye = np.dot(invm, vecwe2ye) / totentr
    '''debug接近尾声'''
    nentr = sum(whoise == 1)
    print('nentr',nentr)
    invmeonly=np.zeros(nentr)
    vecwe2yeonly = vecwe2ye[whoise == 1]
    invmeonly = invm[whoise == 1]
    weights = invmeonly / sum(invmeonly)
    print('vecwe2yeonly',len(vecwe2yeonly))
    we2yemedian = quantilweighted(vecwe2yeonly,invmeonly/sum(invmeonly), 0.5)
    #print(nywork)
    nywork = sum(whoise == 0)
    vecwe2ywonly = vecwe2yw[whoise == 0]
    invmywonly = invm[whoise == 0]
    we2ywmedian = quantilweighted(vecwe2ywonly, invmywonly / sum(invmywonly), 0.5)

    #weights = invmywonly / sum(invmywonly)
    #we2ywmedian = np.quantile(vecwe2ywonly, 0.5, weights=weights)
    govbal = tottaxl + tottaxe + tottaxa + tottaxcw + tottaxce + tottaxbw + tottaxbe - gfrac * gdp - rbar * govdebt - transf * (
                totret - np.sum(prgridoe))
    epsigov = abs(govbal)

    if (iteragov == 1) and (epsigov > epsigovmin):
        taubal1 = taubal
        govbal1 = govbal
        if (govbal < 0):
            taubal = taubal + pertgov
        else:
            taubal = taubal - pertgov
        if (govbal < 0):
            govbalinf = govbal
            taubalinf = taubal
        else:
            govbalsup = govbal
            taubalsup = taubal

    if iteragov > 1 and epsigov > epsigovmin:
        taubalold = taubal1
        govbalold = govbal1
        taubal1 = taubal
        govbal1 = govbal
        taubal = taubal - (1.0 - relaxgov) * govbal * (taubalold - taubal) / (govbalold - govbal)

        # sometimes, you may shoot outside of the brackets, in which case it's best to
        # do bisection with the relevant bracket. keep track of taubals closest
        # if taubal < taubalinf:
        #     taubal = taubal1 - govbal1 * (taubalinf - taubal1) / (govbalinf - govbal1)
        # elif taubal > taubalsup:
        #     taubal = taubal1 - govbal1 * (taubalsup - taubal1) / (govbalsup - govbal1)

        if govbal1 < 0.0 and govbal1 > govbalinf:
            govbalinf = govbal1
            taubalinf = taubal1
        if govbal1 > 0.0 and govbal1 < govbalsup:
            govbalsup = govbal1
            taubalsup = taubal1

    iteragov += 1
    print("govbal", govbal, "iteragov", iteragov, "  taubal=", taubal)
    print("totincw=", totincw)
    fname = "cacca"
    with open(fname, mode="a") as f:
        f.write(f"results for: iteragov {iteragov - 1}\n")
        if epsigov > epsigovmin:
            f.write(f"govbal {govbal}  taubal={taubal1}\n")
            f.write(f"new taubal={taubal}\n")
        else:
            f.write(f"govbal {govbal}  taubal={taubal}\n")
            f.write("reached convergence\n")

    print(f"tottaxl= {tottaxl}  tottaxe= {tottaxe} tottaxa= {tottaxa}")
    print(f"tottacw= {tottaxcw} tottace= {tottaxce}")
    print(f"tottabw= {tottaxbw} tottabe= {tottaxbe}")
    print(f"tota= {tota} gdp= {gdp}")
    print(f"totincw= {totincw}")
    print(f"we2yw= {we2yw}   we2ye= {we2ye}")
    print(f"we2ywmedian= {we2ywmedian}   we2yemedian= {we2yemedian}")
    # closing the file is not needed in Python
    staxw = staxwbase * totincw ** (-ptaxw)
    staxe = staxebase * totincw ** (-ptaxe)
    # CCCCCCCCCCCCC change here
    # epsigov=epsigovmin/2.0



fname = "cacca"
with open(fname, mode="a") as file:
    file.write("***********************************\n")
    file.write(f"program stopped in iterar={iterar-1} iteragov={iteragov-1}\n")
    file.write(f"RIMPLIED{rimplied} rbar{rbar}\n")
    file.write(f"rbarmin{rbarmin} rbarmax{rbarmax}\n")
    file.write(f"wageimplied={wageimplied}\n")
    file.write(f"fundiffnow={fundiffnow}\n")
    file.write(f"fundiffmin={fundiffmin} fundiffmax={fundiffmax}\n")
    file.write(f"k2gdp={k2gdp} gdp={gdp}\n")
    file.write(f"taubal{taubal}\n")
    file.write(f"tota={tota} totkcorp={totkcorp} totk={totk}\n")
    file.write(f"totayw={totayw} totaye={totaye}\n")
    file.write(f"totaow={totaow} totaoe={totaoe}\n")
    file.write(f"govbal= {govbal}\n")
    file.write("taxes:\n")
    file.write(f"lab={tottaxl} cap inc={tottaxa} entr={tottaxe}\n")
    file.write("cw,ce,bw,be\n")
    file.write(f"{tottaxcw} {tottaxce} {tottaxbw} {tottaxbe}\n")
    file.write(f"totincw={totincw}\n")
    file.write(f"we2yw={we2yw} we2ye={we2ye}\n")
    file.write(f"we2ywmedian={we2ywmedian} we2yemedian={we2yemedian}\n")
    file.write(" \n")

#待查
#printwe2inc()
#printtotgov()

quantilweighted(vecwe2yw,invm/totL,0.5,we2ywmedian)
quantilweighted(vecwe2ye,invm/totentr,0.5,we2yemedian)


def printvalfun():
    fname = "valfun"
    # saving value functions
    with open(fname, "w") as f:
        for val in [da, dy, dr, dk]:
            f.write(str(val) + "\n")

        for i in range(da):
            f.write(str(a[i]) + "\n")

        for i in range(dy):
            f.write(str(y[i]) + "\n")

        for i in range(dr):
            f.write(str(r[i]) + "\n")

        for i in range(dk):
            f.write(str(k[i]) + "\n")

        for i in range(da):
            for j in range(dy):
                for jj in range(dr):
                    f.write(str(Vy[i][j][jj]) + "\n")

        for i in range(da):
            for j in range(dy):
                for jj in range(dr):
                    f.write(str(Vyw[i][j][jj]) + "\n")

        for i in range(da):
            for j in range(dy):
                for jj in range(dr):
                    f.write(str(Vye[i][j][jj]) + "\n")

        for i in range(da):
            f.write(str(Vow[i]) + "\n")

        for i in range(da):
            for jj in range(dr):
                f.write(str(Voe[i][jj]) + "\n")

        for i in range(da):
            for jj in range(dr):
                f.write(str(Voee[i][jj]) + "\n")

        for i in range(dk):
            f.write(str(Vokeff[i]) + "\n")

        for i in range(da):
            for j in range(dy):
                for jj in range(dr):
                    f.write(str(apoly[i][j][jj]) + "\n")

        for i in range(da):
            for j in range(dy):
                for jj in range(dr):
                    f.write(str(k[i][j][jj]) + "\n")

        for i in range(da):
            for j in range(dy):
                for jj in range(dr):
                    f.write(str(apolye[i][j][jj]) + "\n")

        for i in range(da):
            for j in range(dy):
                for jj in range(dr):
                    f.write(str(kpolye[i][j][jj]) + "\n")

        for i in range(da):
            for j in range(dy):
                for jj in range(dr):
                    f.write(str(apolyw[i][j][jj]) + "\n")

        for i in range(da):
            f.write(str(apolow[i]) + "\n")

        for i in range(da):
            f.write(str(apolownet[i]) + "\n")

        for i in range(da):
            for jj in range(dr):
                f.write(str(apoloe[i][jj]) + "\n")

        for i in range(da):
            for jj in range(dr):
                f.write(str(apoloenet[i][jj]) + "\n")

        for i in range(da):
            for jj in range(dr):
                f.write(str(kpoloe[i][jj]) + "\n")

        for i in range(da):
            for j in range(dy):
                for jj in range(dr):
                    f.write(str(kyhat[i][j][jj]) + "\n")

        for i in range(da):
            for jj in range(dr):
                f.write(str(kohat[i][jj]) + "\n")

def printinvdistr():
    fname = "invdistr"
    # saving invariant distr
    with open(fname, mode='w') as f:
        nstates_str = str(nstates)
        nyoung_str = str(nyoung)
        noe_str = str(noe)
        f.write(nstates_str + "\n")
        f.write(nyoung_str + "\n")
        f.write(noe_str + "\n")
        for i in range(1, nstates + 1):
            f.write(str(invm[i]) + "\n")
        for i in range(1, da + 1):
            f.write(str(prgrid[i]) + "\n")
        for i in range(1, da + 1):
            f.write(str(prgridyw[i]) + "\n")
        for i in range(1, da + 1):
            f.write(str(prgridye[i]) + "\n")
        for i in range(1, da + 1):
            f.write(str(prgridoe[i]) + "\n")
        for i in range(1, da + 1):
            f.write(str(prgridow[i]) + "\n")
        tota_str = str(tota)
        totk_str = str(totk)
        inck_str = str(inck)
        f.write(tota_str + "\n")
        f.write(totk_str + "\n")
        f.write(inck_str + "\n")
        for i in range(1, nstates - da + 1):
            f.write(str(invlevk[i]) + "\n")
        for i in range(1, nstates - da + 1):
            f.write(str(invpolk[i]) + "\n")
        for i in range(1, nstates - da + 1):
            f.write(str(invrk[i]) + "\n")
        k2gdp_str = str(k2gdp)
        ykshare_str = str(ykshare)
        rbar_str = str(rbar)
        f.write(k2gdp_str + "\n")
        f.write(ykshare_str + "\n")
        f.write(rbar_str + "\n")
        for i in range(1, dr + 1):
            f.write(str(invr[i]) + "\n")
        totke_str = str(totke)
        bet_str = str(bet)
        gam_str = str(gam)
        eff_str = str(eff)
        eta_str = str(eta)
        ni_str = str(ni)
        f.write(totke_str + "\n")
        f.write(bet_str + "\n")
        f.write(gam_str + "\n")
        f.write(eff_str + "\n")
        f.write(eta_str + "\n")
        f.write(ni_str + "\n")
        # attention change reading mfile here
        propewswitch_str = str(propewswitch)
        propweswitch_str = str(propweswitch)
        alph_str = str(alph)
        f.write(propewswitch_str + "\n")
        f.write(propweswitch_str + "\n")
        f.write(alph_str + "\n")
        for i in range(1, dy + 1):
            f.write(str(invy[i]) + "\n")
        for i in range(1, dr + 1):
            for j in range(1, dr + 1):
                f.write(str(Pr[i][j]) + "\n")
        yktotsh_str = str(yktotsh)
        totentr_str = str(totentr)
        totret_str  = str(totret)
        totL_str = str(totL)
        wage_str  = str(wage)
        #
        f.write(yktotsh_str + "\n")
        f.write(totentr_str + "\n")
        f.write(totret_str + "\n")
        f.write(totL_str  + "\n")
        f.write(wage_str + "\n")

        beq2gdp_str = str(beq2gdp)
        totkborr_str = str(totkborr)
        totyshe_str = str(totyshe)
        toteffL_str = str(toteffL)
        #
        f.write(beq2gdp_str + "\n")
        f.write(totkborr_str + "\n")
        f.write(totyshe_str + "\n")
        f.write(toteffL_str + "\n")

def printtotfun():
    fname = "funtot"
    # saving tots
    with open(fname, mode="w") as f:
        for i in range(1, iterar+1):
            print(fundiff(i), file=f)
        for i in range(1, iterar+1):
            print(funtota(i), file=f)
        for i in range(1, iterar+1):
            print(funtotk(i), file=f)
        for i in range(1, iterar+1):
            print(funrbar(i), file=f)

def printwe2inc():
    fname = "we2inc"
    # saving tots
    with open(fname, mode="w") as f:
        print(we2ye, file=f)
        print(we2yw, file=f)
def printtotgov():
    fname = "govtax"
    # saving totals for government
    with open(fname, mode="w") as f:
        print(gdp, file=f)
        print(govdebt, file=f)
        print(transf, file=f)
        print(gfrac, file=f)
        print(tottaxl, file=f)
        print(tottaxe, file=f)
        print(tottaxa, file=f)
        print(tottaxcw, file=f)
        print(tottaxce, file=f)
        print(tottaxbw, file=f)
        print(tottaxbe, file=f)
        print(govbal, file=f)
        print(taubal, file=f)

def checksumrowM():
    sumrowM = np.zeros(sizeM)
    for i in range(1, sizeM+1):
        for j in range(1, nonzero+1):
            if rowM(j) == i:
                sumrowM[i-1] += valM(j)
        if abs(sumrowM[i-1] - 1.0) >= 1e-5:
            print(f"sum of row {i} equals {sumrowM[i-1]}")

def checkcolM(posiz):
    fname3 = 'indcolM'
    if colM[counter-1] > nstates:
        with open(fname3, mode="a") as f:
            f.write(f"{posiz} {l} {j} {i} {ll} {jj} {counter} {apoly(i,j,l)}\n")


def linspace(xmin, xmax, npoints):
    lspace = np.zeros(npoints)
    for i in range(npoints):
        lspace[i] = i/(npoints-1)*(xmax-xmin) + xmin
    return lspace




def checkrow1(A, dA):
    for i in range(dA):
        if abs(np.sum(A[i, :]) - 1) > 1e-6:
            print(f"row {i} of matrix doesn't sum to 1")
"""
def quantilweighted(series, weights, qprop):
    lvec = len(series)
    seriesord = np.sort(series)
    iperm = np.argsort(series)
    weightord = weights[iperm]
    csum = np.zeros(lvec)
    csum[0] = weightord[0] / 2
    for i in range(1, lvec):
        csum[i] = csum[i-1] + weightord[i]/2 + weightord[i-1]/2
    cuth = np.where(csum > qprop)[0][0]
    if cuth == 0:
        quant = seriesord[0]
    elif qprop >= csum[-1]:
        quant = seriesord[-1]
    else:
        quant = seriesord[cuth-1] + (qprop - csum[cuth-1]) / (csum[cuth] - csum[cuth-1]) * (seriesord[cuth] - seriesord[cuth-1])
    return quant
"""
