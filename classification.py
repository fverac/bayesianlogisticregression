
import numpy as np 
import pandas as pd 


from numpy.linalg import inv
from numpy import matmul as mm
from numpy import transpose as tp
from numpy.linalg import eigvals

import matplotlib.pyplot as plt

import difflib

import GPy

datafiles = [
  "irlstest.csv",
  "ionosphere.csv",
  "diabetes.csv",
  "crashes.csv",
  "B.csv",
  "A.csv"
]

yfiles = [
  "labels-irlstest.csv",
  "labels-ionosphere.csv",
  "labels-diabetes.csv",
  "labels-crashes.csv",
  "labels-B.csv",
  "labels-A.csv"
]


datasets = {}

for labelfile in yfiles:
  featurefile = difflib.get_close_matches(labelfile[7:], datafiles, 1 )[0] #this is the corresponding y file
  print(labelfile, featurefile)
  dataname = featurefile[:-4]
  print(dataname)
  dataset = {}
  values = pd.read_csv(featurefile, header = None).values
  valuesy = np.array( list( map(lambda x : x[0], pd.read_csv(labelfile, header = None).values) )  ) #converts 1999,1 array to 1999, array

  print(values.shape, valuesy.shape)
  dataset["x"] = values
  dataset["y"] = valuesy
  datasets[dataname] = dataset


print('done reading in data\n\n\n')



class BayesLogReg:
  def __init__(self, alpha = 1, w = 0):
    self.alpha = alpha
    self.w = None
    self.wprev = w
    self.k = 0
    self.x = None
    self.y = None


    pass

  def train(self, x, y):
    n = x.shape[0]
    d = x.shape[1] + 1
    x = np.append(x,np.ones([len(x),1]),1) #adds a feature of all 1s to account for bias term in weight

    self.x = x
    self.y = y

    self.w = np.zeros(d)
    self.wprev = self.w + 10**-30

    self.update() # perform one iteration to get w and wprev to be different


    while(self.shouldistop() == False):
      self.update()
    

    return self.w
    








    pass
  

  def sigmoid(self, x):
    x = np.clip(x, a_min = -500, a_max = None) #clip x to preserve numerical stability. i.e. e^-800 = 0
    return 1 / (1+np.exp(-x))

  def calc_sigmund(self):
    x = self.x
    w = self.w
    multiplied = mm(x,w)
    sigmund = self.sigmoid(multiplied)
    return sigmund

  def calc_sn(self):
    x = self.x
    d = x.shape[1]
    alpha = self.alpha
    sigmund = self.calc_sigmund()


    weirdo = sigmund * (1-sigmund)

    r = np.diag(weirdo) 

    left = mm(tp(x),r)
    left = mm(left,x)

    right = np.diag( np.ones(d) * alpha )

    added = left+right
    inverted = inv(added)

    return inverted


  def update(self):
    w = self.w
    x = self.x
    y = self.y
    alpha = self.alpha

    sn = self.calc_sn() #check after verifying right

    sigmund = self.calc_sigmund()
    left = mm(tp(x),sigmund-y) #another line where i do multiple thigns
    right = alpha*w
    added = left + right
    increment = mm(sn, added)


    wnew = w - increment
    self.wprev = w
    self.w = wnew
    self.k += 1 #increment the number of iterations
    return w









  def predict(self, test):
    w = self.w
    test = np.append(test,np.ones([len(test),1]),1) #adds a feature of all 1s to account for bias term in weight

    sn = self.calc_sn()

    right = mm(sn, tp(test)) #output is dxn. next either np.diag (mm (left,right) ) or some other way
    sigasq = (test * right.T).sum(-1)

    denom = np.sqrt( 1+(np.pi/8)*sigasq )
    scores = mm(test,w)

    proby1 = self.sigmoid(scores/denom)
    return (proby1 > 0.5) + 0

  def accuracy(self, preds, y):
    return 1 - np.abs(preds - y).sum()/len(y)

  def predwithaccuracy(self, test, y):
    preds = self.predict(test)
    return self.accuracy(preds, y)


    
    
  
  def shouldistop(self):
    w = self.w
    wprev = self.wprev
    if ( np.linalg.norm(w-wprev) / np.linalg.norm(wprev) < 10**-3):
      return True
    if (self.k >= 100):
      return True
    
    return False

  def find_best_param(self,x,y):

    for l in range(10):
      w = self.train(x,y) #calculate new w, also stored as class memberpip 
      squiggle = self.calc_squiggle()

      newalpha = squiggle/np.sum(w*w)
      self.alpha = newalpha
    
    self.w = w
    return self.alpha





  def calc_squiggle(self):
    alpha = self.alpha #alpha would need to be updated every time

    sigmund = self.calc_sigmund()
    weirdo = sigmund * (1-sigmund)
    r = np.diag(weirdo)
    x = self.x
    xtrx = mm( mm(x.T, r) , x )
    lambdas = eigvals(xtrx)
    squiggle = np.sum(lambdas/ (lambdas+alpha) )
    return squiggle
  





# #NOTE testing BAYES
# print('\n\n\n\niterating through all datasets')
# for dataset in datasets:
#   model = BayesLogReg()
#   x = datasets[dataset]['x']
#   y = datasets[dataset]['y']
#   model.train(x,y)
#   vanilla = model.predwithaccuracy(x,y)

#   model = BayesLogReg()
#   bestalpha = model.find_best_param(x,y)
#   datasets[dataset]['bestalpha'] = bestalpha
#   optimal = model.predwithaccuracy(x,y)
#   # print(bestalpha)
#   model = BayesLogReg(alpha = bestalpha)
#   model.train(x,y)
#   # optimal = model.predwithaccuracy(x,y)
#   print(dataset + ' with accuracy', vanilla, optimal)
# # works when testing for all datasets on 


# x = datasets["irlstest"]['x']
# y = datasets["irlstest"]['y']
# model = BayesLogReg()
# bestalpha = model.find_best_param(x,y)
# print(bestalpha)
# model = BayesLogReg(alpha = bestalpha)
# model.train(x,y)
# modselacc = model.predwithaccuracy(x,y)
# print(modselacc)




# # NOTE GPY UNIT TEST
# Xtrain = x
# ytrain = np.expand_dims(y, axis = 1)
# print(ytrain.shape)

# m = GPy.models.GPClassification(Xtrain, ytrain, kernel=GPy.kern.RBF(Xtrain.shape[1], ARD=True), inference_method=GPy.inference.latent_function_inference.laplace.Laplace())
# m.optimize()
# preds = np.squeeze(m.predict(x)[0]) > 0.5
# print('kernel', model.accuracy(preds, y) )
# # print(preds)
# #wtf the kernel is a legend. unstoppable























#NOTE GENERATIVE

class Generative:
  def __init__(self):
    self.w = None
    self.mu0 = None
    self.mu1 = None
    self.covar = None


  #learn w
  def train(self,x,y):
    self.x = x
    self.y = y

    mu0, mu1 = self.calcmu()
    covar = self.calccovar()
    siginv = np.linalg.pinv(covar)

    w = mm( siginv , mu1-mu0)
    
    left = -0.5 * mm( mm(mu1.T, siginv) , mu1 )
    right = 0.5 * mm( mm(mu0.T, siginv) , mu0 )

    w0 = left + right

    self.ww0 = np.append(w, w0)
    self.w = w
    self.w0 = w0
    return self.ww0

    pass
  
  def calcmu(self):
    x = self.x
    y = self.y

    class0mask = y == 0
    class1mask = y == 1

    xofclass0 = x[class0mask]
    xofclass1 = x[class1mask]

    self.mu0 = np.mean(xofclass0,axis=0)
    self.mu1 = np.mean(xofclass1,axis=0)
    return self.mu0,self.mu1


  def calccovar(self):
    x = self.x
    y = self.y
    n = x.shape[0]

    class0mask = y == 0
    class1mask = y == 1

    xofclass0 = x[class0mask]
    xofclass1 = x[class1mask]

    centeredxclass0 = xofclass0 - self.mu0
    centeredxclass1 = xofclass1 - self.mu1

    class0covar = mm(centeredxclass0.T,centeredxclass0)
    class1covar = mm(centeredxclass1.T,centeredxclass1)

    finalcovar = (class0covar+class1covar)/n
    self.covar = finalcovar
    return finalcovar

  def sigmoid(self, x):
    x = np.clip(x, a_min = -500, a_max = None) #clip x to preserve numerical stability. i.e. e^-800 = 0
    return 1 / (1+np.exp(-x))

  def predict(self, test):
    test = np.append(test,np.ones([len(test),1]),1) #adds a feature of all 1s to account for bias term in weight
    ww0 = self.ww0
    scores = mm(test,ww0)
    proby1 = self.sigmoid(scores)
    return (proby1 > 0.5) + 0
  
  def accuracy(self, preds, y):
    return 1 - np.abs(preds - y).sum()/len(y)

  def predwithaccuracy(self, test, y):
    preds = self.predict(test)
    return self.accuracy(preds, y)    






#NOTE GENERATIVE UNIT TEST
# x = datasets["irlstest"]['x']
# y = datasets["irlstest"]['y']
# jenny = Generative()
# jenny.train(x,y)
# print('jennys accuracy', jenny.predwithaccuracy(x,y) )


# print('\n\n\n\niterating through all datasets')
# for dataset in datasets:
#   jenny = Generative()
#   x = datasets[dataset]['x']
#   y = datasets[dataset]['y']
#   jenny.train(x,y)

#   print('jennys accuracy on ' + dataset, jenny.predwithaccuracy(x,y))
















#
#
##
#for every dataset perform model evaluation 

#NOTE
#NOTE
#NOTE
#NOTE
for dataset in datasets:

  x = datasets[dataset]['x']
  y = datasets[dataset]['y']
  m = 10
  n = x.shape[0]

  #initialize the arrays that will hold the accuracy data for every training size
  datasets[dataset]['genmetrics'] = {}
  datasets[dataset]['bayesmetrics'] = {}
  datasets[dataset]['optbayesmetrics'] = {}
  datasets[dataset]['gpmetrics'] = {}
  for trainingsize in np.linspace(n/m, 0.6*n, m, dtype=int):
    datasets[dataset]['genmetrics'][str(trainingsize)] = np.array([])
    datasets[dataset]['bayesmetrics'][str(trainingsize)] = np.array([])
    datasets[dataset]['optbayesmetrics'][str(trainingsize)] = np.array([])
    datasets[dataset]['gpmetrics'][str(trainingsize)] = np.array([0])



  for it in range(30):

    #prepare the data for this round
    testsize = int(0.4*n)
    restsize = n - testsize
    perm = np.random.permutation(n)
    x = x[perm]
    y = y[perm]
    testx = x[0:testsize]
    testy = y[0:testsize]

    restx = x[testsize:]
    resty = y[testsize:]





    #for every training size, rand select a batch and train/predict/record accuracy
    for trainingsize in np.linspace(n/m, 0.6*n, m, dtype=int):
      perm = np.random.permutation(restsize)
      trainingx = (restx[perm])[0:trainingsize]
      trainingy = (resty[perm])[0:trainingsize]
      
      bayesmodel = BayesLogReg()
      bayesmodel.train(trainingx,trainingy)
      bayesaccuracy = bayesmodel.predwithaccuracy(testx, testy)

      optbayesmodel = BayesLogReg()
      optbayesmodel.find_best_param(trainingx,trainingy)
      optbayesaccuracy = optbayesmodel.predwithaccuracy(testx, testy)

      #in the interest of time
      if (it < 5 and dataset != 'A'):
        trainingygp = np.expand_dims(trainingy, axis = 1)
        gpm = GPy.models.GPClassification(trainingx, trainingygp, kernel=GPy.kern.RBF(trainingx.shape[1], ARD=True), inference_method=GPy.inference.latent_function_inference.laplace.Laplace())
        gpm.optimize()
        preds = np.squeeze(gpm.predict(testx)[0]) > 0.5
        gpaccuracy = bayesmodel.accuracy(preds, testy)
        datasets[dataset]['gpmetrics'][str(trainingsize)] = np.append(datasets[dataset]['gpmetrics'][str(trainingsize)], gpaccuracy)


      jenny = Generative()
      jenny.train(trainingx,trainingy)
      genaccuracy = jenny.predwithaccuracy(testx,testy)

      print('accuracy at trainining size: ' , trainingsize, bayesaccuracy,genaccuracy, optbayesaccuracy)#, gpaccuracy)

      datasets[dataset]['genmetrics'][str(trainingsize)] = np.append(datasets[dataset]['genmetrics'][str(trainingsize)], genaccuracy)
      datasets[dataset]['bayesmetrics'][str(trainingsize)] = np.append(datasets[dataset]['bayesmetrics'][str(trainingsize)], bayesaccuracy)
      datasets[dataset]['optbayesmetrics'][str(trainingsize)] = np.append(datasets[dataset]['optbayesmetrics'][str(trainingsize)], optbayesaccuracy)


    #END STEP 3





#Graphing
for dataset in datasets:

  genmetrics = datasets[dataset]['genmetrics']
  bayesmetrics = datasets[dataset]['bayesmetrics']
  optbayesmetrics = datasets[dataset]['optbayesmetrics']
  gpmetrics = datasets[dataset]['gpmetrics']


  trainingsizes = list(map(lambda x : int(x) , datasets[dataset]['genmetrics'].keys() )  ) 
  
  genmeans = []
  genstds = []

  bayesmeans = []
  bayesstds = []

  optbayesmeans = []
  optbayesstds = []

  gpmeans = []
  gpstds = []
  for trainingsize in genmetrics:
    genmean = 1- np.mean( genmetrics[trainingsize] )
    genstd = np.std( genmetrics[trainingsize] )
    bayesmean = 1- np.mean( bayesmetrics[trainingsize] )
    bayesstd = np.std( bayesmetrics[trainingsize] )
    optbayesmean = 1- np.mean( optbayesmetrics[trainingsize] )
    optbayesstd = np.std( optbayesmetrics[trainingsize] )
    gpmean = 1- np.mean( gpmetrics[trainingsize] )
    gpstd = np.std( gpmetrics[trainingsize] )

    genmeans.append(genmean)
    genstds.append(genstd)
    bayesmeans.append(bayesmean)
    bayesstds.append(bayesstd)

    optbayesmeans.append(optbayesmean)
    optbayesstds.append(optbayesstd)
    gpmeans.append(gpmean)
    gpstds.append(gpstd)




  plt.errorbar(trainingsizes, genmeans, yerr = genstds,  label = "generative")
  plt.legend()
  plt.xlabel('training size')
  plt.ylabel('error')
  plt.title("generative test error vs training size: " + dataset)
  plt.show()
  
  plt.figure(1)
  plt.errorbar(trainingsizes, bayesmeans, yerr = bayesstds, label = "bayes")
  plt.legend()
  plt.xlabel('training size')
  plt.ylabel('error') 
  plt.title("bayes test error vs training size: " + dataset)
  plt.show()

  plt.figure(2)
  plt.errorbar(trainingsizes, optbayesmeans, yerr = optbayesstds, label = "optbayes")
  plt.legend()
  plt.xlabel('training size')
  plt.ylabel('error') 
  plt.title("optbayes test error vs training size: " + dataset)
  plt.show()

  plt.figure(3)
  plt.errorbar(trainingsizes, gpmeans, yerr = gpstds, label = "gp")
  plt.legend()
  plt.xlabel('training size')
  plt.ylabel('error') 
  plt.title("gp test error vs training size: " + dataset)
  plt.show()


  
#NOTE
#NOTE
#NOTE
#NOTE



