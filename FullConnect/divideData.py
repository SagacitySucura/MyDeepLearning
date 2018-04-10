import os
import os.path
import numpy as np
import linecache
rootdir="H:/deep_learning/stripe_surface/deep_learning/allconnectnet/coord_data/"
#rootdir="H:/deep_learning/ringAtListOne/deep_learning/Covnet/data/"
os.chdir(rootdir)
lines=linecache.getlines('trainDataRandom.txt')
linecache.clearcache()
n=0
for line in lines:
    n+=1
    index=int(n/100)
    filename='trainColloction/train'+str(index)+'.txt'
    filehandle=open(filename,'a',encoding='utf-8')
    filehandle.write(line)
    filehandle.close()