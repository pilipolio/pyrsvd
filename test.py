"""
Test
----
This is a test
"""

#log# Automatic Logger file. *** THIS MUST BE THE FIRST LINE ***
#log# DO NOT CHANGE THIS LINE OR THE TWO BELOW
#log# opts = Struct({'__allownew': True, 'logfile': 'ipython_log.py'})
#log# args = []
#log# It is safe to make manual edits below here.
#log#-----------------------------------------------------------------------
import numpy as np

f=open('data/ratings.arr','r')
rating=np.dtype('H,I,B')
print 'load ratings'
ratings=np.fromfile(f,dtype=rating)

f.close()
f=open('data/probe_ratings.arr','r')
probeRatings=np.fromfile(f,dtype=rating)
f.close()
from rsvd import RSVD
model=RSVD.train(128,ratings,(17770,480189),probeRatings,150,randomize=False)
print "model trained..."

model.save("models/t_128_001_011_150")


