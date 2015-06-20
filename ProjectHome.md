## Overview ##

**PyRSVD** provides an efficient [python](http://www.python.org/) implementation of a regularized singular value decomposition solver. The module is primarily aimed at applications in [collaborative filtering](http://en.wikipedia.org/wiki/Collaborative_filtering), in particular the [Netflix competition](http://netflixprize.com).

### Matrix Factorization ###

The solver is used to compute a low-rank approximation of a rating matrix _R_, which is usually a large partial matrix (i.e. lots of missing values).

More formally,

![http://www.sitmo.com/gg/latex/latex2png.2.php?z=80&eq=\mathbf{R}%20\approx%20\mathbf{U}\%2C%20\mathbf{V}^T%20\text{%20where%20}%20\mathbf{R}%20%3D%20m\times%20n%20%2C%20\mathbf{U}%3D%20m%20\times%20k%2C%20%20\mathbf{V}%3D%20n\times%20k%20\text{.%20Usually%2C%20%20}%20k%20%3C%3C%20m%20%3C%20n&onsense=something_that_ends_with.png](http://www.sitmo.com/gg/latex/latex2png.2.php?z=80&eq=\mathbf{R}%20\approx%20\mathbf{U}\%2C%20\mathbf{V}^T%20\text{%20where%20}%20\mathbf{R}%20%3D%20m\times%20n%20%2C%20\mathbf{U}%3D%20m%20\times%20k%2C%20%20\mathbf{V}%3D%20n\times%20k%20\text{.%20Usually%2C%20%20}%20k%20%3C%3C%20m%20%3C%20n&onsense=something_that_ends_with.png)

The goodness of the approximation is measured in terms of the frobenius norm  with respect to the known ratings. Minimizing the frobenius norm between the rating matrix _R_ and the factorization is equivalent to minimize the squared error. Due to the huge number of parameters, overfitting is a serious problem. It is avoided by adding a regularization term to the squared error function, which penalizes large parameters.
The regularized error function is given by,

![http://www.sitmo.com/gg/latex/latex2png.2.php?z=80&eq=\mathcal{L}(\mathbf{U}%2C\mathbf{V})%20%3D%20\sum_{i%2Cj%20\in%20R}%20(r_{i%2Cj}%20-%20\mathbf{U}_i%20\%20%20\mathbf{V}_j^T)%20%2B%20\lambda%20(\sum_i%20\lVert%20\mathbf{U}_i%20\rVert^2%20%2B%20\sum_j%20\lVert%20\mathbf{V}_j%20\rVert^2)&onsense=something_that_ends_with.png](http://www.sitmo.com/gg/latex/latex2png.2.php?z=80&eq=\mathcal{L}(\mathbf{U}%2C\mathbf{V})%20%3D%20\sum_{i%2Cj%20\in%20R}%20(r_{i%2Cj}%20-%20\mathbf{U}_i%20\%20%20\mathbf{V}_j^T)%20%2B%20\lambda%20(\sum_i%20\lVert%20\mathbf{U}_i%20\rVert^2%20%2B%20\sum_j%20\lVert%20\mathbf{V}_j%20\rVert^2)&onsense=something_that_ends_with.png)

The solver uses stochastic gradient decent to minimize the above error function.
<a href='Hidden comment: 
\mathcal{L}(\mathbf{U},\mathbf{V}) = \sum_{i,j \in R} (r_{i,j} - \mathbf{U}_i \  \mathbf{V}_jT) + \lambda (\sum_i \lVert \mathbf{U}_i \rVert2 + \sum_j \lVert \mathbf{V}_j \rVert^2)
'></a>

Matrix approximation has been applied very successfully in collaborative
filtering. The factors reveal some of the latent structure in the rating data which is subsequently used to predict user preferences. The factorization produced by the solver can
directly be used to predict ratings or as a preprocessing step, e.g. to represent each user by a vector of latent factors he or she is interested in.

## Dependencies ##

The python module makes heavy use of [numpy](http://www.scipy.org/NumPy). The critical sections are written in [cython](http://www.cython.org).
Although the module does not depend on [pyflix](http://pyflix.python-hosting.com/), it nicely integrates into the pythonic Netflix library.

## Performance ##

The runtime of the algorithm depends on two parameters: a) the number of latent factors _k_ and b) the number of epochs. The table below shows the performance of the algorithm
on the training set of the Netflix data. The table shows the average time per epoch as well as the RMSE of the final model on the training and probe set. The baseline of Netflix (CineMatch) scores [0.9474](http://www.netflixprize.com/faq) on the probe set (lower is better).

| **Factors** | **Epochs** | **sec/Epoch** | **Train RMSE** | **Probe RMSE** |
|:------------|:-----------|:--------------|:---------------|:---------------|
| 10          | 100        | 34            | 0.8125         | 0.9260         |
| 64          | 100        | 149           | 0.7785         | 0.9165         |
| 128         | 104        | 190| 0.694209       | 0.907149       |
| 256         | 106        | 350| 0.660420       | 0.905564       |

The plot below shows the learning curve of a model with 256 factors (learn rate=0.001, regularization=0.011). The probe RMSE is plotted against the number of training epochs. The ticks on the right mark the performance of a simple movie average predictor, CineMatch and the qualification RMSE for the Grand Prize.

![http://chart.apis.google.com/chart?cht=lxy&chd=t:0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,104|1.015898,0.991847,0.977383,0.966254,0.958173,0.950704,0.944143,0.938950,0.934577,0.930751,0.927352,0.924332,0.921639,0.919228,0.917068,0.915136,0.913411,0.911880,0.910532,0.909358,0.908352,0.907509,0.906825,0.906293,0.905910,0.905670,0.905567&chs=500x300&chds=0,100,0.85,1.055,0.85,1.055&chxt=x,y&chxt=x,y,r&chxr=0,0,100|1,0.85,1.055|2,0.85,1.055&chxl=2:|CineMatch|Average|Win&chxp=2,0.9474,1.0528,0.8572&chxtc=2,-500&chxs=2,,11,-1,lt,AFAFAF&nonsense=foo.png](http://chart.apis.google.com/chart?cht=lxy&chd=t:0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,104|1.015898,0.991847,0.977383,0.966254,0.958173,0.950704,0.944143,0.938950,0.934577,0.930751,0.927352,0.924332,0.921639,0.919228,0.917068,0.915136,0.913411,0.911880,0.910532,0.909358,0.908352,0.907509,0.906825,0.906293,0.905910,0.905670,0.905567&chs=500x300&chds=0,100,0.85,1.055,0.85,1.055&chxt=x,y&chxt=x,y,r&chxr=0,0,100|1,0.85,1.055|2,0.85,1.055&chxl=2:|CineMatch|Average|Win&chxp=2,0.9474,1.0528,0.8572&chxtc=2,-500&chxs=2,,11,-1,lt,AFAFAF&nonsense=foo.png)

## Installation ##
To install the module simply run,
```
python setup.py install
```

If you modify rsvd.pyx you have to run the cython compiler. Cython will create the file rsvd.c. The reference C compiler for the project is GCC 4.2.3. To avoid [structure padding](http://en.wikipedia.org/wiki/Packed) you have to invoke ./instrument.py which adds `__attribute__ ((__packed__))` to the Rating struct.
```
cython rsvd.pyx
./instrument.py rsvd.c
python setup.py install
```


## Usage ##

To train a model, simply use the `RSVD.train` classmethod:
```
import numpy as np
from rsvd import RSVD, rating_t
ratings=np.fromfile('training.arr',dtype=rating_t)
probeRatings=np.fromfile('probe.arr',dtype=rating_t)

model = RSVD.train(10,ratings,(17770,480189),probeRatings)

# predict r_ij, the rating of user j and movie i
model(i,j)

```
For more information on the arguments of `RSVD.train` type `help(RSVD.train)`. The rating data is assumed to be stored in a numpy record array. Each record is a triple (movieID,userID,rating) where movieID is a uint16, userID is a uint32 and rating is a ~~uint8~~ float (see `rating_t`). Furthermore, it is assumed that the movie ids start from 1 whereas the user ids start from 0 (missing user and movie ids are not permitted).
So for the netflix dataset you can leave the movie ids as they are but you have to map the
user ids to the interval [0,480189].

You can also use the `rsvd_train` shell script to train a model. For more information type `rsvd_train --help`.

```
$ ./rsvd_train -f 10 -l 0.001 -r 0.02 --probe data/probe.arr data/training.arr 17770 480189 models/t_10_001_02_100
```

## Further Information ##
PyRSVD trains the factors simultaneously - other approaches train one factor at a time. For further information on factor-at-a-time approaches see:
  * [LingPipe's SVDMatrix](http://alias-i.com/lingpipe/docs/api/com/aliasi/matrix/SvdMatrix.html)
  * [Timely Development](http://www.timelydevelopment.com/demos/NetflixPrize.aspx)
  * [Simon Funk SVD](http://sifter.org/~simon/Journal/20061211.html)

![http://www.python.org/community/logos/python-powered-w-100x40.png](http://www.python.org/community/logos/python-powered-w-100x40.png)
