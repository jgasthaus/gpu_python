from utils import *
import model
import inference
import numpy.random as R
import numpy as N
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='debug.log',
                    filemode='w')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# set a format which is simpler for console use
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)


params = model.DiagonalConjugate.HyperParams(array(1.),array(2.),array(0.),array(0.01))
inference_params = inference.InferenceParams(
        rho=0.99,
        alpha=0.001,
        p_uniform_deletion=0.99
        )
#print params
#print inference_params
m = model.DiagonalConjugate(params)

data = R.random_sample((2,100))
data_time = N.cumsum(R.rand(100))

pf = inference.ParticleFilter(
        m,
        data,
        data_time,
        inference_params,
        100
        )
pf.run()
