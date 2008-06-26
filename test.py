from utils import *
import model
import inference

params = model.DiagonalConjugate.HyperParams(array(1.),array(2.),array(0.),array(0.01))
print params
m = model.DiagonalConjugate(params)
print m.p_log_predictive(10)
print m.kernel
print m.walk(array(0),array(1))
print m.sample_posterior()

