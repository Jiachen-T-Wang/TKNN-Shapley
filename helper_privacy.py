from autodp.autodp_core import Mechanism
from autodp.transformer_zoo import Composition 
from autodp import mechanism_zoo, transformer_zoo





class PrivateKNN_mech(Mechanism):
  def __init__(self, prob, sigma, niter, name='private-knn'):
    Mechanism.__init__(self)
    self.name=name
    self.params={'prob':prob, 'sigma':sigma, 'niter':niter}
        
    # create such a mechanism as in previously
    subsample = transformer_zoo.AmplificationBySampling() # by default this is using poisson sampling
    compose = transformer_zoo.Composition()

    mech = mechanism_zoo.GaussianMechanism(sigma=sigma)

    if prob < 1:
      mech = subsample(mech, prob, improved_bound_flag=True)

    mech = compose([mech], [niter])
    rdp_total = mech.RenyiDP
    self.propagate_updates(rdp_total, type_of_update='RDP')




class PrivateKNN_SV_RJ_mech(Mechanism):
  def __init__(self, prob, sigma, niter, K, name='private-knn'):
    Mechanism.__init__(self)
    self.name=name
    self.params={'prob':prob, 'sigma':sigma, 'niter':niter}
        
    # create such a mechanism as in previously
    subsample = transformer_zoo.AmplificationBySampling() # by default this is using poisson sampling
    compose = transformer_zoo.Composition()

    mech = mechanism_zoo.GaussianMechanism(sigma=sigma * (K*(K+1)))

    if prob < 1:
      mech = subsample(mech, prob, improved_bound_flag=True)

    mech = compose([mech], [niter])
    rdp_total = mech.RenyiDP
    self.propagate_updates(rdp_total, type_of_update='RDP')


