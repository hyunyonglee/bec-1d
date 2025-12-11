# Copyright 2025 Hyun-Yong Lee

from tenpy.models.lattice import Chain
from tenpy.models.model import CouplingModel, MPOModel
from tenpy.tools.params import Config
from tenpy.networks.site import BosonSite
__all__ = ['BOSE_HUBBARD']


class BOSE_HUBBARD(CouplingModel,MPOModel):
    
    def __init__(self, model_params):
        
        # 0) read out/set default parameters 
        if not isinstance(model_params, Config):
            model_params = Config(model_params, "BOSE_HUBBARD")
        L = model_params.get('L', 10)
        t = model_params.get('t', 0.)
        U = model_params.get('U', 0.)
        Ncut = model_params.get('Ncut', 2)
        
        site = BosonSite( Nmax=Ncut, conserve='N', filling=0.0 )
        site.multiply_operators(['B','B'])
        site.multiply_operators(['Bd','Bd'])

        lat = Chain( L=L, site=site, bc='open', bc_MPS='finite', order='default' )
        CouplingModel.__init__(self, lat)

        # 2-site hopping
        self.add_coupling( -t, 0, 'B', 0, 'Bd', 1, plus_hc=True)
        
        # Onsite Hubbard Interaction
        self.add_onsite( U/2., 0, 'NN')

        MPOModel.__init__(self, lat, self.calc_H_MPO())
