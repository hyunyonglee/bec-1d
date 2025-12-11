import numpy as np
import model
from tenpy.algorithms import tdvp
from tenpy.networks.mps import MPS
import os, os.path
import argparse
import logging.config
import h5py
from tenpy.tools import hdf5_io

def ensure_dir(f):
    d=os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)
    return d



def measurements(psi, L):
    
    # Measurements
    Ns = psi.expectation_value("N")
    NNs = psi.expectation_value("NN")
    EE = psi.entanglement_entropy()

    # Measuring Hopping Expectation Values
    Js_real = []
    Js_imag = []
    for i in range(0,L-1): 
        J = psi.expectation_value_term([('Bd',i),('B',i+1)])
        Js_real.append( J.real )
        Js_imag.append( J.imag )
    
    return Ns, NNs, EE, Js_real, Js_imag



def write_data( Ns, NNs, EE, Js_real, Js_imag, time, path ):

    ensure_dir(path+"/observables/")
    ensure_dir(path+"/mps/")

    # data = {"psi": psi}
    # with h5py.File(path+"/mps/psi_time_%.3f.h5" % time, 'w') as f:
    #     hdf5_io.save_to_hdf5(f, data)

    file_EE = open(path+"/observables/EE.txt","a", 1)    
    file_Ns = open(path+"/observables/Ns.txt","a", 1)
    file_NNs = open(path+"/observables/NNs.txt","a", 1)
    file_Js_real = open(path+"/observables/Js_real.txt","a", 1)
    file_Js_imag = open(path+"/observables/Js_imag.txt","a", 1)
    
    
    file_EE.write(repr(time) + " " + "  ".join(map(str, EE)) + " " + "\n")
    file_Ns.write(repr(time) + " " + "  ".join(map(str, Ns)) + " " + "\n")
    file_NNs.write(repr(time) + " " + "  ".join(map(str, NNs)) + " " + "\n")
    file_Js_real.write(repr(time) + " " + "  ".join(map(str, Js_real)) + " " + "\n")
    file_Js_imag.write(repr(time) + " " + "  ".join(map(str, Js_imag)) + " " + "\n")
    
    file_EE.close()
    file_Ns.close()
    file_NNs.close()
    file_Js_real.close()
    file_Js_imag.close()
    

if __name__ == "__main__":
    
    current_directory = os.getcwd()

    conf = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {'custom': {'format': '%(levelname)-8s: %(message)s'}},
    'handlers': {'to_file': {'class': 'logging.FileHandler',
                             'filename': 'log',
                             'formatter': 'custom',
                             'level': 'INFO',
                             'mode': 'a'},
                'to_stdout': {'class': 'logging.StreamHandler',
                              'formatter': 'custom',
                              'level': 'INFO',
                              'stream': 'ext://sys.stdout'}},
    'root': {'handlers': ['to_stdout', 'to_file'], 'level': 'DEBUG'},
    }
    logging.config.dictConfig(conf)

    parser=argparse.ArgumentParser()
    parser.add_argument("--L", default='10', help="Length of chain")
    parser.add_argument("--t", default='1.0', help="Single-particle hopping amplitude")
    parser.add_argument("--U", default='1.0', help="On-site Hubbard interaction")
    parser.add_argument("--chi", default='64', help="Bond dimension")
    parser.add_argument("--Ncut", default='4', help="Cut-off boson number")
    parser.add_argument("--Ntot", default='10', help="Total time steps")
    parser.add_argument("--Mstep", default='5', help="Measurement time step")
    parser.add_argument("--dt", default='0.1', help="Delta time")
    parser.add_argument("--init_state", default='2', help="Initial state")
    parser.add_argument("--path", default=current_directory, help="path for saving data")
    args=parser.parse_args()

    L = int(args.L)
    t = float(args.t)
    U = float(args.U)
    chi = int(args.chi)
    Ncut = int(args.Ncut)
    Ntot = int(args.Ntot)
    Mstep = int(args.Mstep)
    dt = float(args.dt)
    init_state = args.init_state
    path = args.path
    
    model_params = {
    "L": L, 
    "t": t,
    "U": U,
    "Ncut": Ncut
    }

    tdvp_params = {
        'start_time': 0,
        'dt': dt,
        'N_steps': 1,
        'trunc_params': {
            'chi_max': chi,
            'svd_min': 1.e-8,
            'trunc_cut': None
        }
    }

    print("Loading wavefunction from file...")
    # file_path = "/Users/hyunyong-lee/Dropbox/Programs/TENSOR_NETWORK/TENPY/BEC_1D/L%d_init_psi.h5" % int(L)
    file_path = "/home/hylee/bec-1d/L%d_init_psi.h5" % int(L)
    with h5py.File(file_path, 'r') as f:
        data = hdf5_io.load_from_hdf5(f)
        psi = data["psi"]
    print("Wavefunction loaded.")

    Ns, NNs, EE, Js_real, Js_imag = measurements(psi, L)
    write_data( Ns, NNs, EE, Js_real, Js_imag, 0, path )

    BHM = model.BOSE_HUBBARD(model_params)
    tdvp_engine = tdvp.TwoSiteTDVPEngine(psi, BHM, tdvp_params)
    for i in range(Ntot):
        tdvp_engine.run()
        if (i+1) % Mstep == 0:
            Ns, NNs, EE, Js_real, Js_imag = measurements(psi, L)
            write_data( Ns, NNs, EE, Js_real, Js_imag, tdvp_engine.evolved_time, path )