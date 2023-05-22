import utils
from diff_tem.simulation_grad import Simulation
import taichi as ti
import torch
import logging

_logger = logging.getLogger(__name__)

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    ti.init(arch=ti.cpu)
    args = utils.get_args()
    print("Hello DiffTEM!")
    print(f"file path = {args.input_file}")
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().setLevel(logging.DEBUG)
    simulation = Simulation.construct_from_file(args.input_file)
    simulation.finish_init_()
    simulation.generate_micrographs()
