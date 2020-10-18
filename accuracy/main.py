import time

import torch

import accuracy.Best_EI.accuracy as best_ei
import accuracy.Best_USB.accuracy as best_ucb
import accuracy.Best_MV.accuracy as best_mv
import accuracy.Best_qNEI.accuracy as best_qnei
import accuracy.qNEI1_qNEI2.accuracy as qnei_x2
import accuracy.MV_EI.accuracy as mv_ei

if __name__ == '__main__':
    # start = time.time()
    N = 200
    M = 70
    DIM = 6
    NOISE = 0.1
    BOUNDS = torch.stack([torch.zeros(DIM),
                          torch.ones(DIM)])

    best_ei.main(n=N, m=M, dim=DIM, noise=NOISE, bounds=BOUNDS)
    best_ucb.main(n=N, m=M, dim=DIM, noise=NOISE, bounds=BOUNDS)
    best_mv.main(n=N, m=M, dim=DIM, noise=NOISE, bounds=BOUNDS)
    best_qnei.main(n=N, m=M, dim=DIM, noise=NOISE, bounds=BOUNDS)
    mv_ei.main(n=N, m=M, dim=DIM, noise=NOISE, bounds=BOUNDS)
    qnei_x2.main(n=N, m=M, dim=DIM, noise=NOISE, bounds=BOUNDS)

    # elapsed_time = time.time() - start
    # print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
