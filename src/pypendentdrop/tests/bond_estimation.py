import pypendentdrop as ppd

from .. import logfacility

import numpy as np

testdata_filepath = 'src/pypendentdrop/tests/testdata/water_2.tif'
testdata_pxldensity = 57.0
testdata_rhog = 9.81
testdata_roi = [10, 90, 300, 335]

if __name__ == "__main__":
    logfacility.set_verbose(3)

    testradii = np.linspace(.1, 10, 1000)

    De = np.empty_like(testradii)
    Zmax = np.empty_like(testradii)
    Zmax_est = np.empty_like(testradii)

    for i, testradius in enumerate(testradii):
        R, Z = ppd.compute_nondimensional_profile(testradius)
        De[i] = R.max() - R.min()
        Zmax[i] = Z.max()
        Zmax_est[i] = ppd.greater_possible_zMax(testradius)

    import matplotlib
    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(testradii, Zmax, c='k')
    plt.plot(testradii, Zmax_est, c='gray', ls='--')

    plt.figure()
    ax = plt.gca()
    ax.plot(De / testradii, 1/testradii, c='r')
    ax.set_xlabel('De/r0')
    ax.set_ylabel('lc/r0')


    plt.show()

