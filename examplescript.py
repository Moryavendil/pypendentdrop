import pypendentdrop as ppd

# the path to the image file
filepath = './src/pypendentdrop/tests/testdata/water_2.tif'
# The Region Of Interest: roi = [TLx, TLy, BRx, BRy]
# Use roi = None or roi = [None, None, None, None] if you do not care
roi = [10, 90, 300, 335]

# 1. Import the image
import_successful, image = ppd.import_image(filepath)

if not import_successful:
    raise FileNotFoundError(f'Could not import image at {filepath}')

# 2. Choose a threshold (here automatically)
threshold = ppd.best_threshold(image, roi=roi)

# 3. Detect the contour
contour = ppd.detect_main_contour(image, threshold, roi=roi)

# 4. Pixel density of the image, in px/mm
pxldensity = 57.0

# 5. Estimate roughly the parameters of the drop
estimated_parameters = ppd.estimate_parameters(image, contour, pxldensity)

# Set manually an estimation of the capillary length
estimated_parameters.set_l_mm(2.65)

# Print the estimated parameters in the console
estimated_parameters.describe(name='estimated')

# 6. Optimize these parameters
optimization_successful, optimized_parameters = ppd.optimize_profile(contour, estimated_parameters)

if optimization_successful:
    # Print the optimized parameters in the console
    optimized_parameters.describe(name='optimized')

    # Print the bond number corresponding to the drop
    print(f'Bond number: {round(optimized_parameters.get_bond(), 3)}')

    # The density contrast
    rhog = 9.812
    optimized_parameters.set_densitycontrast(rhog)

    # 7. Compute the surface tension
    print(f'Surface tension gamma: {round(optimized_parameters.get_surface_tension(), 3)} mN/m')
else:
    print('Optimization failed :(')

import matplotlib.pyplot as plt
from pypendentdrop import plot

if not optimization_successful:
    fig, (ax1, ax1) = plt.subplots(1, 1)
    plot.plot_image_contour(ax1, image, contour, estimated_parameters, 'estimated', roi=roi)
    plt.savefig('deleteme_estimatedparameters.png', dpi=300)
else:
    ### Plotting a comparison between the estimated and optimized parameters

    fig, (ax1, ax2) = plt.subplots(1, 2)
    plot.plot_image_contour(ax1, image, contour, estimated_parameters, 'estimated', roi=roi)
    plot.plot_image_contour(ax2, image, contour, optimized_parameters, 'optimized', roi=roi)
    plt.savefig('deleteme_comparison.png', dpi=300)
