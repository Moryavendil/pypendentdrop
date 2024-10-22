import pypendentdrop as ppd
	
filepath = './assets/test_data/water_dsc1884.tif'
pxldensity = 57.0
rhog = 9.812
roi = [10, 90, 300, 335]

success, img = ppd.import_image(filepath)

threshold = ppd.best_threshold(img, roi=roi)

cnt = ppd.find_mainContour(img, threshold, roi=roi)

estimated_parameters = ppd.estimate_parameters(ppd.image_centre(img), cnt, pxldensity)

opti_success, optimized_parameters = ppd.optimize_profile(cnt, px_per_mm=pxldensity, parameters_initialguess=estimated_parameters)

gravity_angle, y_tip_position, x_tip_position, r0_mm, capillary_length_mm = optimized_parameters

bond = (r0_mm / capillary_length_mm)**2
print(f'Bond number: {round(bond, 3)}')

gamma = rhog * capillary_length_mm**2
print(f'Surface tension gamma: {round(gamma, 3)} mN/m')

### Plotting a comparison between the estimated and optimized parameters
import matplotlib.pyplot as plt
from ppd import plotresults

fig, (ax1, ax2) = plt.subplots(1, 2)
plotresults.plot_image_contour(ax1, img, cnt, pxldensity, estimated_parameters, 'estimated', roi=roi)
plotresults.plot_image_contour(ax2, img, cnt, pxldensity, optimized_parameters, 'optimized', roi=roi)

plt.savefig('comparison.png', dpi=300)
