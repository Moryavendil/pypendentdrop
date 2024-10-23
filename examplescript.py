import pypendentdrop as ppd
	
filepath = './src/pypendentdrop/tests/testdata/water_2.tif'
pxldensity = 57.0
rhog = 9.812
roi = [10, 90, 300, 335] # use roi=None if you do not care about the ROI

importsuccess, image = ppd.import_image(filepath)

if not importsuccess:
    raise FileNotFoundError(f'Could not import image at {filepath}')

threshold = ppd.best_threshold(image, roi=roi)

contour = ppd.find_mainContour(image, threshold, roi=roi)

estimated_parameters = ppd.estimate_parameters(ppd.image_centre(image), contour, pxldensity)

estimated_parameters.set_capillary_length_mm(2.65) # set manually an estimation of the capillary length

estimated_parameters.describe(name='estimated')# print the estimated parameters in the console

opti_success, optimized_parameters = ppd.optimize_profile(contour, estimated_parameters)

if not opti_success:
    print('optimization failed :(')
else:
    optimized_parameters.describe(name='optimized')# print the optimized parameters in the console

    print(f'Bond number: {round(optimized_parameters.get_bond(), 3)}')

    optimized_parameters.set_densitycontrast(rhog)
    print(f'Surface tension gamma: {round(optimized_parameters.get_surface_tension(), 3)} mN/m')

    ### Plotting a comparison between the estimated and optimized parameters
    import matplotlib.pyplot as plt
    from pypendentdrop import plot

    fig, (ax1, ax2) = plt.subplots(1, 2)
    plot.plot_image_contour(ax1, image, contour, estimated_parameters, 'estimated', roi=roi)
    plot.plot_image_contour(ax2, image, contour, optimized_parameters, 'optimized', roi=roi)
    plt.savefig('deleteme_comparison.png', dpi=300)
