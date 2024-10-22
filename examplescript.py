import pypendentdrop as ppd
	
filepath = './src/pypendentdrop/tests/testdata/water_2.tif'
pxldensity = 57.0
rhog = 9.812
roi = [10, 90, 300, 335]

importsuccess, img = ppd.import_image(filepath)

if not importsuccess:
    raise FileNotFoundError(f'Could not import image at {filepath}')

threshold = ppd.best_threshold(img, roi=roi)

cnt = ppd.find_mainContour(img, threshold, roi=roi)

estimated_parameters = ppd.estimate_parameters(ppd.image_centre(img), cnt, pxldensity)

estimated_parameters.describe(name='estimated')

opti_success, optimized_parameters = ppd.optimize_profile(cnt, parameters_initialguess=estimated_parameters)

if not opti_success:
    print('optimization failed :(')
else:
    optimized_parameters.describe(name='optimized')

    print(f'Bond number: {round(optimized_parameters.get_bond(), 3)}')

    optimized_parameters.set_densitycontrast(rhog)
    print(f'Surface tension gamma: {round(optimized_parameters.get_surface_tension(), 3)} mN/m')

    ### Plotting a comparison between the estimated and optimized parameters
    import matplotlib.pyplot as plt
    from pypendentdrop import plot

    fig, (ax1, ax2) = plt.subplots(1, 2)
    plot.plot_image_contour(ax1, img, cnt, estimated_parameters, 'estimated', roi=roi)
    plot.plot_image_contour(ax2, img, cnt, optimized_parameters, 'optimized', roi=roi)
    plt.savefig('deleteme_comparison.png', dpi=300)
