from ... import pypendentdrop as ppd

from .. import logfacility

logfacility.set_verbose(3)

testdata_filepath = 'src/pypendentdrop/tests/testdata/water_2.tif'
testdata_pxldensity = 57.0
testdata_rhog = 9.81
testdata_roi = [10, 90, 300, 335]

ppd.logger.trace(f'Importing image {testdata_filepath}')
success, img = ppd.import_image(testdata_filepath)

if not success:
    ppd.logger.error('Image import failed')

threshold = ppd.best_threshold(img, roi=testdata_roi)

cnt = ppd.find_mainContour(img, threshold, roi=testdata_roi)

estimated_parameters = ppd.estimate_parameters(ppd.image_centre(img), cnt, testdata_pxldensity)

# print(estimated_parameters)

# ### Plotting a comparison between the estimated and optimized parameters
# import matplotlib
# matplotlib.use('Qt5Agg')
# import matplotlib.pyplot as plt
# from .. import plot
#
# fig, (ax1, ax2) = plt.subplots(1, 2)
# ax = ax1
# image = img
# contour = cnt
# params = estimated_parameters
# roi = testdata_roi
#
# roi = ppd.format_roi(image, roi)
# roi[2] = roi[2] or image.shape[1]
# roi[3] = roi[3] or image.shape[0]
# ax.set_title(f'Drop image and contour')
# ax.imshow(image, cmap='gray')
# ax.plot([roi[0], roi[0], roi[2], roi[2], roi[0]], [roi[1], roi[3], roi[3], roi[1], roi[1]], lw=2, c='y', ls=':', label='ROI')
#
# xcontour, ycontour = contour[0], contour[1]
# ax.plot(xcontour, ycontour, c='lime', lw=2, label='Detected contour')
#
# gravity_angle, y_tip_position, x_tip_position, r0_mm, capillary_length_mm = params
#
# l = max(image.shape)
# ax.plot((x_tip_position + l * np.sin(-gravity_angle), x_tip_position - l * np.sin(-gravity_angle)), (y_tip_position - l * np.cos(-gravity_angle), y_tip_position + l * np.cos(-gravity_angle)),
#         color='b', lw=2, ls='--', label=f'Direction of gravity')
#
# ax.scatter(x_tip_position, y_tip_position, s=50, fc='k', ec='lime', linewidths=2, label=f'Tip position', zorder=4)
#
# ax.legend()
# ax.set_xlabel('x [px]')
# ax.set_xlim(0, image.shape[1])
# ax.set_ylabel('y [px]')
# ax.set_ylim(image.shape[0], 0)
#
#
# plt.show()
# plt.savefig('comparison.png', dpi=300)