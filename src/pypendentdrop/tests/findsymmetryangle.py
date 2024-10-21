from ... import pypendentdrop as ppd

import numpy as np

ppd.set_verbose(3)

testdata_filepath = 'src/pypendentdrop/tests/testdata/water_2.tif'
testdata_pxldensity = 57.0
testdata_rhog = 9.81
testdata_roi = [10, 90, 300, 335]

success, img = ppd.import_image(testdata_filepath)

if not success:
    ppd.error('Image import failed')

threshold = ppd.best_threshold(img, roi=testdata_roi)

cnt = ppd.find_mainContour(img, threshold, roi=testdata_roi)

estimated_parameters = ppd.estimate_parameters(ppd.image_centre(img), cnt, testdata_pxldensity)

print(estimated_parameters)

### Plotting a comparison between the estimated and optimized parameters
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from .. import plot

fig, (ax1, ax2) = plt.subplots(1, 2)
plot.plot_image_contour(ax1, img, cnt, testdata_pxldensity, estimated_parameters, 'estimated', roi=testdata_roi)
plt.show()
# plt.savefig('comparison.png', dpi=300)