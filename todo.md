### TODO

In addition to the todo in the code

### Asap
* **[DOC]** give more details as to the pendent drop method and the methods used here
* **[LIB]** add more user_friendly methods to Parameters (`set_capillary_length`, `set_tip_radius`, etc...)
* **[GUI]** actualize autothreshold not when ROI moved but when ROI let go (sigRegionChangeFinished)

### Bonus
* **[GUI]** green isoline for main contour, yellow for other lines
*  calculate and output information that helps assessing the reliability of the measurement

* (Bond or Worthington ?) number (ratio of weight to maximum capillary retaining force), indicates how far we are from break-off (thus how much the shape is influenced by gravity)

* error bars from Hessian matrix around optimum parameter set

* more precise/informative display of parameters

* When changing pixel size, change r0 and lc in mm (update pxpermm in parameters, then update the display of mm)
