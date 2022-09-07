# PompeiiVision
Summer project for computer vision applying on Pompeii city

Visualizations are from
```
├──visualization
│   └── svd_vis_bokeh.py
```

The visualization is based on the feature data generated from
```
├──utils
│   └── svd_mat_generator.py
```

So far we generate 4 types of features:
- Auto features: auto generated image geatures, including [shape index feature](https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_shape_index.html) and [key point feature (CENSURE feature)](https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_censure.html) 
- Manual features: read from csv file (/csv/manual.csv)
- 20x20 raw pixels
- 50x50 raw pixels

The structure of the data:

```
├──mat
│   └── 20220303
│       └── auto_features.mat
│       └── manual_features.mat
│       └── raw_20.mat
│       └── raw_50.mat
```

The mat files are uploaded separately due to their size. The mat folder can be found at https://drive.google.com/file/d/1gsF8Mf5YPPLyOcHdo0Ytlq_iusXkJh5j/view?usp=sharing

svd_vis_bokeh.py reads the data mat file to do the SVD calculation. The result is visualized by Bokeh package.


