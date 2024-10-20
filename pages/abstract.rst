.. title: Abstract
.. slug: abstract
.. date: 2021-01-15 12:58:08 UTC
.. tags: 
.. category: 
.. link: 
.. description: 
.. type: text

| Machine Learning provides a tool for the modelling and analysis of
  geoscientific data. I have placed recent developments in deep learning
  into the greater context of machine learning by reviewing the
  approaches and challenges of the use of machine learning in
  geoscience. The thesis consists of six peer-reviewed publications and
  one submitted journal paper. Furthermore, five peer-reviewed
  publications are placed in the appendix.
| The aim of this thesis is to apply recent developments in computer
  vision systems, neural networks, and machine learning to geoscientific
  data, particularly 4D seismic analysis. Neural networks are a type of
  machine learning that has made significant contributions to modern
  artificial intelligence and automation. The applicability of neural
  networks for their capability of being a universal function
  approximator was recognized within geophysics from an early stage.
  Following the recent interest in deep learning, neural networks have
  experienced a renaissance in geoscience applications, particularly in
  automatic seismic interpretation, inversion processes and sequence
  modelling.
| This is followed by an exploration of unsupervised machine learning to
  segment chalk sediments in back-scatter scanning electron microscopy
  data. The next chapter shows that using neural networks pre-trained on
  natural images can reduce the data necessary for transfer learning to
  geoscience problems. This is followed by a chapter showing that
  complex-valued convolutions can stabilize training and data
  compression on non-stationary physical data. Subsequently,
  pressure-saturation data is extracted from 4D seismic amplitude
  difference maps using a novel deep dense sample-based encoder-decoder
  network. The network contains a low-assumption physical basis
  (Amplitude Versus Offset) as explicit features and learns the residual
  for the regression of the "inversion" data. This work shows that
  transfer from simulation data to field data is possible.
| Finally, an unsupervised method is devised to extract 3D time-shifts
  from two 4D seismic cubes. The network extracts these 3D time-shifts
  including uncertainty measures. Commonly, time-shifts are extracted in
  1D, due to processing speed, computational cost and poor performance
  of 3D methods. Within the training loop, the stationary velocity field
  is numerically integrated to obtain 3D time shifts that are
  constrained by the topology in a geologically consistent manner. The
  unsupervised implementation of the network structure ensures that
  biases from other time-shift extraction methods are not implicitly
  included in the network. This application utilizes unsupervised
  learning by devising a way of behaviour for the network to follow
  instead of supplying ground truth labels. Moreover, this results in a
  way to increase trust in the system, by limiting the extraction
  process to the deep learning system and performing well-defined
  operations within the network to automate the unsupervised training.
