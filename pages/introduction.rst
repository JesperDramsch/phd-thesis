.. title: Introduction
.. slug: introduction
.. date: 2021-01-15 10:55:28 UTC
.. tags: 
.. category: 
.. link: 
.. description: 
.. type: text

This thesis explores machine learning in geoscience with a special focus
on deep learning in 4D seismics. Recently, machine learning and neural
networks in particular have made essential impacts in many scientific
disciplines, with geoscience exploring these new approaches as well.
This study contributes to this body of emerging work in deep neural
networks and computer vision systems for the modelling and analysis of
geoscientific data. The main contribution being a physics-based neural
architecture for pressure-saturation inversion and a novel algorithm for
3D timeshift extraction in 4D seismic.

The growing interest in machine learning sometimes overlooks the fact
that the underlying idea of machine learning was introduced in 1950.
`11.2 <#sec:mlingeo>`__ reviews the history of machine learning with a special focus
on geoscience. Geoscience and in particular geophysics has followed the
innovation in artificial intelligence and especially neural networks
closely. Early applications of neural networks include seismic
processing and seismic inversion. Moreover, gps were early introduced in
geostatistics as kriging, which have gained interest in a wider machine learning
context as gp. Recently, dl becoming popular and particularly
breakthroughs in computer vision have sparked interest in applying
machine learning computer vision to asi in the hopes for increased
accuracy, reproducibility and automation.

In recent years, 4D seismic itself has made an impact in geophysical
reservoir analysis and other geophysical areas. The method enables
imaging of changes in the subsurface. This is essential in hydrocarbon
production, enabling extended production, reducing the direct
environmental footprint and ensuring resource safety. Moreover, it
enables CO\ :sub:`2` sequestration monitoring for reservoir and seal
integrity and has applications in nuclear test treaty compliance, waste
storage, and deep geothermal monitoring. 4D seismic matching has exposed
deficits in 3D seismic processing, therefore furthered our understanding
of amplitude-preserving and surface-consistent processing steps.
Additionally, furthering our understanding of in-situ validation of
geomechanical concepts and update of heterogeneous subsurface models.

The structure of this study is composed of topical groupings of five
peer-reviewed and two submitted publications into chapters. Each chapter
will provide an individual introduction to the topic and outline
relevant theoretical and methodological aspects, where the publication
falls short. This is particularly relevant for the shorter workshop and
conference papers.

`11 <#sec:theory>`__ provides a theoretical introduction into 4D seismic
principles, followed by a thorough overview of the development of
machine learning with a special focus on geoscience. This chapter
focuses particularly on the development of machine learning applications in geoscience
through history. The main contribution in this chapter is a
peer-reviewed book chapter published in Advances in Geophysics (Jesper
Sören Dramsch 2020c).

`12 <#sec:gaussian>`__ contains a workshop paper (Jesper Sören Dramsch,
Amour, and Lüthje 2018), which explores the application of unsupervised
learning to the segmentation of chalk grains in bsem images. The chapter
expands on the method and provides a theoretical treatment of the
methods applied in the short paper. The method is also compared to
classical image processing techniques. Then an overview of additional
computational granulometry based on the segmentation maps is presented
to apply the work and close out the chapter.

`13 <#sec:transfer>`__ discusses a conference paper contribution to asi
using dl (Jesper Sören Dramsch and Lüthje 2018a). The paper uses
transfer learning of neural networks pre-trained on natural image data sets to
fine-tune the network to perform asi on seismic data. The chapter
expands on the data and training of the neural network. The chapter then expands on
the applications that resulted from the paper, using the composition of
nns into more adequate architectures for a task that is called semantic
segmentation, which more closely resembles asi.

`14 <#sec:complex>`__ covers a journal paper on the application of
complex-valued convolutional neural networks to seismic data (Jesper Sören Dramsch, Lüthje, and
Christensen 2019). These networks perform a complex convolution in the
nn layers. The paper tests the hypothesis that providing phase
information explicitly can improve the capacity of the convolutional neural network, which is
tested on an ae architecture, which lossily compresses the data at
different rates and measures the reconstruction error. The phase
information is derived directly from the seismic data via a Hilbert
transform, hence a dnn could, in theory, extract this information
automatically. For this chapter, networks at varying compression were
trained for both real-valued and complex-valued networks to perform an
adequate comparison.

`15 <#sec:inversion>`__ consists of two workshop papers which introduce
a dnn architecture for 4D quantitative pressure-saturation inversion
(Jesper Sören Dramsch, Corte, et al. 2019d, 2019a). The dnn regression
model implements a layer that computes basic physical knowledge within
the network architecture to stabilize the network. The physical
knowledge encoded in the layer is the avo gradient between the input
seismic data. This data is passed into a vae architecture. In this work,
we show that this network can be trained on simulation data and
transferred to field data by applying Gaussian noise to the noise-free
simulation input data to condition the network to accept noisy inputs
from field data.

`16 <#sec:timeshift>`__ is comprised of a re-submitted journal paper and
introduces a robust method for 3D time shift extraction in 4D data
(Jesper Sören Dramsch, Christensen, et al. 2019). Time shifts in 4D data
are commonly extracted in 1D due to computational cost and often poor
performance of 3D methods. This method uses a self-supervised deep
learning system to extract the timeshift mapping of two seismic volumes
without supplying a-priori timeshift data. Moreover, the method limits
the neural network to the extraction of the stationary timeshift but
leaves the matching to a non-learning 3D interpolation to increase the
transparency of the method. Additionally, the method supplies
uncertainty values for the warp velocity. Constraining the possible 3D
time shifts is vital to ensure sensible results for the time shifts, as
well as, the aligned monitor seismic. This is ensured by implementing a
geologically intuitive constraint on the 3D timeshifts, which prohibits
crossing or looping of reflectors after mapping the seismic volumes.
This learning-based method can be trained in advance, providing fast 3D
results on previously unseen data, which is essential in 4D seismic
analysis.

Finally, `17 <#sec:conclusion>`__ is the conclusion of this thesis
recapitulating the contributions and findings of the papers and
scientific work. The contributions span multiple geoscientific
disciplines with a focus in geophysics and particularly 4D seismic
unified by machine learning.
