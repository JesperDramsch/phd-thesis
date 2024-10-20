.. title: Conclusion
.. slug: conclusion
.. date: 2021-01-15 14:03:42 UTC
.. tags: 
.. category: 
.. link: 
.. description: 
.. type: text
.. _sec:conclusion:

This thesis contributes machine learning applications in geoscience with a focus on
field data applications in 4D seismic, bsem, and asi. Additionally, the
introduction contains a published review of the history of machine learning in
geoscience with insights into the recent interest around the topic.

The book chapter in `11.2 <#sec:mlingeo>`__ discusses the historic
development of machine learning in geoscience. It highlights key papers and
developments through the decades, relating the developments to larger
developments in the field of ai and machine learning. In the book key algorithms are
detailed including svm, rf, gp and the development from kriging, as well
as, key neural network developments and dl architectures that enable modern
applications throughout many scientific disciplines including geoscience
as a whole.

The exploration of bsem data in `12 <#sec:gaussian>`__ introduced a
novel unsupervised method to extract chalk grain boundaries from image
data and shows the improvement of subsequent morphological filtering
(Jesper Sören Dramsch, Amour, and Lüthje 2018). These methods reduce
labour-intensive manual tasks, introducing varying degrees of automation
in geoscience workflows. Following the extraction of the boundaries in
the bsem images, computational granulometry can be performed. This
includes statistics about grain size and circularity of the grains and
the orientation of grains. Commonly this data had to be obtained by
manual measurement of every grain. The unsupervised nature of this
application means that no training data is necessary; in turn, it can be
used to obtain high-quality training data for subsequent supervised
machine learning tasks.

The research in `13 <#sec:transfer>`__ showed that transfer learning
could alleviate the necessity for large amounts of labelled data, by
re-using a neural network trained on natural images. This study showed that neural networks can
be transferred to seismic data and outperform smaller networks trained
from scratch. The smaller network size was necessary to avoid
overfitting. The source code for this research was made available and
has been of use to multiple researchers (Jesper Sören Dramsch 2018h).
This has broad applications in industry and research settings performing
asi. The limited availability of labelled data and wide availability of
pre-trained network architectures makes this a viable option to obtain
improved results and more robust models. Moreover, this insight is
applicable to pre-training geoscientific neural networks.

Jesper Sören Dramsch, Lüthje, and Christensen (2019) shows that
explicitly using phase information as input in a complex-valued neural
network can stabilize the reconstruction of compressed seismic data. The
smaller complex-valued network in `14 <#sec:complex>`__ outperforms
larger real-valued networks; however, a very large real-valued network
that does not compress the seismic data can implicitly learn partial
phase information. The paper touches on deficits of current metrics
applied to geoscience and exposes a periodic dimming effect of
frequencies from neural networks that should be further investigated,
particularly in the context of aliasing. This paper led to the creation
of the open-source software package ``keras complex`` to enable
complex-valued deep learning in ``tf`` (Manual in
`22.5 <#section:keras-complex>`__). Considering the modularity of neural
networks, this insight can be transferred to other deep learning tasks
on physical data like seismic data. Additionally, this research could
lead to further investigation of including known physical information in
neural networks not limited to explicitly using the phase information as
input.

`15 <#sec:inversion>`__ introduces a novel method to perform
pressure-saturation inversion on amplitude difference maps (Jesper Sören
Dramsch, Corte, et al. 2019a). This work incorporates basic physical
relationships directly as features into the neural network architecture,
which was shown to stabilize the training result. Moreover, this work
shows the possibility of training dnns on simulation data and
subsequently transferring the network to field data. This particularly
was enabled by applying Gaussian noise within the network. The dnn
results were compared to results from the Bayesian inversion showing a
promising application of dnns in 4D qi (Jesper Sören Dramsch, Corte, et
al. 2019a). While this work has attracted interest in a sponsors meeting
and the workshop presentations (Jesper Sören Dramsch, Corte, et al.
2019a, 2019d), further investigation into model explainability and lower
complexity baseline models is necessary (Côrte et al. 2020; G. Corte et
al. 2020).

In `16 <#sec:timeshift>`__ a novel method for time-shift extraction is
presented. This method combines recent advancements in diffeomorphic
mapping, dl and unsupervised learning to introduce a 3D time shift
extraction method including uncertainty values, where 1D extraction is
the standard (Jesper Sören Dramsch, Christensen, et al. 2019). The
method is shown to work on 3D seismic post-stack data with strongly
differing acquisition parameters, without supplying any time shift
information. After applying the method, the 3D seismic volumes are well
aligned, with the diffeomorphic constraint performing well on seismic
data. This work tests the trained network on two other 3D seismic volume
pairs to test the generalization of the convolutional neural network after training. The two test
sets show that the trained model on a single 3D seismic volume pair
transfers well to the same field with different acquisition parameters
and even a different field with a vastly different geological setting.

Overall, this thesis shows dl applications in seismic geophysics and
resulted in multiple workshop, conference, journal papers, and a book
chapter, including reproducible Python code for all publications. The
publications, developed through interdepartmental and international
collaboration, have been disseminated at international workshops and
conferences. Two novel methods for 4D seismic analysis were introduced
and compared to conventional methods. Moreover, transfer learning as a
viable application in asi was shown and has found wide application. The
Python code in this thesis has been open-sourced for all published
papers for reproducibility including the open-source package "keras
complex".