.. title: Popular Science Summary
.. slug: popular-science-summary
.. date: 2021-01-15 10:29:45 UTC
.. tags:
.. category:
.. link:
.. description:
.. type: text

Machine learning is a new paradigm in instructing computers to perform a
task. Classically, computers were told rules to follow, when given some
data, whereas, in machine learning computers are provided with data and
answers and left with a method to figure out the rules. Recently these
methods have become increasingly powerful in identifying images, which
enabled image search and self-driving cars.

Geophysics is the discipline of imaging the Earth to gain understanding,
find natural resources and analyze geohazards. In 4D seismic, we image
the same geological area repeatedly throughout time. Changes in the
imaging equipment, the underlying geology, and variations in the way we
image the area between surveys make the problem highly complicated. The
data obtained from geophysical imaging can be fed to the same machine
learning algorithms used for images to build systems that can automate
tedious work tasks or provide new insights.

In this study, I combine powerful machine learning algorithms such as
biologically inspired neural networks with geophysical insights. At the
core of geophysics lie signal processing knowledge, which we can
investigate in neural networks itself, shedding light on the internal
properties of these networks. Moreover, incorporating physical knowledge
in neural networks directly, bears promise to gain accurate and reliable
systems that increase our understanding in Earth’s processes.

This thesis investigates the fundamental properties of neural networks
in geophysical applications. These include re-using trained neural
networks that are excellent at identifying images and applying them to
identify rock layers and geological events in geophysical images. This
thesis does a deep dive to evaluate whether the theory of including
specific information from seismic data, which is known to be very
beneficial in classical approaches in neural networks improves the
performance. We show that smaller networks that incorporate this
complex-valued information perform better than their real-valued
equivalent, decreasing computational cost.

In addition to this fundamental work, this thesis contains two
applications of machine learning to real-world problems. The first,
being that the geophysical data over hydrocarbon fields contains a
plethora of information from different effects. In this application we
develop a network that incorporates basic physical relationships of the
geophysical input data to separate the effects of changes in pressure
and saturation of water and gas in a thin reservoir in the UK North Sea.
The second application introduces a novel algorithm that evaluates a
problem that is usually approached in a one-dimensional view and extends
it to a three-dimensional algorithm. This method corrects for the slight
changes of the imaged subsurface between surveys. In addition to
extending the problem to three dimensions, the method provides
uncertainty values for the geophysicist to evaluate. Moreover, this
algorithm works unsupervised, which means that we do not have to provide
the machine learning system with information on how to align the images.
Instead, we apply a mathematical constraint that ensures that the
algorithm does not cross geological layers, a simple yet powerful
limitation to guides the algorithm to develop physically and
geologically sensible matching patterns.
