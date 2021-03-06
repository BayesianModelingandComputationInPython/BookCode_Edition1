(foreword)=

# Foreword

```{epigraph}
Bayesian modeling provides an elegant approach to many data science and
decision-making problems. However, it can be hard to make it work well
in practice. In particular, although there are many software packages
that make it easy to specify complex hierarchical models such as Stan,
PyMC3, TensorFlow Probability (TFP), and Pyro, users still need
additional tools to diagnose whether the results of their computations
are correct or not. They may also need advice on what to do when things
do go wrong.

This book focuses on the ArviZ library, which enables users to perform
exploratory analysis of Bayesian models, for example, diagnostics of
posterior samples generated by any inference method. This can be used to
diagnose a variety of failure modes in Bayesian inference. The book also
discusses various modeling strategies (such as centering) that can be
employed to eliminate many of the most common problems. Most of the
examples in the book use PyMC3, although some also use TFP; a brief
comparison of other probabilistic programming languages is also
included.

The authors are all experts in the area of Bayesian software and are
major contributors to the PyMC3, ArviZ, and TFP libraries. They also
have significant experience applying Bayesian data analysis in practice,
and this is reflected in the practical approach adopted in this book.
Overall, I think this is a valuable addition to the literature, which
should hopefully further the adoption of Bayesian methods.


-- Kevin P. Murphy
```
