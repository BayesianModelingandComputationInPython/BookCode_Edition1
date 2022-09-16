(preface)=

# Preface

The name Bayesian statistics is attributed to Thomas Bayes (1702--1761),
a Presbyterian minister, and amateur mathematician, who for the first
time derived what we now know as Bayes' theorem, which was published
(posthumously) in 1763. However, one of the first people to really
develop Bayesian methods was Pierre-Simon Laplace (1749--1827), so
perhaps it would be a bit more correct to talk about Laplacian
Statistics. Nevertheless, we will honor Stigler's law of eponymy and
also stick to tradition and keep talking about Bayesian approaches for
the rest of this book. From the pioneering days of Bayes and Laplace
(and many others) to the present day, a lot has happened - new ideas
were developed, many of which were motivated and or being enabled by
computers. The intent of this book is to provide a modern perspective on
the subject, from the fundamentals in order to build a solid foundation
into the application of a modern Bayesian workflow and tooling.

We write this book to help beginner Bayesian practitioners to become
intermediate modelers. We do not claim this will automatically happen
after you finish reading this book, but we hope the book can guide you
in a fruitful direction specially if you read it thoroughly, do the
exercises, apply the ideas in the book to your own problems and continue
to learn from others.

Specifically stated this book targets the Bayesian practitioners who are
interested in applying Bayesian models to solve data analysis problems.
Often times a distinction is made between academia and industry. This
book makes no such distinction, as it will be equally useful for a
student in a university as it is for a machine learning engineer at a
company.

It is our intent that upon completion of this book you will not only be
familiar with **Bayesian Inference** but also feel comfortable
performing **Exploratory Analysis of Bayesian Models**, including model
comparison, diagnostics, evaluation and communication of the results. It
is also our intent to teach all this from a modern and computational
perspective. For us, Bayesian statistics is better understood and
applied if we take a **computational** approach, this means, for
example, that we care more about empirically checking how our
assumptions are violated than trying to prove assumptions to be right.
This also means we use many visualizations (if we do not do more is to
avoid having a 1000 pages book). Other implications of the modeling
approach will become clear as we progress through the pages.

Finally, as stated in the book's title, we use the Python programming
language in this book. More specifically, we will mainly focus on PyMC3
{cite:p}`Salvatier2016` and TensorFlow Probability (TFP)
{cite:p}`dillon2017tensorflow`, as the main probabilistic programming languages
(PPLs) for model building and inference, and use ArviZ as the main
library for exploratory analysis of Bayesian models {cite:p}`Kumar2019`. We do
not intend to give an exhaustive survey and comparison of all Python
PPLs in this book, as there are many choices, and they rapidly evolve.
We instead focus on the practical aspects of Bayesian analysis.
Programming languages and libraries are merely bridges to get where we
want to go.

Even though our programming language of choice for this book is Python,
with few selected libraries, the statistical and modeling concepts we
cover are language and library agnostic and available in many computer
programming languages such as R, Julia, and Scala among others. A
motivated reader with knowledge of these languages but not Python can
still benefit from reading the book, especially if they find the
suitable packages that support, or code, the equivalent functionality in
their language of choice to gain hands on practice. Furthermore, the
authors encourage others to translate the code examples in this work to
other languages or frameworks. Please get in touch if you like to do so.

(prior-knowledge)=

## Prior knowledge

As we write this book to help beginners to become intermediate
practitioners, we assume prior exposure, but not mastery, of the basic
ideas from Bayesian statistics such as priors, likelihoods and
posteriors as well as some basic statistical concepts like random
variables, probability distributions, expectations. For those of you
that are a little bit rusty, we provide a whole section inside Chapter [11](app),
with a refresher about basic statistical concepts.
A couple of good books explaining these concepts
in more depth are Understanding Advanced Statistical Methods
{cite:p}`WestfallUnderstandingAdvancedStatistical2013` and Introduction to
Probability {cite:p}`blitzstein_2019`. The latter is a little bit more
theoretical, but both keep application in mind.

If you have a good understanding of statistics, either by practice or
formal training, but you have never been exposed to Bayesian
statistics, you may still use this book as an introduction to the
subject, the pace at the start (mostly the first two chapters) will be a
bit rapid, and may require a couple read-throughs.

We expect you to be familiar with some mathematical concepts like
integrals, derivatives, and properties of logarithms. The level of
writing will be the one generally taught at a technical high school or
maybe the first year of college in science, technology, engineering, and
mathematics careers. For those who need a refresher of such mathematical
concepts we recommend the series of videos from 3Blue1Brown [^1]. We
will not ask you to solve many mathematical exercises instead, we will
primarily ask you to use code and an interactive computing environment
to understand and solve problems. Mathematical formulas throughout the
text are used only when they help to provide a better understanding of
Bayesian statistical modeling.

This book assumes that the reader comes with some knowledge of
scientific computer programming. Using the Python language we will also
use a number of specialized packages, in particular Probabilistic
Programming Languages. It will help, but is not necessary, to have fit
at least one model in a Probabilistic Programming language prior to
reading this book. For a reference on how to setup the
computation environment needed for this book, read [environment](https://github.com/BayesianModelingandComputationInPython/BookCode_Edition1#environment-installation) installation.

(how-to-read-this-book)=

## How to read this book

We will use toy models to understand important concepts without the data
obscuring the main concepts and then use real datasets to approximate
real practical problems such as sampling issues, reparametrization,
prior/posterior calibration, etc. We encourage you to run these models
in an interactive code environment while reading the book.

We strongly encourage you to read and use the online documentation for
the various libraries. While we do our best to keep this book
self-contained, there is an extensive amount of documentation on these
tools online and referring it will aid in both learning this book, as
well as utilizing the tools on your own.

[Chapter 1](chap1) offers a refresher or a quick introduction to the basic and
central notions in Bayesian inference. The concepts from this chapter
are revisited and applied in the rest of the book.

[Chapter 2](chap1bis) offers an introduction to Exploratory Analysis of Bayesian
models. Namely introduces many of the concepts that are part of the
Bayesian workflow but are not inference itself. We apply and revisit the
concepts from this chapter in the rest of the book.

[Chapter 3](chap2) is the first chapter dedicated to a specific model
architecture. It offers an introduction to Linear Regression models and
establishes the basic groundwork for the next 5 chapters. Chapter 3 also
fully introduces the primary probabilistic programming languages used in
the book, PyMC3 and TensorFlow Probability.

[Chapter 4](chap3) extends Linear Regression models and discusses more advanced
topics like robust regression, hierarchical models and model
reparametrization. This chapter uses PyMC3 and TensorFlow Probability.

[Chapter 5](chap3_5) introduces basis functions and in particular splines as an
extension to linear models that allows us to build more flexible models.
This chapter uses PyMC3.

[Chapter 6](chap4) focuses on time series models, from modeling time series as a
regression to more complex model like ARIMA and linear Gaussian State
Space model. This chapter uses TensorFlow Probability.

[Chapter 7](chap6) offers an introduction to Bayesian additive regression trees a
non-parametric model. We discuss the interpretability of this model and
variable importance. This Chapter use PyMC3.

[Chapter 8](chap8) brings the attention to the Approximate Bayesian Computation
(ABC) framework, which is useful for problems where we do not have an
explicit formulation for the likelihood. This chapter uses PyMC3.

[Chapter 9](chap9)  gives an overview of end-to-end Bayesian workflows. It
showcases both an observational study in a business setting and an
experimental study in a research setting. This chapter uses PyMC3.

[Chapter 10](chap10)  provides a deep dive on Probabilistic Programming Languages.
Various different Probabilistic Programming languages are shown in this
chapter.

[Chapter 11](app) serves as a support when reading other chapters, as the
topics inside it are loosely related to each other, and you may not want
to read linearly.


(text-highlights)=

### Text Highlights

Text in this book will be emphasized with **bold** or *italics*. **Bold
text** will highlight new concepts or emphasis of a concept. *Italic
text* will indicate a colloquial or non-rigorous expression. When a
specific code is mentioned they are also highlighted: `pm.sample`.

(code)=

### Code

Blocks of code in the book are marked by a shaded box with the lines
numbers on the left. And are referenced using the chapter number
followed by the number of the Code Block. For example:

```python
for i in range(3):
    print(i**2)
```

```none
0
1
4
```

Every time you see a code block look for a result. Often times it is a
figure, a number, code output, or a table. Conversely most figures in
the book have an associated code block, sometimes we omit code blocks in
the book to save space, but you can still access them at the [GitHub
repository](https://github.com/BayesianModelingandComputationInPython).
The repository also includes additional material for some exercises. The
notebooks in that repository may also include additional figures, code,
or outputs not seen in the book, but that were used to develop the
models seen in the book. Also included in GitHub are instructions for
how to create a standard computation environment on whatever equipment
you have.

(boxes)=

### Boxes

We use boxes to provide a quick reference for statistical, mathematical,
or (Python) Programming concepts that are important for you to know. We
also provide references for you to continue learning about the topic.

:::{admonition} Central Limit Theorem

In probability theory, the central limit theorem
establishes that, in some situations, when independent random variables
are added, their properly normalized sum tends toward a normal
distribution even if the original variables themselves are not normally
distributed.

Let $X_1, X_2, X_3, ...$ be i.i.d. with mean $\mu$ and standard
deviation $\sigma$. As $n \rightarrow \infty$, we got:

```{math} 
\sqrt{n} \left(\frac{\bar{X}-\mu}{\sigma} \right) \xrightarrow{\text{d}} \mathcal{N}(0, 1)
```

The book Introduction to Probability {cite:p}`blitzstein_2019` is a good
resource for learning many theoretical aspects of probability that are
useful in practice.
:::

(code-imports)=

### Code Imports

In this book we use the following conventions when importing Python
packages.

```python
# Basic
import numpy as np
from scipy import stats
import pandas as pd
from patsy import bs, dmatrix
import matplotlib.pyplot as plt

# Exploratory Analysis of Bayesian Models
import arviz as az

# Probabilistic programming languages
import bambi as bmb
import pymc3 as pm
import tensorflow_probability as tfp

tfd = tfp.distributions

# Computational Backend
import theano
import theano.tensor as tt
import tensorflow as tf
```

We also use the ArviZ style `az.style.use("arviz-grayscale")`

(how-to-interact-with-this-book)=

### How to interact with this book

As our audience is not a *Bayesian reader*, but a Bayesian practitioner.
We will be providing the materials to practice Bayesian inference and
exploratory analysis of Bayesian models. As leveraging computation and
code is a core skill required for modern Bayesian practitioners, we will
provide you with examples that can be played around with to build
intuition over many tries. Our expectation is that the code in this book
is read, executed, modified by the reader, and executed again many
times. We can only show so many examples in this book, but you can make
an infinite amount of examples for yourself using your computer. This
way you learn not only the statistical concepts, but how to use your
computer to generate value from those concepts.

Computers will also remove you from the limitations of printed text, for
example lack of colors, lack of animation, and side-by-side comparisons.
Modern Bayesian practitioners leverage the flexibility afforded by
monitors and quick computational "double checks" and we have
specifically created our examples to allow for the same level of
interactivity. We have included exercises to test your learning and
extra practice at the end of each chapter as well. Exercises are labeled
Easy (E), Medium (M), and Hard (H). Solutions are available on request.

(acknowledgments)=

## Acknowledgments

We are grateful to our friends and colleagues that have been kind enough
to provide their time and energy to read early drafts and propose and
provide useful feedback that helps us to improve the book and also helps
us to fix many bugs in the book. Thank you:

Oriol Abril-Pla, Alex Andorra, Paul Anzel, Dan Becker, Tomás Capretto,
Allen Downey, Christopher Fonnesbeck, Meenal Jhajharia, Will Kurt, Asael
Matamoros, Kevin Murphy, and Aki Vehtari.

[^1]: <https://www.youtube.com/channel/UCYO_jab_esuFRV4b17AJtAw>, we
    recommend these videos even if you do not need a refresher.
