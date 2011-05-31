.. |MATLAB| replace:: ``MATLAB``
.. |T2| replace:: :math:`T^2`
.. |t| replace:: :math:`t`

PAGE NUMBERS ARE ALL MESSED UP
=================================

This document describes how to implement a monitoring system for real-time monitoring of a batch process, using latent variable methods.

We assume that a latent variable model for the batch system has already been built using historical batch data.  Now we wish to apply that model to a new batch, in real-time.

This document focuses on using |MATLAB| code to describe the implementation, however, the description is general enough so that any programming language could be used.

Terminology
============

Variables
	These are the measurements (tags) from the batch system, such as temperature, flow rate, electrical current, motor speed, and so on.  There are :math:`K` variables, and we reference them as :math:`1, 2, \ldots, k, \ldots, K`, where :math:`k` is a generic measurement.
	
Time steps
	Each batch evolves over time.  The time steps are equal-duration periods over which the batch evolves, for example: each time step could be 30 seconds apart.  There are a total of :math:`J` time steps, and we reference them as :math:`1, 2, \ldots, j, \ldots, J`, where :math:`j` is a generic time step.
	
Missing data
	For real-time monitoring purposes we consider the future measurements from a new batch to be missing.  Once the batch is terminated, then that data is available.  But while a new batch is evolving we have a certain amount of missing data, which gradually is reduced until the batch is complete.
	
	We denote missing data with an indicator: ``NaN``.  A missing value should never be denoted by zero, 0, or any other numeric indicator, such as -99.
	
	We might also have additional missing data when one or more variables, such as a temperature sensor, goes off-line, or is broken, or is providing bad values. In these cases we also set the bad value as a missing value, ``NaN``.
	
Latent variable model
	A latent variable model is one type of empirical model developed from historical data.  More details can be provided on request, however for the purposed of this implementation document, we just need to differentiate between PCA (principal component analysis) and PLS (project to latent structure) models.  Both models have similar outputs, but the PLS model provides an additional output: a prediction of the batch outcome (quality space).  
	
	Both PCA and PLS models have a dimensionality parameter, called :math:`A`, which is the number of latent variables in the model.
	

Overview of workflow for every time step
=========================================

We will perform the following calculations at every time step, :math:`j`, while a new batch is evolving.

#.	Collect new measurements from the :math:`K` process variables and store them.
#.	Preprocess these measurements.
#.	Apply these measurements to the existing latent variable model.
#.	Collecting and storing the results of the model output.
#.	Plot the results; use these plots to detect any problems with the batch.
#.	Show contribution plot to diagnose any problems that might have been flagged in the previous step

Each step is discussed in more detail in the next sections.

Data collection and storage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We collect measurements from :math:`K` variables at :math:`J` time steps.  So at batch termination we have :math:`JK` measurements for the new batch.  If we have :math:`K=10` process measurements, taken over :math:`J=350` time steps (one minute apart), we will have assembled 3500 measurements in total by the end of a batch.

The illustration below shows how we store these measurements at the first time step (:math:`j=1`), at an intermediate time step (:math:`j=j`), and at the end of the batch (:math:`j=J`).  The key point is that only the most recent :math:`K` entries in the vector are new, while earlier entries are fixed.

.. .. figure:: images/batch-data-evolution.png
.. 	:alt:	images/batch-data-evolution.svg
.. 	:scale: 60%
.. 	:width: 750px
.. 	:align: center
	 
The shaded entries represent actual measurements, while the open space in the rest of the vector represents missing data.  The vector of measurements will grow larger and larger over time and the amount of missing data decreases over time.  The above vector is called the *unfolded data vector*, since the values have been unfolded into a single long row.  We will call this vector :math:`\mathbf{x}_\text{unfolded}`.

In some programming languages it might be more efficient to preallocate storage for all :math:`JK` entries ahead of time, however it many cases it doesn't practically matter if the vector grows over time.

Data preprocessing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The data preprocessing steps are actually part of the latent variable model, however for implementation purposes we consider it a separate step.

The unfolded vector, :math:`\mathbf{x}_\text{unfolded}`, is preprocessed (the values are modified), but the length of the vector before and after preprocessing is identical.  In particular, any missing values before preprocessing must be left as missing after preprocessing, :math:`\mathbf{x}_\text{pp}`. 

There are actually two steps in the preprocessing stage, but we can write them in one go:

.. math::
	\mathbf{x}_\text{pp} = \left(\mathbf{x}_\text{unfolded} - \mathbf{x}_\text{mean} \right) \cdot \mathbf{x}_\text{scale}

where :math:`\mathbf{x}_\text{mean}` is the mean-centering vector and :math:`\mathbf{x}_\text{scale}` is the scaling vector.  Both of these vectors have :math:`JK` entries.  The subtraction and multiplication (:math:`\cdot`) operations are performed on an element-by-element basis, so that afterwards the :math:`\mathbf{x}_\text{pp}` vector has the same size as :math:`\mathbf{x}_\text{unfolded}`.

For efficiency purposes one only needs to preprocess the :math:`K` most recent entries in :math:`\mathbf{x}_\text{unfolded}` to obtain the last :math:`K` entries in :math:`\mathbf{x}_\text{pp}`.  Entries from prior time steps do not change in future time steps.

Applying the data to the latent variable model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this step we use the :math:`\mathbf{x}_\text{pp}` preprocessed vector and apply it to the latent variable model. 

The key variable in a PCA (principal component analysis)  model is a matrix of loadings, called :math:`\mathbf{P}`.  The key variable in a PLS (project to latent structure) model is a matrix of weights, called :math:`\mathbf{W}`.  Both matrices are rectangular with :math:`JK` rows and :math:`A` columns.  We will only use a portion of either matrix, depending on how far the batch has evolved.  The subset of :math:`\mathbf{P}` that will be used is called :math:`\mathbf{P}_*`; similarly the subset of :math:`\mathbf{W}` will be called :math:`\mathbf{W}_*`.

So for a PCA model:

*	At :math:`j=1` we will only use the first :math:`K` rows, and the :math:`a^\text{th}` column. We say that ``P*(:,a)`` is a :math:`K \times 1` vector.
*	At :math:`j=2` we will only use the first :math:`2K` rows, and the :math:`a^\text{th}` column. So ``P*(:,a)`` is a :math:`2K \times 1` vector.
*	At :math:`j=j` we will only use the first :math:`jK` rows, and the :math:`a^\text{th}` column. So ``P*(:,a)`` is a :math:`jK \times 1` vector.
*	At :math:`j=J` we will only use all rows, :math:`JK` rows, and the :math:`a^\text{th}` column. So ``P*(:,a) = P(:,a)`` and is the full :math:`JK \times 1` vector.

.. _batch-score-calculations-SCP:

Score calculations
~~~~~~~~~~~~~~~~~~~~~~~~~

The major issue with using a batch model on-line is the problem of missing future values. There are several methods to deal with missing values; below we use the single component projection (SCP) method: this method projects each component's loadings onto the available x-data to obtain the score, and it only does so one component at a time.

At the :math:`j^\text{th}` time step we must perform the following matrix algebra steps to obtain a new vector of scores, called :math:`\mathbf{t}_{\text{new},j}`.  This vector has :math:`A` entries, each entry is obtained successively in a loop of :math:`A` iterations: :math:`1, 2, \ldots, a, \ldots A`.

In a loop that is repeated :math:`A` times, perform the following **two instructions**, using the :math:`a^\text{th}` column in :math:`\mathbf{P}_*` and/or in :math:`\mathbf{W}_*`:

PCA models
	.. math::
		t_{\text{new},j}(a) &= (\mathbf{P}'_*(:,a) \mathbf{x}_\text{pp}) / \left(\mathbf{P}'_*(:,a) \mathbf{P}_*(:,a) \right) \\
		\mathbf{x}_\text{pp} &= \mathbf{x}_\text{pp} - t_{\text{new},j}(a) \mathbf{P}'_*(:,a) \\

PLS models
	.. math::
		t_{\text{new},j}(a) &= (\mathbf{W}'_*(:,a) \mathbf{x}_\text{pp}) / \left(\mathbf{W}'_*(:,a) \mathbf{W}_*(:,a) \right) \\
		\mathbf{x}_\text{pp} &= \mathbf{x}_\text{pp} - t_{\text{new},j}(a) \mathbf{P}'_*(:,a) \\

The PCA model only uses :math:`\mathbf{P}`, while the PLS model used both the :math:`\mathbf{P}` and :math:`\mathbf{W}` matrices.  Note that the last instruction modifies the values in :math:`\mathbf{x}_\text{pp}` for the next iteration of the loop.  This instruction uses :math:`\mathbf{P}` for both PCA and PLS models.

The loadings, :math:`\mathbf{P}`,  are stored in ``model.P``, and the weights, :math:`\mathbf{W}`,  are stored in ``model.W``.

A complicating factor in the above calculations is that :math:`\mathbf{x}_\text{pp}` may contain missing entries, indicated by ``NaN``.  These entries should just be skipped over.  An easy way to achieve this is to temporarily set missing entries to zero, then perform the matrix algebra.  Setting missing values to zero at this point is permitted, because this removes their effect in this particular calculation.

.. Since this is matrix algebra, we must ensure the dimensions are consistent:
.. 
..  .. math::
.. 	\mathbf{P}_* &  \qquad jK \times A  \\
.. 	\mathbf{P}'_*&  \qquad A \times jK \\
.. 	\mathbf{P}'_* \mathbf{P}_* & \qquad A \times A \\
.. 	\left(\mathbf{P}'_* \mathbf{P}_*\right)^{-1} & \qquad A \times A \\
.. 	\mathbf{x}_\text{pp}& \qquad jK \times 1 \\
.. 	\mathbf{t}_{\text{new},j} & \qquad A \times 1

The Hotelling's |T2| value
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both PCA and PLS models derive the Hotelling's |T2| value from the scores :math:`\mathbf{t}_\text{new}`:

.. math:: 
	T^2_{\text{new},j} = \dfrac{t^2_{\text{new},j}(1)}{s_{1,j}^2} + \dfrac{t^2_{\text{new},j}(2)}{s_{2,j}^2} + \ldots + \dfrac{t^2_{\text{new},j}(A)}{s_{A,j}^2}

The above equation shows that at the :math:`j^\text{th}` time step we use the score values, :math:`\mathbf{t}_{\text{new},j}`, which is a vector of :math:`A` entries and the scaling values, :math:`\mathbf{s}_{a,j}`, also a vector of :math:`A` entries.  We add up the sum of squares for each of the :math:`A` entries in :math:`\mathbf{t}_{\text{new},j}`.

The scaling values are stored in ``model.S``.

.. _batch-instantaneous-SPE-values:

The squared prediction error (SPE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The SPE value at time step :math:`j` is calculated in several steps:

* :math:`\widehat{\mathbf{x}} = \mathbf{P}_* \mathbf{t}_\text{new}` which has dimension: :math:`(jK \times A)(A \times 1) = jK \times 1`
* :math:`\mathbf{e}_\text{new} = \mathbf{x}_\text{pp} - \widehat{\mathbf{x}}` (all three vectors are :math:`jK \times 1`)
* :math:`\mathbf{e}_{\text{new},j}` = take the last :math:`K` entries from :math:`\mathbf{e}_\text{new}`, corresponding to the :math:`K` variables at time step :math:`j`.
* Then take the sum of squares of these :math:`K` values to get the SPE value at time step :math:`j`: :math:`\text{SPE}_j = \sqrt{\mathbf{e}'_{\text{new},j} \mathbf{e}_{\text{new},j}} = \sqrt{\mathbf{e}_{\text{new},j} \times \mathbf{e}_{\text{new},j}}`, where the multiplication is performed element-by-element.

PLS only: predicted final quality attributes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The PLS batch model will, at every time step, also provide an estimate of the *final* product quality attributes (FQA).  There are :math:`M` of these FQA's, denoted as :math:`\widehat{\mathbf{y}}`.  Please note the PLS model provides a prediction of the FQA at the end of the batch, even though we might be at time step :math:`j<J`.  It is not a prediction of the FQA at time step :math:`j`.

Once we have the new scores, :math:`\mathbf{t}_\text{new}`, then we can also calculate the model predictions using this matrix algebra equation.  The matrix dimensions appear below in parenthesis.

.. NOTE: when transferred this to book, then please rewrite the equation dimensions to be consistent.

.. math::
	\widehat{\mathbf{y}}_j &= \mathbf{C}' \mathbf{t}_{\text{new},j}  \qquad (M \times 1) = (M \times A) (A \times 1) \\
	\widehat{\mathbf{y}}_j &= \widehat{\mathbf{y}}_j \div \mathbf{y}_\text{scale} + \mathbf{y}_\text{mean}

The first line gives a prediction that is scaled and centered; we have to undo the scaling and centering, as shown in the second line, to get the prediction in real-world units.   The matrix :math:`\mathbf{C}` is stored in ``model.C``.

Collecting and storing the results from the model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following outputs are available at every time step and should be stored in a data historian.

#. The :math:`A` score values: :math:`t_{j,1}, t_{j,2}, \ldots, t_{j,A}`
#. The Hotelling's |T2| value: :math:`T^2_{\text{new},j}`
#. The SPE value: :math:`\text{SPE}_j`
#. The :math:`M` values of the predicted final product quality: :math:`\widehat{y}_{j,1}, \widehat{y}_{j,2}, \ldots, \widehat{y}_{j,M}`.  

For archiving purposes, only the last entry, at time step :math:`j=J` need be stored for the prediction final quality attributes: :math:`\widehat{\mathbf{y}}_J`, the predictions at previous time steps are not of too much interest.

Plotting the model results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Several plots are useful to monitoring a new batch.  Some examples are given below.

#.	Time-based plots of the :math:`A` score values.

	.. .. figure:: images/scores.jpg
	.. 	:scale: 82%
	.. 	:width: 750px
	.. 	:align: center
		
	Since the scores are linear combinations of the :math:`\mathbf{x}` data, with weights given by the loadings (PCA) or :math:`\mathbf{W}`'s (PLS), we can assume that these scores will be normally distributed, according to the central limit theorem.  
	
	The upper and lower limits are shown with 2:math:`\sigma` (approximately 95\%) and 3:math:`\sigma` (approximately 99.95%) limits.   These limits change at every time step.  The variance is usually higher at the start of a batch, but then stabilizes as the batch proceeds.
	
	Use the ``model.limits.t_X_95`` and ``model.limits.t_X_99`` vectors in the model as the limits.
	
#.	Time-based plots of Hotelling's |T2| and SPE

	Strictly speaking, these plots also have time-varying limits, however, their limits can be quite variable.  So it is better to rather normalize the SPE values by dividing through by the limits.  These normalized values appear in the ``T2_norm`` and ``SPE_norm`` vectors.  The limit limits on these plots are at 0.8 and 1.0.  Values above 0.8 are warning limits, values above 1.0 indicate a serious error in the batch process.   

	.. .. figure:: images/T2-and-SPE.jpg
	.. 	:scale: 80%
	.. 	:width: 750px
	.. 	:align: center
		
	One may find that the limits are too wide or too narrow; this is usual for any initial deployment of a monitoring system.  The normalizing factors may be adjusted, so that the alarms occur at the desired frequency (not too many false alarms balanced against Type-II error, not detecting the alarm soon enough).
		
#.	Time-based plots of the predictions (PLS models) only
	
	For PLS models we also show the predicted values of each final quality attribute, :math:`\widehat{\mathbf{y}}_j`, at every time step.  Shown here is one such attribute.  Each point represents a prediction of the *final* attribute.  The prediction is more uncertain at the start of the batch than at the end, which is why they  stabilize towards the end of the batch. 
	
	.. .. figure:: images/predicted-Y.jpg
	.. 	:scale: 45%
	.. 	:width: 750px
	.. 	:align: center

#.	Score target plots; useful only for models with :math:`A=2`.

	Some models where there are only two components, :math:`A=2`, can use a target plot of the scores instead of a Hotelling's |T2| plot.  This plot conveys the same information, but does loose the time dimension. 
	
	The plot evolves over time, creating a snake-like pattern as it moves around.  A bold marker should be used to show the current point in time.  In the example here, the batch started off inside the target zone, but moved outside the limits towards the end.

	.. .. figure:: images/scores-target.jpg
	.. 	:scale: 50%
	.. 	:width: 750px
	.. 	:align: center
		
	The code to plot the elliptical limits is provided in the accompany software files.

	.. The Hotelling's |T2| limits are calculated from the following relationship:
	.. 
	.. .. math::
	.. 	\dfrac{(N-1)(N+1)(A)}{(N)(N-A)}\cdot F_{\alpha}(A, N-A)
	.. 
	.. where :math:`F_{\alpha}(A, N-A)` is the critical value at :math:`\alpha` from the F-distribution with :math:`A` and :math:`N-A` degrees of freedom.

Contribution values
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If the monitoring limits for any plot are exceeded, then it is an indication an unusual event has taken place.  The next logical step is to investigate which of the variables in the batch system are most related with this event.  

**Please note**: contributions are not always conclusive - they will only highlight the variables *related with* the event, not necessarily the cause of the problem.

The contribution should be plotted according to the type of limit that was exceeded; for example, plot SPE contributions only when the SPE limit is exceeded.   Contribution values are non-zero even if the value is below the alarm limit.  For this reason, it only make sense to display contribution values if the limit is exceeded. 

Scores contributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If for example the score, :math:`t_2`, exceeded its :math:`2\sigma` limit at time step :math:`j=12`, then the user should be able to click on that point which is beyond the :math:`2\sigma` limit, and generate a contribution plot for :math:`t_2`.

In the accompanying code, the following lines would be run in MATLAB::

	>> j=12;
	>> [c_scores, c_SPE, c_T2] = get_contributions(model, state, j);
	>> c_scores
	-2.9361    0.1776
	      0         0
	-4.8154   -8.9682
	-11.8279 -11.5577
	-0.4386  -11.6870
	24.8405   12.7165
	      0         0
	      0         0

The first column represents the contributions for the first component, :math:`t_1`, and the second column would be for :math:`t_2`, one value for each tag.  Those :math:`K=8` values should be plotted as a bar chart, indicating here that 3 tags related to the impeller and the product temperature were most related to the problem.

.. .. figure:: images/contribution-plot-t2.pdf
.. 	:scale: 100%
.. 	:width: 750px
.. 	:align: center

The next step is for the operator or user to look back at the raw data for this batch, specifically the 4 tags highlighted from the contribution plot.

.. The score value is given by :math:`t_a = \mathbf{x}_\text{pp} \mathbf{P}_*` for a PCA model, which is a linear combination of the raw data and the loadings.  The details are :ref:`in this section <batch-score-calculations-SCP>`.  To calculate the contributions, we simply add up the part in each score that is due to each variable.  Furthermore, since we are only interested in the contribution, we use the absolute value of the loadings, since a large positive or a large negative loading is important. 

.. contribs = zeros(idx_end, model.A);
.. for a = 1:model.A
..     if model.M > 0
..         temp = abs(model.W(1:idx_end,a));
..     else
..         temp = abs(model.P(1:idx_end,a));
..     end
..     
..     contribs(:,a) = x_pp(1:idx_end,1) .* temp / (temp'*temp);
..     x_pp(1:idx_end,1) = x_pp(1:idx_end,1) - state.scores(timestep, a) * model.P(1:idx_end, a);
.. end

|T2| and SPE contributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The contributions to |T2| and SPE are used in the same way as the individual score contributions: if either of these statistics exceeds their limit, then a vector of :math:`K` contributions for each tag are calculated in the same way.

.. The instantaneous SPE value is calculated as shown :ref:`in this section <batch-instantaneous-SPE-values>`.  The contributions are the part of the SPE value that is due to each variable.  The SPE value is the sum of squares of the errors, so at time step :math:`j`, for variable :math:`k`, the SPE contributions are:
.. 
.. .. math::
.. 	\text{sign}(e_{k,j}) \cdot e_{k,j}^2
.. 
.. where :math:`\text{sign}(e_{k,j})` returns either a :math:`+1` or :math:`-1`, depending on the sign of the input.  We retain the sign information, because that can be useful in the interpretation of the SPE contributions.

.. c_SPE = sign(error_j) .* error_j .^2;   % error_hj

Model specification
====================================

We require a batch monitoring model in order to use the on-line batch monitoring code.  This section describes the dimensions of the various matrices, vectors and scalars that make up a multivariate model.

* ``model.A``, the number of latent variables.  Indexed as :math:`1, 2, \ldots, a, \ldots, A`.
* ``model.K``, the number of tags (variables) used in the model.  Indexed as :math:`1, 2, \ldots, k, \ldots, K`.
* ``model.M``, the number of final quality attributes for a predictive PLS model.  Indexed as :math:`1, 2, \ldots, m, \ldots, M`.  Note that :math:`M=0` for PCA models.
* ``model.J``, the number of time steps within one batch.  Indexed as :math:`1, 2, \ldots, j, \ldots, J`.
* ``model.N``, the number of batches used to build the model.
* ``model.S``, the score scaling matrix; an :math:`J \times A` matrix
* ``model.P``, the loadings for a PCA and PLS model; an :math:`JK \times A` matrix.
* ``model.W``, the weights for a PLS model; an :math:`JK \times A` matrix.
* ``model.C``, the Y-space weights for a PLS model; an :math:`M \times A` matrix.
* ``model.ppx.center``, the preprocessing centering vector for the X-space, :math:`\mathbf{x}_\text{mean}`: :math:`KJ \times 1` in length, to mean-center the raw data.
* ``model.ppx.scale``, the preprocessing scaling vector for the X-space, :math:`\mathbf{x}_\text{scale}`: a vector, :math:`KJ \times 1` in length, to scale the centered data.
* ``model.ppy.center``, the centering vector for the Y-space, :math:`\mathbf{y}_\text{mean}`: :math:`M \times 1` in length, to mean-center the raw data.
* ``model.ppy.scale``, the scaling vector for the Y-space, :math:`\mathbf{y}_\text{scale}`: a vector, :math:`M \times 1` in length, to scale the centered data.
* ``model.limits.t_X_95``, the 95% limits, or more accurately the :math:`2\sigma` limits, for the scores: :math:`J \times A` matrix, one column per score.
* ``model.limits.t_X_99``, the 99.9% limits, or more accurately the :math:`3\sigma` limits, for the scores: :math:`J \times A` matrix, one column per score.
* ``model.limits.T2_X_95``, the 95% limits for Hotelling's |T2|: :math:`J \times 1` vector.
* ``model.limits.T2_X_99``, the 99% limits for Hotelling's |T2|: :math:`J \times 1` vector.
* ``model.limits.SPE_X_95``, the 95% limits for SPE: :math:`J \times 1` vector.
* ``model.limits.SPE_X_99``, the 99% limits for SPE: :math:`J \times 1` vector.

Support
====================================

Please email or call if there are any questions.

* Kevin Dunn
* kevin.dunn@connectmv.com
* Cell: (905) 921 5803
