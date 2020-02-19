Trainer package
===============
Package for files related to model training. 

Trainer.Model\_Trainer
------------------------------

.. automodule:: Trainer.ModelTrainer
    :members:
    :undoc-members:
    :show-inheritance:

Trainer.evaluation
-------------------------
Evaluation classes must provide three functions even if not all of them have functionality: 

* commit(output, label): updates the evaluation class with a new pair of a single prediction and a single label
* print_metrics(): prints a set of application-specific print_metrics
* reset: Resets the internal metrics of an evaluator, e.g. after a evaluation loop is finished.  


.. automodule:: Trainer.evaluation
    :members:
    :undoc-members:
    :show-inheritance:
