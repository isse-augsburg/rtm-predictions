Utils package
===============
Package containing various util files. 

Utils.dry\_spot\_detection\_leoben
----------------------------------
* Enter a path wich shall be scanned in the code.
* Run it and keep the stdout.
* Use the following snippets to get out the runs with atypical behaviour:

.. code-block:: none

    cat dryspots.txt | grep '\[\]' | grep -x '.\{220,600\}' > dryspots_filtered.txt
    
    cat dryspots_filtered.txt | cut -c 30-59 > blacklist.txt


.. automodule:: Utils.dry_spot_detection_leoben
    :members:
    :undoc-members:
    :show-inheritance:

Utils.eval\_utils
-------------------------
The SINGULARITY_DOCKER_PASSWORD is inserted automatically now.
Make sure to insert the current docker container: Current state of the art: pytorch_19.10

.. automodule:: Utils.eval_utils
    :members:
    :undoc-members:
    :show-inheritance:

Utils.img\_utils
-------------------------

.. automodule:: Utils.img_utils
    :members:
    :undoc-members:
    :show-inheritance:

Utils.logging\_cfg
-------------------------

.. automodule:: Utils.logging_cfg
    :members:
    :undoc-members:
    :show-inheritance:

Utils.natural\_sorting
-------------------------

.. automodule:: Utils.natural_sorting
    :members:
    :undoc-members:
    :show-inheritance:

Utils.training\_utils
-------------------------

.. automodule:: Utils.training_utils
    :members:
    :undoc-members:
    :show-inheritance:

Utils.useless\_frame\_detection
-------------------------------

.. automodule:: Utils.useless_frame_detection
    :members:
    :undoc-members:
    :show-inheritance:
