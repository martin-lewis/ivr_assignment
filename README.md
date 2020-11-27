IVR Assignment

Completed by Maksymilian Mozolewski & Martin Lewis

It requires the sympy library, can be installed with
>pip3 install sympy

It also requires that full absolute filepaths be used for the two template images. Lines 446 and 456 in tasks_node.py need to go from root (/) to the two files in the github repo (template-box.png and template-sphere.png), please update these to your machine otherwise they will not work and will return errors.

The requested code for Task 3 can be run by simply calling the command
>rosrun ivr_assignment tasks_node.py
