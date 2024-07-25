#!/bin/tcsh

    ### LSF syntax
    #BSUB -nnodes 8                   #number of nodes
    #BSUB -W 240                      #walltime in minutes
    #BSUB -G guests                   #account
    #BSUB -e myerrors.txt             #stderr
    #BSUB -o myoutput.txt             #stdout
    #BSUB -J myjob                    #name of job
    #BSUB -q pbatch                   #queue to use

   python3 newTransformer1.py