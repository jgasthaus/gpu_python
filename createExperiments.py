#!/usr/bin/env python

import sys, os

"""Create a file containg option settings for performing grid-search experiments

This script creates files containing lines of command-line arguments that can
be used with the runExperiments.py script to run several experiments in
batch mode. 

Usage: createExperiments.py

The script does not take any command-line arguments. All parameters have to 
be specified in the source file (see below for an example).
"""

__version__ = "$Id: createExperiments.py 152 2007-09-09 18:00:59Z sm $"

##### EXPERIMENT DEFINITION BEGIN

# parameters that are always availbe and should be iterated over 
# all possible values
# Format name:[value1,...,valueN] format.




# possible parameter are specified as functions from a string representing
# a partial parameter selection to a list of possible additional parameters
    
GENERAL_PARAMETERS = {"cfg":lambda x: ["sensitivity/s-good.cfg", "sensitivity/s-narrow.cfg", "sensitivity/s-broad.cfg"],
                      "aux": lambda x: ["--aux-vars " + str(i) for i in [1,30,100]],
                      "rho": lambda x: ["--rho " + i for i in ["0.9", "0.985", "1"]], 
                      "alpha": lambda x: ["--alpha " + i for i in ["1", "0.1", "0.01"]],
                      "id" : lambda x :  ["--suffix=" + x.replace(" --","_").replace(" ","_")], 
                      "rows": lambda x: ["--rows 1000\""],
                      }

PARAMETER_ORDER = ["aux", "rho", "alpha","id", "rows"]
##### EXPERIMENT DEFINITION END


def usage():
    pass

def generate():
    current = [""]
    for param in PARAMETER_ORDER:
        current = [i + " " + j for i in current for j in GENERAL_PARAMETERS[param](i)]
    return current
    
def main():
    usage()    
    params = generate()
    #print len(params)
    for p in params:
        print p.strip()
    
if __name__ == "__main__":
    main()
