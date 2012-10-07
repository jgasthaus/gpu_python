# Copyright (c) 2008-2011, Jan Gasthaus
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer. 
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.  
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
