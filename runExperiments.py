#!/usr/bin/env python

import sys, os
import subprocess
from code import interact
import readline
import Queue
from threading import Thread
from time import sleep
import pexpect

"""This script can be used to run a command with several different command
line options on several computers. The results that are printed on the 
command line are written to a file.

Usage:

runExperiments.py [command file] [command] [additional (fixed) parameters]

Typical usage: 

runExperiments.py params.txt "java -jar experimenter.jar" "--batch"

This will run the commands "java -jar experimenter.jar --batch" combined with
each line of command line options contained in params.txt. 

The script runs these commands in separate threads (called workers), which 
can either be local or remote via a SSH connection. The script starts out 
with no workers, so in order to start the commands, workers have to be 
added using either

add_local(# local workers)   or
add_workers(["hostname1", "hostname2", ...])

The command prompt shown by the script is a full-blown Python shell, so
commands like

add_workers(["worker" + str(i) + ".foobar.com" for i in range(1,100)]) 

are also possible. The command

status()

shows the commands currently processed by the workers and also the number 
of remaining jobs.  
"""

__version__ = "$Id: runExperiments.py 153 2007-09-09 18:09:33Z sm $"

PARAM_FILE = "parameters.txt"
CMD = "./run_experiment"
CWD = "experiments"
ADDITIONAL_PARAMS = ""
NUM_LOCAL_WORKERS = 2
SSH_WORKERS = {}
LOCAL_WORKERS = {}
JOB_QUEUE = Queue.Queue(0)
RESULT_QUEUE = Queue.Queue(0)
KILLCMD = "killall python; sleep 2; ps xu | grep python"

    
## hack taken from libsvm's grid.py
## to turn queue into a stack
def _put(self,item):
    if sys.hexversion >= 0x020400A1:
        self.queue.appendleft(item)
    else:
        self.queue.insert(0,item)

import new
JOB_QUEUE._put = new.instancemethod(_put,JOB_QUEUE,JOB_QUEUE.__class__)


class ResultWatcher(Thread):
    def __init__(self,result_queue):
        Thread.__init__(self)
        self.result_queue = result_queue
        self.outfile = open("results.txt","a")
    def run(self):
       while True:
           result = self.result_queue.get()
           self.handle_result(result) 
           self.result_queue.task_done()
    
    def handle_result(self,result):
        self.outfile.write(result[2])
        self.outfile.flush()
        #print result[0] + ": " + result[2]
        
   
           
class Worker(Thread):
    def __init__(self,name,job_queue,result_queue):
        Thread.__init__(self)
        self.name = name
        self.job_queue = job_queue
        self.result_queue = result_queue
        self.running = True
        self.current_job = "idle"
        self.failures = 0
        self.status = "status not set"
    def run(self):
        while self.running:
            command = self.job_queue.get()
            try:
                self.current_job = command
                result = self.run_command(command)
                if result is None: 
                    raise Exception("get no result")
            except Exception, inst:
                # we failed, put in queue again
                self.job_queue.put(command)
                self.failures += 1
                sleep(10) # wait for 10 seconds before we try again
                if self.failures > 3:
                    sleep(60) # wait a little longer
                if self.failures > 5:
                    print "worker %s had more than 5 failures; stopping" % self.name
                    self.current_job = "failed"
                    break
            else:
                self.result_queue.put((self.name,command,result))
                self.failures = 0
            finally:
                self.job_queue.task_done()
                if self.current_job != "failed":
                    self.current_job = "idle"
        self.current_job = "stopped"

    
    def stop(self):
        self.running = False
        
def checkResultLine(line):
    """Checks whether the output line is a valid result. 
    
    Result lines consist of at least 16 fields (15 parameter values and one result)
    """
    return len(line.split('\t')) > 15 or line.strip() == "Done"

class LocalWorker(Worker):
    def run_command(self,cmdline):
        p = pexpect.spawn("nice -n 19 %s" % cmdline)
        while True:
            i = p.expect(["Done","\r\n",pexpect.EOF])
            if i == 1:
                self.status=p.before
            elif i==0:
                self.status = "Done"
                return "Done"
            else:
                print pexpect.before
                return

class SSHWorker(Worker):
    def __init__(self,host,job_queue,result_queue):
        Worker.__init__(self,host,job_queue,result_queue)
        self.host = host
        self.cwd = CWD
    def run_command(self,command):
        cmdline = 'ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -x %s "cd %s; nice -n 19 %s"' % \
          (self.host,self.cwd,command)
        p = pexpect.spawn(cmdline)
        while True:
            i = p.expect(["Done","\r\n",pexpect.EOF])
            if i == 1:
                self.status=p.before
            elif i==0:
                self.status = "Done"
                return "Done"
            else:
                print pexpect.before
                return

def usage():
    print """Usage:

    runExperiments.py [command file] [command] [additional (fixed) parameters]

Typical usage: 

    runExperiments.py params.txt "java -jar experimenter.jar" "--batch"
    """

    
def add_jobs(jobfile):
    params = [p.strip() for p in open(jobfile,"r").readlines()]
    cmds = [(CMD + " " + p + " " + ADDITIONAL_PARAMS).strip() for p in params]
    for cmd in cmds:
        JOB_QUEUE.put(cmd)        
    
def add_workers(workerList):
    for worker in workerList:
        SSH_WORKERS[worker] = SSHWorker(worker,JOB_QUEUE,RESULT_QUEUE)
        SSH_WORKERS[worker].setDaemon(True)
        SSH_WORKERS[worker].start()

def add_local(num):
    for i in range(num):
        LOCAL_WORKERS["local" + str(i)] = LocalWorker('local' + str(i),JOB_QUEUE,RESULT_QUEUE)
        LOCAL_WORKERS["local" + str(i)].setDaemon(True)
        LOCAL_WORKERS["local" + str(i)].start()
    
        
def stop_workers(workerlist):
    for worker in workerlist:
        if worker in SSH_WORKERS:
            SSH_WORKERS[worker].stop()
        if worker in LOCAL_WORKERS:
            LOCAL_WORKERS[worker].stop()
            
def stop():
    for worker in SSH_WORKERS:
        SSH_WORKERS[worker].stop()
    for worker in LOCAL_WORKERS:
        LOCAL_WORKERS[worker].stop()

def kill(workerlist):
    for worker in workerlist:
        if worker in SSH_WORKERS:
            cmdline = 'ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -x %s "%s"' % \
        (worker,KILLCMD)
            (input,result) = os.popen2(cmdline,'r')
            for line in result.readlines():
                print line

def killall():
    kill(SSH_WORKERS.keys())

def status():
    print_workers()
    print "Remaining jobs: " + str(JOB_QUEUE.qsize())    

def save_rest(filename):
   pass 
        
def print_workers():
    for worker in LOCAL_WORKERS:
        print (worker + "\t" + str(LOCAL_WORKERS[worker].failures) + "\t" 
                + LOCAL_WORKERS[worker].current_job + "\t"
                + LOCAL_WORKERS[worker].status)
    for worker in SSH_WORKERS:
        print (worker + "\t" + str(SSH_WORKERS[worker].failures) + "\t" 
                + SSH_WORKERS[worker].current_job + "\t"
                + SSH_WORKERS[worker].status)
       
## shortcuts 

    
def main():
    global PARAM_FILE, CMD, ADDITIONAL_PARAMS
    if len(sys.argv) == 1:
        usage()
        exit()
    if len(sys.argv) > 1:
        PARAM_FILE = sys.argv[1]
    if len(sys.argv) > 2:
        CMD = sys.argv[2]
    if len(sys.argv) > 3:
        ADDITIONAL_PARAMS = ' '.join(sys.argv[3:])
    print "FILE: " + PARAM_FILE + "; CMD: " + CMD + "; ADDITIONAL_PARAMS: " + ADDITIONAL_PARAMS
    
    add_jobs(PARAM_FILE)
    res = ResultWatcher(RESULT_QUEUE)
    res.setDaemon(True)
    res.start()
    
    interact("runExperiments Interactive Shell",raw_input,globals())
    
    JOB_QUEUE.join()
    print "All jobs completed."
    RESULT_QUEUE.join()
    
if __name__ == "__main__":
    main()
