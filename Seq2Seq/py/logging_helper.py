import logging
import sys

def mylog(msg):
    print(msg)
    sys.stdout.flush()
    logging.info(msg)


def mylog_section(section_name):
    mylog("======== {} ========".format(section_name)) 

def mylog_subsection(section_name):
    mylog("-------- {} --------".format(section_name)) 

def mylog_line(section_name, message):
    mylog("[{}] {}".format(section_name, message))
