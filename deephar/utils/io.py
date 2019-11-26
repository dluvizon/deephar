import sys

HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'

def printc(color, vmsg):
    print (color + vmsg + ENDC, end='')
    sys.stdout.flush()

def printcn(color, vmsg):
    print (color + vmsg + ENDC)
    sys.stdout.flush()

def printnl(vmsg):
    sys.stdout.write(vmsg + '\n')
    sys.stdout.flush()

def warning(vmsg):
    sys.stderr.write(WARNING + vmsg + ENDC + '\n')
    sys.stderr.flush()

def sprintcn(color, vmsg):
    return color + vmsg + ENDC + '\n'

def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

def sizeof_eng_fmt(num):
    for unit in ['','K','M','G','T','P','E','Z']:
        if abs(num) < 1e3:
            return "%3.1f%s" % (num, unit)
        num /= 1e3
    return "%3.1f%s" % (num, 'Y')
