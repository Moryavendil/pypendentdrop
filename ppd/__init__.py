
verbose = 0
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def set_verbose(v:int) -> None:
    global verbose
    verbose = v

def error(text:str):
    if True:
        print(f'{bcolors.FAIL}ERROR:{bcolors.ENDC} {text}')

def warning(text:str):
    if True:
        print(f'{bcolors.WARNING}WARNING:{bcolors.ENDC} {text}')

def info(text:str):
    global verbose
    if verbose > 0:
        print(f'{bcolors.OKGREEN}INFO:{bcolors.ENDC} {text}')

def debug(text:str):
    global verbose
    if verbose > 1:
        print(f'\t{bcolors.OKCYAN}DEBUG:{bcolors.ENDC} {text}')

def trace(text:str):
    global verbose
    if verbose > 2:
        print(f'\t\t{bcolors.OKBLUE}TRACE:{bcolors.ENDC} {text}')

