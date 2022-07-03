import os

CODE_DIR = 'E2FGVI'
os.makedirs(f'./{CODE_DIR}')
os.system("cd " + CODE_DIR)
os.system("git clone https://github.com/MCG-NKU/E2FGVI.git")
os.chdir(f'./{CODE_DIR}')
