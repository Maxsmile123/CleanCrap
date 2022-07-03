from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os


gauth = GoogleAuth()
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)

file_obj = drive.CreateFile({'id': '1tNJMTJ2gmWdIXJoHVi5-H504uImUiJW9'})
file_obj.GetContentFile('./E2FGVI/release_model/E2FGVI_CVPR22_models.zip')

try:
    os.system('unzip ./E2FGVI/release_model/E2FGVI_CVPR22_models.zip')
except:
    pass

file_obj = drive.CreateFile({'id': '10wGdKSUOie0XmCr8SQ2A2FeDe-mfn5w3'})
file_obj.GetContentFile('./E2FGVI/release_model/E2FGVI-HQ-CVPR22.pth')

