import streamlit as st
import cv2
import os
from os import listdir
import stat
import Third_party.E2FGVI.test as model


def load_video():
    """
    function load source video
    :return: void
    """
    uploaded_file = st.file_uploader(
        label='Upload video', type=['mp4'])
    if uploaded_file is not None:
        if not uploaded_file.type == "video/mp4":
            st.error('Need only mp4')
            return
        root_dir = 'dataset'
        fullname = os.path.join(root_dir, 'video')
        if not os.path.exists(root_dir):
            os.mkdir(root_dir)
        if not os.path.exists(fullname):
            os.mkdir(fullname)

        with open(os.path.join(fullname, "video.mp4"), "wb") as f:
            f.write(uploaded_file.getbuffer())
        try:
            read_frame_from_videos(os.path.join(fullname, "video.mp4"), 'video_frames')
        except FileNotFoundError as error:
            print("[-]", error)
            st.error("The file does not fit :(")
            return
    else:
        return


def load_masks():
    """
    function load mask_video
    :return: void
    """
    uploaded_file = st.file_uploader(
        label='Upload mask', type=['mp4'])
    if uploaded_file is not None:
        if not uploaded_file.type == "video/mp4":
            st.error('Need only mp4')
            return
        root_dir = 'dataset'
        fullname = os.path.join(root_dir, 'mask')
        if not os.path.exists(root_dir):
            os.mkdir(root_dir)
        if not os.path.exists(fullname):
            os.mkdir(fullname)
        with open(os.path.join(fullname, "video.mp4"), "wb") as f:
            f.write(uploaded_file.getbuffer())
        try:
            read_frame_from_videos(os.path.join(fullname, "video.mp4"), 'mask_frames')
        except FileNotFoundError as error:
            print("[-]", error)
            st.error("The file does not fit :(")
            return
    else:
        return


def clean_thrash(paths: list):
    """
    function delete all files in paths's list
    :param paths: list of paths to delete
    :return: void
    """
    for path in paths:
        try:
            dirlist = listdir(path)
            for f in dirlist:
                fullname = os.path.join(path, f)
                if os.path.isfile(fullname):
                    os.chmod(fullname, stat.S_IWRITE)
                    os.remove(fullname)
                if os.path.isdir(fullname):
                    clean_thrash([fullname])
        except FileNotFoundError as error:
            print("[-]", error)


def clean_folders(paths: list):
    """
    function delete all empty's folders
    Throw exception if folder isn't empty
    :param paths: list of paths to delete
    :return: void
    """
    for path in paths:
        try:
            dirlist = listdir(path)
            for file in dirlist:
                fullname = os.path.join(path, file)
                if os.path.isdir(fullname):
                    try:
                        os.rmdir(fullname)
                    except OSError:
                        clean_folders([fullname])
                else:
                    raise OSError("[-] Folders need to be empty! Path: " + str(fullname))
            os.rmdir(path)
        except FileNotFoundError as error:
            print("[-]", error)


def read_frame_from_videos(path, name):
    """
    function that make frames from video
    :param path: path to video
    :param name: path to dir for save frames
    :return: void
    """
    if not os.path.exists(path):
        raise FileNotFoundError("[-] File doesn't exist! Path:" + path)
    if not os.path.exists(os.path.join('dataset', name)):
        os.mkdir(os.path.join('dataset', name))
    capture = cv2.VideoCapture(path)
    frameNr = 0
    while True:
        success, frame = capture.read()
        if success:
            cv2.imwrite(f'dataset/{name}/{frameNr}.jpg', frame)
        else:
            break
        frameNr += 1
    capture.release()


st.title('Deleting object from video')

load_video()
load_masks()

result = st.button('Start deleting')
if result:
    with st.spinner('We are making some magic...'):
        model.main_worker("e2fgvi_hq", "dataset/video_frames", "dataset/mask_frames",
                          "Third_party/E2FGVI/release_model/E2FGVI-"
                          "HQ-CVPR22.pth")
    st.success('Done!')
    with open('results/result.mp4', 'rb') as video:
        st.download_button(
            label="Download result video",
            data=video,
            file_name="result.mp4",
            mime="video/mp4"
        )
    clean_thrash(['dataset', 'results'])
    clean_folders(['dataset', 'results'])
