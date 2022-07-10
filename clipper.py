import cv2
import os
import numpy as np

save_mask_path = './mask.mp4'
save_video_path = './video.mp4'

def load_images_from_folder(folder):
    """
    function that return list of frames from folder path
    :param folder: path to frames
    :return: list of frames
    """
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


def create_video(path_to_folder, save_path, default_fps=30, size=(854, 480)):
    """
    function that creates video from frames
    :param size: size of video. Default for test dataset
    :param default_fps: fps of video
    :param path_to_folder: path to frames
    :param save_path: path to save video
    :return: void
    """
    try:
        writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 default_fps, size)

        image = load_images_from_folder(path_to_folder)

        for f in range(len(image)):
            writer.write(cv2.cvtColor(image[f].astype(np.uint8), cv2.COLOR_BGR2RGB))
        writer.release()
    except ... as e:
        print(e)


if __name__ == "__main__":
    path_to_folder = input()
    save_path = input()
    create_video(path_to_folder, save_path)
