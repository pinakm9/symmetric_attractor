import cv2
import os, shutil
import utility as ut
from PIL import Image, ImageEnhance

class CollageFromVideo:
    """
    Description:
        A class for creating a collage out of a video file
    
    Attributes:
        file_in: input video file path
        file_out: out image file path
        shape: shape of the collage
        width: width of the collage in pixels
        heght: height of the collage in pixels
        n_frames: number of frames to extract
        temp_folder: a temporary folder for storing extracted frames
    """
    def __init__(self, file_in, file_out, width=2000, height=2000):
        self.file_in = file_in
        self.file_out = file_out
        self.width = width
        self.height = height
        
        # create a temp folder for storing the extracted images 
        self.temp_folder = os.path.dirname(file_out) + '/collage_temp'
        if not os.path.isdir(self.temp_folder):
            os.makedirs(self.temp_folder)
        

    
    @ut.timer
    def extract_frames(self, shape):
        """
        Extracts necessary frames from the video files
        """
        self.shape = shape
        self.n_frames = shape[0] * shape[1]
        # access video and figure out interval size
        video = cv2.VideoCapture(self.file_in)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        interval = int(total_frames / self.n_frames)
        # extract frames
        count = 0
        success, image = video.read()
        while count < self.n_frames:
            video.set(cv2.CAP_PROP_POS_MSEC, (1000 * count * interval / fps)) 
            success, image = video.read()
            print ('Extracting frame #{}'.format(count), end='\r')
            # save frame as JPEG file
            cv2.imwrite(self.temp_folder + '/frame_{}.jpg'.format(count), image)
            count += 1

    
    @ut.timer
    def make_collage(self, shape):
        """
        Stiches the extracted frames into the required shape
        """
        # extract frames
        self.extract_frames(shape=shape)
        # make collage
        w, h = int(self.width/self.shape[1]), int(self.height/self.shape[0])
        collage = Image.new('RGB', size=(self.width, self.height))
        for row in range(self.shape[0]):
            for col in range(self.shape[1]):
                frame_id = row * self.shape[1] + col 
                img = Image.open(self.temp_folder + '/frame_{}.jpg'.format(frame_id))
                img = img.resize((w, h))
                collage.paste(img, (col * w, row * h))
        collage.save(self.file_out)
        # clean up
        shutil.rmtree(self.temp_folder)



class CollageFromImages:
    """
    Description:
        A class for creating a collage out of a video file
    
    Attributes:
        img_paths: a list of image locations
        file_out: out image file path
        shape: shape of the collage
        width: width of the collage in pixels
        heght: height of the collage in pixels

    """
    def __init__(self, file_out, width=2000, height=2000):
        self.file_out = file_out
        self.width = width
        self.height = height
        self.img_paths = []

    
    def add_image(self, path):
        """
        Adds the location of a new image to the list of image locations
        """
        self.img_paths.append(path)

    def add_images(self, paths):
        """
        Adds the locations of new images to the list of image locations
        """
        self.img_paths += paths

    
    @ut.timer
    def make_collage(self, shape, clean_up=False):
        """
        Stiches the images into the required shape
        """
        print(self.img_paths)
        self.shape = shape
        # make collage
        w, h = int(self.width/self.shape[1]), int(self.height/self.shape[0])
        collage = Image.new('RGB', size=(self.width, self.height))
        for row in range(self.shape[0]):
            for col in range(self.shape[1]):
                frame_id = row * self.shape[1] + col 
                img = Image.open(self.img_paths[frame_id])
                img = img.resize((w, h))
                collage.paste(img, (col * w, row * h))
                # clean up if needed
                if clean_up:
                    os.remove(self.img_paths[frame_id])
        collage.save(self.file_out)