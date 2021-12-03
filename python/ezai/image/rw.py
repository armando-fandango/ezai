import numpy as np
def image_write(filename:str, image:np.ndarray):
    try:
        from cv2 import imwrite as imwrite
        print("using cv2 as image library")
    except ImportError as error:
        try:
            from imageio import imwrite as imwrite
            print("using imageio as image library")
        except ImportError as error:
            try:
                from matplotlib.pyplot import imsave as imwrite
                print("using matplotlib as image library")
            except ImportError as error:
                try:
                    from PIL import Image
                    print("using PIL as image library")

                    def imwrite(filename, arr):
                        im = Image.fromarray(arr)
                        im.save(filename)
                except ImportError as error:
                    def imwrite(filename=None, arr=None):
                        print("unable to load any of the following "
                              "image libraries: cv2, imageio, matplotlib, "
                              "PIL. Install one of these libraries to "
                              "save visuals.")

                    imwrite()
