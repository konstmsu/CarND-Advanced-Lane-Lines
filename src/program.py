import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
from matplotlib.widgets import Slider, Button, RadioButtons

for i in glob("../test_images2/*"):
    print(i)
    img = Image.open(i)
    fig = plt.figure()
    plt.imshow(img)
    samp = Slider(plt, 'Amp', 0.1, 10.0, valinit=2)
    plt.show()

    break
