from tkinter import filedialog, ttk
from guizero import App, PushButton, Text, Box, ListBox, TitleBox, Picture, Slider, Combo
from PIL import Image, ImageDraw
from skimage import io, exposure, util, color, transform
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from io import BytesIO
import logging
from concurrent.futures import ProcessPoolExecutor
import time


logging.basicConfig(level=logging.INFO)
THUMBNAIL_SIZE = 300
dummy_image = Image.new("L", (THUMBNAIL_SIZE, THUMBNAIL_SIZE), "white")
original_img = None

def Start():
    def TimeIt(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            logging.info(f"{func.__name__} executed in {end_time - start_time} seconds")
            return result
        return wrapper

    def On_Slider_AC_Cutoff():
        ac_prev = ComputeAC(original_img, slider_AC_Cutoff.value)
        picture_AC.image = Image.fromarray(ac_prev)
        picture_Hist.image = PlotHistogram(picture_AC.image)

    def On_MouseEnter_Preview(event_data):
        PlotHistogram(event_data.widget.image)

    def On_ListBoxChanged(file):
        if (listBox_Files.items):
            GeneratePreviews(file)

    def On_ComboColormapChanged():
        GeneratePreviews(listBox_Files.value)

    def OpenFolder():
        selectedFolder = filedialog.askdirectory(title="Choose Image Folder", mustexist=True, initialdir=str(Path.home()))
        if (selectedFolder):
            logging.info(f"Selected folder: {selectedFolder}")
            image_files = LoadImages(selectedFolder)
            UpdateListBox(image_files)
        else:
            logging.info(f"Folder selection canceled")

    @TimeIt
    def LoadImages(folder_path):
        image_patterns = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tif', '*.tiff', '*.webp']
        image_files = []
        folder_path = Path(folder_path)

        if folder_path.exists() and folder_path.is_dir():
            for pattern in image_patterns:
                image_files.extend(folder_path.rglob(pattern))
        return image_files

    @TimeIt
    def UpdateListBox(image_files):
        listBox_Files.clear()
        listBox_Files.children[0].tk.insert("end", *image_files)
        msg = f"{len(image_files)} files loaded"
        text_Status.value = msg
        logging.info(msg)

    def GeneratePreviews(image_File):
        global original_img
        img = io.imread(image_File, as_gray=True)
        img = transform.rescale(img, 0.25, anti_aliasing=True)
        original_img = util.img_as_float64(exposure.rescale_intensity(img))
        if(combo_Colormap.value != 'gray'):
            original_img = plt.get_cmap(combo_Colormap.value)(original_img)[:, :, :3]
        original_prev = util.img_as_ubyte(original_img)

        h, w = original_prev.shape[:2]
        prev_height = int(THUMBNAIL_SIZE / (w / h))

        picture_Original.height = prev_height
        original_prev = Image.fromarray(original_prev)
        picture_Original.image = original_prev
        if(not slider_AC_Cutoff.enabled):
            slider_AC_Cutoff.enable()
            slider_AC_Cutoff.value = 2

        ac_prev = ComputeAC(original_img, cutoff=slider_AC_Cutoff.value)
        ac_prev = Image.fromarray(ac_prev)
        picture_AC.height = prev_height
        picture_AC.image = ac_prev

        he_prev = ComputeHE(original_img)
        he_prev = Image.fromarray(he_prev)
        picture_HE.height = prev_height
        picture_HE.image = he_prev

        ahe_prev = ComputeAHE(original_img)
        ahe_prev = Image.fromarray(ahe_prev)
        picture_AHE.height = prev_height
        picture_AHE.image = ahe_prev

        PlotHistogram(picture_Original.image)

    def ComputeAC(image, cutoff=2):
        p_lower, p_upper = np.percentile(image, (0+cutoff, 100-cutoff))
        img_ac = exposure.rescale_intensity(
            image, in_range=(p_lower, p_upper), out_range=np.uint8)
        return img_ac

    def ComputeHE(image):
        img_eq = exposure.equalize_hist(image)
        return util.img_as_ubyte(img_eq)

    def ComputeAHE(image):
        img_ahe = exposure.equalize_adapthist(image)
        return util.img_as_ubyte(img_ahe)
    
    def PlotHistogram(image, image_size=(THUMBNAIL_SIZE, THUMBNAIL_SIZE), background_color='white'):
        histogram = image.histogram()
        img_array = np.array(image)
        img_cdf, bins = exposure.cumulative_distribution(img_array)
        histogram_image = Image.new('RGB', image_size, color=background_color)
        draw = ImageDraw.Draw(histogram_image)
        max_hist_value = max(histogram)
        scale_hist = image_size[1] / max_hist_value
        scale_cdf = image_size[1]
        bin_width = image_size[0] / 256
        for i, value in enumerate(histogram[:256]):
            height = value * scale_hist
            x_start = i * bin_width
            x_end = x_start + bin_width
            draw.rectangle([x_start, image_size[1] - height, x_end, image_size[1]], fill='blue')
        for i in range(1, len(img_cdf)):
            x_start = (i - 1) * bin_width
            x_end = i * bin_width
            y_start = image_size[1] - img_cdf[i - 1] * scale_cdf
            y_end = image_size[1] - img_cdf[i] * scale_cdf
            draw.line([x_start, y_start, x_end, y_end], fill='red', width=2)

        picture_Hist.image = histogram_image

    def PlotHistogramPlt(input_image, width_px=THUMBNAIL_SIZE, height_px=THUMBNAIL_SIZE):
        default_dpi = plt.rcParams['figure.dpi']
        
        figsize_inch_width = width_px / default_dpi
        figsize_inch_height = height_px / default_dpi
        img_array = np.array(input_image)
        img_array = util.img_as_float(img_array)
        if len(img_array.shape) == 3:
            img_array = color.rgb2gray(img_array)
        
        fig, ax_hist = plt.subplots(figsize=(figsize_inch_width, figsize_inch_height))
        ax_cdf = ax_hist.twinx()
        ax_hist.hist(img_array.ravel(), bins=256, color='blue', alpha=0.6)
        ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
        ax_hist.set_xlim(0, 1)
        ax_hist.set_yticks([])
        img_cdf, bins = exposure.cumulative_distribution(img_array)
        ax_cdf.plot(bins, img_cdf, 'r')
        ax_cdf.set_yticks([])
        plt.tight_layout()

        with BytesIO() as buf:
            plt.savefig(buf, format='png', dpi=default_dpi)
            buf.seek(0)
            output_image = Image.open(buf).copy()
        
        plt.close(fig)
        picture_Hist.image = output_image


    app = App(title="XRView", layout="auto", width=1600, height=900)
    app.tk.config(padx=5, pady=5)

    box_Toolbar = Box(app, align="top", width="fill")
    box_Toolbar.tk.config(padx=5, pady=5)
    button_OpenFolder = PushButton(box_Toolbar, text="Open Folder", command=OpenFolder, align="left")
    button_BatchCompute = PushButton(box_Toolbar, text="Batch Compute", align="left")
    combo_Compute = Combo(box_Toolbar, align="left", options=["Auto Contrast", "Histogram Equalization (HE)", "Adaptive HE"])

    box_Main = Box(app, border=0, width="fill", height="fill")
    box_Main.tk.config(padx=5, pady=5)

    titleBox_Files = TitleBox(box_Main, "Image Files", width="fill", height="fill")
    titleBox_Files.tk.config(padx=5, pady=5)
    listBox_Files = ListBox(titleBox_Files, scrollbar=True, width="fill", height="fill", command=On_ListBoxChanged)
    titleBox_Previews = TitleBox(box_Main, "Previewes", width="fill")
    titleBox_Previews.tk.config(padx=5, pady=5)

    box_Colormap = Box(titleBox_Previews, width='fill')
    box_Colormap.tk.config(padx=5, pady=5)
    text_Colormap = Text(box_Colormap, "Colormap: ", align="left")
    combo_Colormap = Combo(box_Colormap, ['gray', 'viridis'], align='left', command=On_ComboColormapChanged)

    box_original = Box(titleBox_Previews, align="left", height="fill")
    titleBox_Original = TitleBox(box_original, "Original", align="top")
    titleBox_Original.tk.config(padx=5, pady=5)
    picture_Original = Picture(titleBox_Original, align="left", image=dummy_image)
    picture_Original.when_mouse_enters = On_MouseEnter_Preview

    # Box(titleBox_Previews, align="left", width="fill")

    box_AC = Box(titleBox_Previews, align="left", height="fill")
    titleBox_AC = TitleBox(box_AC, "Auto Contrast", align="top")
    titleBox_AC.tk.config(padx=5, pady=5)
    picture_AC = Picture(titleBox_AC, align="top", image=dummy_image)
    slider_AC_Cutoff = Slider(titleBox_AC, enabled=False, command=On_Slider_AC_Cutoff, start=0, end=49, align="bottom", width="fill")
    picture_AC.when_mouse_enters = On_MouseEnter_Preview

    # Box(titleBox_Previews, align="left", width="fill")

    box_HE = Box(titleBox_Previews, align="left", height="fill")
    titleBox_HE = TitleBox(box_HE, "Histogram Equalization", align="top")
    titleBox_HE.tk.config(padx=5, pady=5)
    picture_HE = Picture(titleBox_HE, align="left", image=dummy_image)
    picture_HE.when_mouse_enters = On_MouseEnter_Preview

    # Box(titleBox_Previews, align="left", width="fill")

    box_AHE = Box(titleBox_Previews, align="left", height="fill")
    titleBox_AHE = TitleBox(box_AHE, "Adaptive Histogram Equalization", align="top")
    titleBox_AHE.tk.config(padx=5, pady=5)
    picture_AHE = Picture(titleBox_AHE, align="left", image=dummy_image)
    picture_AHE.when_mouse_enters = On_MouseEnter_Preview

    # Box(titleBox_Previews, align="left", width="fill")

    box_Hist = Box(titleBox_Previews, align="left", height="fill")
    titleBox_Hist = TitleBox(box_Hist, "Histogram", align="top")
    titleBox_Hist.tk.config(padx=5, pady=5)
    picture_Hist = Picture(titleBox_Hist, image=dummy_image)

    box_Status = Box(app, width="fill", layout="grid")
    box_Status.tk.config(padx=5, pady=5)
    box_Status.tk.columnconfigure(0, weight=1)
    text_Status = Text(box_Status, "Welcome to XRView",
                       align="left", grid=[0, 0])
    prog_bar = ttk.Progressbar(box_Status.tk, mode="determinate", orient="horizontal")
    prog_bar.grid(row=1, column=0, sticky="ew")
    # box_Status.add_tk_widget(prog_bar, grid=[0,1], width="fill")

    app.display()


if __name__ == "__main__":
    Start()
