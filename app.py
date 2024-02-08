from tkinter import filedialog, ttk
from guizero import App, PushButton, Text, Box, ListBox, TextBox, TitleBox, Picture, Slider, Combo
from PIL import Image, ImageOps, ImageDraw
from skimage import io, exposure, img_as_ubyte, img_as_float
import numpy as np
from pathlib import Path
import logging
from concurrent.futures import ProcessPoolExecutor
import time


def Start():
 
    logging.basicConfig(level=logging.INFO)
    THUMBNAIL_SIZE = 300
    original_img = None
    image_files = []
    dummy_image = Image.new("RGB", (THUMBNAIL_SIZE, THUMBNAIL_SIZE), "white")

    def TimeIt(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            logging.info(f"{func.__name__} executed in {
                         end_time - start_time} seconds")
            return result
        return wrapper

    def OnSliderCutoff():
        # img = picture_Original.image
        global original_img
        ac_prev = ComputeAC(original_img, slider_Cutoff.value)
        ac_prev.thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE))
        picture_AC.image = ac_prev

    def OnListBoxChanged(file):
        if(listBox_Files.items):
            GeneratePreviews(file)

    def OpenFolder():
        # global image_files
        selectedFolder = filedialog.askdirectory(
            title="Choose Image Folder", mustexist=True, initialdir=str(Path.home()))
        if (selectedFolder):
            logging.info(f"Selected folder: {selectedFolder}")
            image_files = LoadImages(selectedFolder)
            listBox_Files.clear()
            add_items_in_batches(image_files)
        else:
            logging.info(f"Folder selection canceled")

    def add_items_in_batches(image_files, index=0, batch_size=100):
        next_index = index + batch_size
        for file in image_files[index:next_index]:
            listBox_Files.append(file)
        if next_index < len(image_files):
            # Schedule the next batch
            listBox_Files.after(1, add_items_in_batches, args=[
                                image_files, next_index, batch_size])
            msg = f"Loading files {next_index}/{len(image_files)}"
            text_Status.value = msg
            logging.info(msg)
        else:
            msg = f"{len(image_files)} files loaded"
            text_Status.value = msg
            logging.info(msg)

    @TimeIt
    def LoadImages(folder_path):
        image_patterns = ['*.jpg', '*.jpeg', '*.png',
                          '*.gif', '*.bmp', '*.tif', '*.tiff', '*.webp']
        image_files = []
        folder_path = Path(folder_path)

        if folder_path.exists() and folder_path.is_dir():
            for pattern in image_patterns:
                image_files.extend(folder_path.rglob(pattern))
        return image_files

    def GeneratePreviews(image_File):
        global original_img
        original_img = Image.open(image_File)
        if original_img.mode in ['I', 'I;16', 'I;16B']:
            logging.info(f"16 bit Image")
            original_prev = original_img.point(
                lambda i: i * (1/256)).convert('L')
        else:
            logging.info(f"Not a 16 bit Image")
            original_prev = original_img.convert('L')

        original_prev.thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE))
        picture_Original.image = original_prev
        ac_prev = ComputeAC(original_img, cutoff=slider_Cutoff.value)
        ac_prev.thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE))
        picture_AC.image = ac_prev
        he_prev = ComputeHE(original_img)
        he_prev.thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE))
        picture_HE.image = he_prev
        ahe_prev = ComputeAHE(original_img)
        ahe_prev.thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE))
        picture_AHE.image = ahe_prev

    def ComputeAC(image, cutoff=1):
        # np_img = np.array(image)
        # p_lower, p_upper = np.percentile(np_img, (2, 98))
        # img_rescaled = exposure.rescale_intensity(np_img, in_range=(p_lower, p_upper))
        # img_cs = np.array(img_rescaled *255, dtype=np.uint8)
        # return Image.fromarray(img_cs)
        if image.mode in ['I', 'I;16', 'I;16B']:
            img_cs = image.point(lambda i: i * (1/256)).convert('L')
        else:
            img_cs = image.convert('L')
        img_cs = ImageOps.autocontrast(img_cs, cutoff=cutoff)
        return img_cs

    def ComputeHE(image):
        np_img = np.array(image)
        img_eq = exposure.equalize_hist(np_img)
        img_eq = np.array(img_eq * 255, dtype=np.uint8)
        return Image.fromarray(img_eq)

    def ComputeAHE(image):
        np_img = np.array(image)
        img_ahe = exposure.equalize_adapthist(np_img)
        img_ahe = np.array(img_ahe * 255, dtype=np.uint8)
        return Image.fromarray(img_ahe)

    def PlotHistogram(image, image_size=(THUMBNAIL_SIZE, THUMBNAIL_SIZE), background_color='white'):
        histogram = image.histogram()
        histogram_image = Image.new('RGB', image_size, color=background_color)
        draw = ImageDraw.Draw(histogram_image)
        max_hist_value = max(histogram)
        scale = image_size[1] / max_hist_value
        bin_width = image_size[0] / 256
        for i, value in enumerate(histogram[:256]):
            height = value * scale
            x_start = i * bin_width
            x_end = x_start + bin_width
            draw.rectangle([x_start, image_size[1] - height, x_end, image_size[1]], fill='black')
        
        picture_Hist.image = histogram_image
    
    def On_MouseClick_Preview(event_data):
        PlotHistogram(event_data.widget.image)


    app = App(title="XRView", layout="auto", width=1440, height=900)
    app.tk.config(padx=5, pady=5)

    box_Toolbar = Box(app, align="top", width="fill")
    box_Toolbar.tk.config(padx=5, pady=5)
    button_OpenFolder = PushButton(
        box_Toolbar, text="Open Folder", command=OpenFolder, align="left")
    button_BatchCompute = PushButton(
        box_Toolbar, text="Batch Compute", align="left")
    combo_Compute = Combo(box_Toolbar, align="left", options=[
                          "Auto Contrast", "Histogram Equalization (HE)", "Adaptive HE"])
    

    box_Main = Box(app, border=0, width="fill", height="fill")
    box_Main.tk.config(padx=5, pady=5)

    titleBox_Files = TitleBox(box_Main, "Image Files",
                              width="fill", height="fill")
    titleBox_Files.tk.config(padx=5, pady=5)
    listBox_Files = ListBox(titleBox_Files, scrollbar=True,
                            width="fill", height="fill", command=OnListBoxChanged)
    titleBox_Previews = TitleBox(box_Main, "Previewes", width="fill")
    titleBox_Previews.tk.config(padx=5, pady=5)

    box_original = Box(titleBox_Previews, align="left", height="fill")
    titleBox_Original = TitleBox(box_original, "Original", align="top")
    titleBox_Original.tk.config(padx=5, pady=5)
    picture_Original = Picture(
        titleBox_Original, align="left", image=dummy_image)
    picture_Original.when_clicked = On_MouseClick_Preview

    # Box(titleBox_Previews, align="left", width="fill")

    box_AC = Box(titleBox_Previews, align="left", height="fill")
    titleBox_AC = TitleBox(box_AC, "Auto Contrast", align="top")
    titleBox_AC.tk.config(padx=5, pady=5)
    picture_AC = Picture(titleBox_AC, align="top", image=dummy_image)
    slider_Cutoff = Slider(titleBox_AC, command=OnSliderCutoff,
                           start=1, end=49, align="bottom", width="fill")
    picture_AC.when_clicked = On_MouseClick_Preview

    # Box(titleBox_Previews, align="left", width="fill")

    box_HE = Box(titleBox_Previews, align="left", height="fill")
    titleBox_HE = TitleBox(box_HE, "Histogram Equalization", align="top")
    titleBox_HE.tk.config(padx=5, pady=5)
    picture_HE = Picture(titleBox_HE, align="left", image=dummy_image)
    picture_HE.when_clicked = On_MouseClick_Preview

    # Box(titleBox_Previews, align="left", width="fill")

    box_AHE = Box(titleBox_Previews, align="left", height="fill")
    titleBox_AHE = TitleBox(
        box_AHE, "Adaptive Histogram Equalization", align="top")
    titleBox_AHE.tk.config(padx=5, pady=5)
    picture_AHE = Picture(
        titleBox_AHE, align="left", image=dummy_image)
    picture_AHE.when_clicked = On_MouseClick_Preview
    
    # Box(titleBox_Previews, align="left", width="fill")

    box_Hist = Box(titleBox_Previews, align="left", height="fill")
    titleBox_Hist = TitleBox(box_Hist, "Histogram", align="top")
    titleBox_Hist.tk.config(padx=5, pady=5)
    picture_Hist = Picture(titleBox_Hist, image=dummy_image)

    box_Status = Box(app, width="fill", layout="grid")
    box_Status.tk.config(padx=5, pady=5)
    box_Status.tk.columnconfigure(0, weight=1)
    text_Status = Text(box_Status, "Welcome to XRView", align="left", grid=[0,0])
    prog_bar = ttk.Progressbar(box_Status.tk, mode="determinate", orient="horizontal")
    prog_bar.grid(row=1, column=0, sticky="ew")
    # box_Status.add_tk_widget(prog_bar, grid=[0,1], width="fill")

    app.display()


if __name__ == "__main__":
    Start()
