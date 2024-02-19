import os
import time
import logging
import threading
from pathlib import Path
from functools import partial
from tkinter import filedialog, ttk
from guizero import App, PushButton, Text, Box, ListBox, TitleBox, Picture, Slider, Combo, CheckBox
import numpy as np
from matplotlib import pyplot as plt
from skimage import io, exposure, util, color, transform
from PIL import Image, ImageDraw
from concurrent.futures import ThreadPoolExecutor, as_completed


logging.basicConfig(level=logging.INFO)
THUMBNAIL_SIZE = 300
dummy_image = Image.new("L", (THUMBNAIL_SIZE, THUMBNAIL_SIZE), "white")
original_img = None

def TimeIt(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"{func.__name__} executed in {end_time - start_time} seconds")
        return result
    return wrapper

def BatchCompute(file, selectedFolder, compute, cmap, cutoff, autoCrop):
    img = io.imread(file)
    original_dtype = img.dtype
    if(original_dtype == np.uint16):
        img = util.img_as_float64(img)
    else:
        img = color.rgb2gray(img)
    if(autoCrop):
        img = AutoCrop(img)
    path = Path(file)
    selectedFolder = Path(selectedFolder)

    if(compute == "Auto Contrast"):
        p_lower, p_upper = np.percentile(img, (0+cutoff, 100-cutoff))
        img_ac = exposure.rescale_intensity(img, in_range=(p_lower, p_upper))
        if(cmap != 'gray'):
            img_ac = plt.get_cmap(cmap)(img_ac)[:, :, :3]
        new_path = selectedFolder / "AutoContrast" / path.name
        new_path.parent.mkdir(parents=True, exist_ok=True)
        if(original_dtype == np.uint8):
            img_ac = util.img_as_ubyte(img_ac)
        elif(original_dtype == np.uint16):
            img_ac = util.img_as_uint(img_ac)
        io.imsave(new_path, img_ac)

    elif(compute == "Histogram Equalization (HE)"):
        img_he = exposure.equalize_hist(img)
        if(cmap != 'gray'):
            img_he = plt.get_cmap(cmap)(img_he)[:, :, :3]
        new_path = selectedFolder / "HistogramEqualization" / path.name
        new_path.parent.mkdir(parents=True, exist_ok=True)
        if(original_dtype == np.uint8):
            img_he = util.img_as_ubyte(img_he)
        elif(original_dtype == np.uint16):
            img_he = util.img_as_uint(img_he)
        io.imsave(new_path, img_he)
    
    elif(compute == "Adaptive HE"):
        img_ahe = exposure.equalize_adapthist(img)
        if(cmap != 'gray'):
            img_ahe = plt.get_cmap(cmap)(img_ahe)[:, :, :3]
        new_path = selectedFolder / "AdaptiveHistogramEqualization" / path.name
        new_path.parent.mkdir(parents=True, exist_ok=True)
        if(original_dtype == np.uint8):
            img_ahe = util.img_as_ubyte(img_ahe)
        elif(original_dtype == np.uint16):
            img_ahe = util.img_as_uint(img_ahe)
        io.imsave(new_path, img_ahe)
    
    return str(new_path)

def AutoCrop(image, threshold=0.9):
    mask = image > threshold
    coords = np.argwhere(~mask)
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1
    cropped = image[x0:x1, y0:y1]
    return cropped

def Start():
    def On_CheckBox_Crop():
        if(file := listBox_Files.value):
            GeneratePreviews(file)
    
    def On_Slider_Crop():
        if(checkBox_Crop.value):
            if(file := listBox_Files.value):
                # logging.info(f"crop: {slider_Crop.value}")
                GeneratePreviews(file)

    def On_Slider_AC_Cutoff():
        ComputeAC(original_img, combo_Colormap.value, slider_AC_Cutoff.value)
        PlotHistogram(picture_AC.image)

    def On_MouseEnter_Preview(event_data):
        PlotHistogram(event_data.widget.image)
    
    def On_DoubleClick_Preview(event):
        if (listBox_Files.items):
            img = io.imread(listBox_Files.value, as_gray=True)
            img = util.img_as_float64(img)
            name = event.widget.name
            if(name == "ac"):
                cutoff = slider_AC_Cutoff.value
                p_lower, p_upper = np.percentile(img, (0+cutoff, 100-cutoff))
                img = exposure.rescale_intensity(img, in_range=(p_lower, p_upper))
            elif(name == "he"):
                img = exposure.equalize_hist(img)
            elif(name == "ahe"):
                img = exposure.equalize_adapthist(img)
            plt.imshow(img, cmap=combo_Colormap.value, vmin=0)
            plt.show()

    def On_ListBoxChanged(file):
        if (listBox_Files.items):
            GeneratePreviews(file)

    def On_ComboColormapChanged():
        if (listBox_Files.items and listBox_Files.value):
            GeneratePreviews(listBox_Files.value)

    def On_OpenFolder():
        selectedFolder = filedialog.askdirectory(title="Choose Image Folder", mustexist=True, initialdir=str(Path.home()))
        if (selectedFolder):
            logging.info(f"Selected folder: {selectedFolder}")
            image_files = LoadImages(selectedFolder)
            UpdateListBox(image_files)
            UpdateProgBar(0)
        else:
            logging.info(f"Folder selection canceled")
    
    def On_BatchCompute():
        if(listBox_Files.items):
            selectedFolder = filedialog.askdirectory(title="Choose Output Folder", mustexist=True, initialdir=str(Path.home()))
            if (selectedFolder):
                files = listBox_Files.items
                tot = len(listBox_Files.items)
                compute = combo_Compute.value
                cmap = combo_Colormap.value
                cutoff = slider_AC_Cutoff.value
                autoCrop = checkBox_Crop.value
                # ct = 0.9
                background_thread = threading.Thread(target=Run_BatchCompute, args=(selectedFolder, files, tot, compute, cmap, cutoff, autoCrop))
                background_thread.start()
    
    @TimeIt
    def Run_BatchCompute(selectedFolder, files, tot, compute, cmap, cutoff, autoCrop):
        partial_BatchCompute = partial(BatchCompute, selectedFolder=selectedFolder, compute=compute, cmap=cmap, cutoff=cutoff, autoCrop=autoCrop)
        with ThreadPoolExecutor(int(os.cpu_count()/2.5)) as executor:
            futures = [executor.submit(partial_BatchCompute, file) for file in files]
            for i, f in enumerate(as_completed(futures), 1):
                res = f.result()
                app.after(0, Update_BatchCompute, [res, i, tot])
    
    def Update_BatchCompute(res, i, tot):
        msg = f"{i}/{tot} computed"
        text_Status.value = msg
        logging.info(f"{i}/{tot} - {res}")
        percent = int((i/tot)*100)
        UpdateProgBar(percent)

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

    def UpdateProgBar(percent):
        prog_bar["value"] = percent

    def GeneratePreviews(image_File):
        global original_img
        global prev_height
        img = util.img_as_float64(io.imread(image_File, as_gray=True))
        if(checkBox_Crop.value):
            img = AutoCrop(img)
        h, w = img.shape[:2]
        prev_height = int(THUMBNAIL_SIZE / (w / h))
        original_img = transform.resize(img, (prev_height, THUMBNAIL_SIZE), anti_aliasing=True)

        if(combo_Colormap.value != 'gray'):
            original_prev = plt.get_cmap(combo_Colormap.value)(original_img)[:, :, :3]
        else:
            original_prev = original_img.copy()
        picture_Original.height = prev_height
        original_prev = Image.fromarray(util.img_as_ubyte(original_prev))
        picture_Original.image = original_prev

        if(not slider_AC_Cutoff.enabled):
            slider_AC_Cutoff.enable()
            slider_AC_Cutoff.value = 2

        app.after(1, ComputeAC, [original_img, combo_Colormap.value, slider_AC_Cutoff.value])
        app.after(2, ComputeHE, [original_img, combo_Colormap.value])
        app.after(3, ComputeAHE, [original_img, combo_Colormap.value])
        app.after(4, PlotHistogram, [picture_Original.image])

    def ComputeAC(image, cmap, cutoff=2):
        p_lower, p_upper = np.percentile(image, (0+cutoff, 100-cutoff))
        img_ac = exposure.rescale_intensity(image, in_range=(p_lower, p_upper))
        if(cmap != 'gray'):
            img_ac = plt.get_cmap(cmap)(img_ac)[:, :, :3]
        img_ac = Image.fromarray(util.img_as_ubyte(img_ac))
        picture_AC.height = prev_height
        picture_AC.image = img_ac

    def ComputeHE(image, cmap):
        img_he = exposure.equalize_hist(image)
        if(cmap != 'gray'):
            img_he = plt.get_cmap(cmap)(img_he)[:, :, :3]
        img_he = Image.fromarray(util.img_as_ubyte(img_he))
        picture_HE.height = prev_height
        picture_HE.image = img_he

    def ComputeAHE(image, cmap):
        img_ahe = exposure.equalize_adapthist(image)
        if(cmap != 'gray'):
            img_ahe = plt.get_cmap(cmap)(img_ahe)[:, :, :3]
        img_ahe = Image.fromarray(util.img_as_ubyte(img_ahe))
        picture_AHE.height = prev_height
        picture_AHE.image = img_ahe
    
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


    app = App(title="RGView", layout="auto", width=1600, height=900)
    app.tk.config(padx=5, pady=5)

    box_Toolbar = Box(app, align="top", width="fill")
    box_Toolbar.tk.config(padx=5, pady=5)
    button_OpenFolder = PushButton(box_Toolbar, text="Open Folder", command=On_OpenFolder, align="left")
    combo_Compute = Combo(box_Toolbar, align="left", options=["Auto Contrast", "Histogram Equalization (HE)", "Adaptive HE"])
    button_BatchCompute = PushButton(box_Toolbar, text="Batch Compute", align="left", command=On_BatchCompute)

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
    text_Colormap.visible = False
    combo_Colormap = Combo(box_Colormap, ['gray', 'viridis', 'plasma', 'inferno', 'magma', 'cividis'], align='left', command=On_ComboColormapChanged)
    combo_Colormap.visible = False

    box_original = Box(titleBox_Previews, align="left", height="fill")
    titleBox_Original = TitleBox(box_original, "Original", align="top")
    titleBox_Original.tk.config(padx=5, pady=5)
    picture_Original = Picture(titleBox_Original, align="top", image=dummy_image)
    picture_Original.name = "original"
    box_Crop = Box(titleBox_Original, align="bottom", width="fill")
    checkBox_Crop = CheckBox(box_Crop, text="Auto Crop", command=On_CheckBox_Crop, align="left")
    checkBox_Crop.value = 1
    # slider_Crop = Slider(box_Crop, 1, 99, align="left", width="fill", command=On_Slider_Crop)
    picture_Original.when_mouse_enters = On_MouseEnter_Preview
    picture_Original.when_double_clicked = On_DoubleClick_Preview

    # Box(titleBox_Previews, align="left", width="fill")

    box_AC = Box(titleBox_Previews, align="left", height="fill")
    titleBox_AC = TitleBox(box_AC, "Auto Contrast", align="top")
    titleBox_AC.tk.config(padx=5, pady=5)
    picture_AC = Picture(titleBox_AC, align="top", image=dummy_image)
    picture_AC.name = "ac"
    slider_AC_Cutoff = Slider(titleBox_AC, enabled=False, command=On_Slider_AC_Cutoff, start=0, end=49, align="bottom", width="fill")
    picture_AC.when_mouse_enters = On_MouseEnter_Preview
    picture_AC.when_double_clicked = On_DoubleClick_Preview

    # Box(titleBox_Previews, align="left", width="fill")

    box_HE = Box(titleBox_Previews, align="left", height="fill")
    titleBox_HE = TitleBox(box_HE, "Histogram Equalization", align="top")
    titleBox_HE.tk.config(padx=5, pady=5)
    picture_HE = Picture(titleBox_HE, align="left", image=dummy_image)
    picture_HE.name = "he"
    picture_HE.when_mouse_enters = On_MouseEnter_Preview
    picture_HE.when_double_clicked = On_DoubleClick_Preview

    # Box(titleBox_Previews, align="left", width="fill")

    box_AHE = Box(titleBox_Previews, align="left", height="fill")
    titleBox_AHE = TitleBox(box_AHE, "Adaptive Histogram Equalization", align="top")
    titleBox_AHE.tk.config(padx=5, pady=5)
    picture_AHE = Picture(titleBox_AHE, align="left", image=dummy_image)
    picture_AHE.name = "ahe"
    picture_AHE.when_mouse_enters = On_MouseEnter_Preview
    picture_AHE.when_double_clicked = On_DoubleClick_Preview

    # Box(titleBox_Previews, align="left", width="fill")

    box_Hist = Box(titleBox_Previews, align="left", height="fill")
    titleBox_Hist = TitleBox(box_Hist, "Histogram", align="top")
    titleBox_Hist.tk.config(padx=5, pady=5)
    picture_Hist = Picture(titleBox_Hist, image=dummy_image)

    box_Status = Box(app, width="fill", layout="grid")
    box_Status.tk.config(padx=5, pady=5)
    box_Status.tk.columnconfigure(0, weight=1)
    text_Status = Text(box_Status, "Welcome to RGView",
                       align="left", grid=[0, 0])
    prog_bar = ttk.Progressbar(box_Status.tk, mode="determinate", orient="horizontal")
    prog_bar.grid(row=1, column=0, sticky="ew")
    # box_Status.add_tk_widget(prog_bar, grid=[0,1], width="fill")

    app.display()


if (__name__ == "__main__"):
    Start()
