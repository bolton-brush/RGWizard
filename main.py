# No need for freeze_support if using threading with pyinstaller
# from multiprocessing import freeze_support
import sys
import time
import logging
import threading
from pathlib import Path
from functools import partial
from tkinter import filedialog, ttk
from guizero import App, PushButton, Text, Box, ListBox, TitleBox, Picture, Slider, Combo, CheckBox
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from skimage import io, exposure, util, color, transform
from PIL import Image, ImageDraw
from concurrent.futures import ThreadPoolExecutor, as_completed
from model_manager import ModelManager
from orientation import apply_fliprot_label, prepare_fliprot_prediction_image

logging.basicConfig(level=logging.INFO)
THUMBNAIL_SIZE = 300
dummy_image = Image.new("L", (THUMBNAIL_SIZE, THUMBNAIL_SIZE), "white")
original_img = None

ENHANCEMENT_OUTPUT_FOLDERS = {
    "Auto Contrast": "AutoContrast",
    "Histogram Equalization (HE)": "HistogramEqualization",
    "Adaptive HE": "AdaptiveHistogramEqualization",
}

def stop_watch(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"{func.__name__} executed in {end_time - start_time} seconds")
        return result
    return wrapper

def prepare_processing_image(image):
    if(image.dtype == np.uint16 or image.ndim == 2):
        return util.img_as_float64(image)
    return color.rgb2gray(image[:, :, :3])

def apply_enhancement(image, compute, cutoff=2):
    if(compute == "Auto Contrast"):
        p_lower, p_upper = np.percentile(image, (0+cutoff, 100-cutoff))
        return exposure.rescale_intensity(image, in_range=(p_lower, p_upper))
    if(compute == "Histogram Equalization (HE)"):
        return exposure.equalize_hist(image)
    if(compute == "Adaptive HE"):
        return exposure.equalize_adapthist(image)
    raise ValueError(f"Unknown enhancement: {compute}")

def apply_colormap(image, cmap):
    if(cmap == 'gray'):
        return image
    return plt.get_cmap(cmap)(image)[:, :, :3]

def image_to_pil(image, cmap='gray'):
    return Image.fromarray(util.img_as_ubyte(apply_colormap(image, cmap)))

def restore_original_dtype(image, original_dtype):
    if(original_dtype == np.uint8):
        return util.img_as_ubyte(image)
    if(original_dtype == np.uint16):
        return util.img_as_uint(image)
    return image

def preview_height_for(image):
    h, w = image.shape[:2]
    return int(THUMBNAIL_SIZE / (w / h))

def get_fliprot_model():
    manager = globals().get('model_manager')
    if manager is None:
        return None
    return manager.get_model(target="CEPH", task="fliprot")

def compute_image(file, selectedFolder, compute, cmap, cutoff, autoCrop, fliprot):
    img_raw = io.imread(file)
    original_dtype = img_raw.dtype
    img = prepare_processing_image(img_raw)
    if(autoCrop):
        img = autocrop_image(img)
    if(fliprot):
        _, img = autofliprot_image(util.img_as_ubyte(img))
    path = Path(file)
    selectedFolder = Path(selectedFolder)

    img_out = apply_enhancement(img, compute, cutoff)
    img_out = apply_colormap(img_out, cmap)
    img_out = restore_original_dtype(img_out, original_dtype)
    new_path = selectedFolder / ENHANCEMENT_OUTPUT_FOLDERS[compute] / path.name
    new_path.parent.mkdir(parents=True, exist_ok=True)
    io.imsave(new_path, img_out)
    
    return str(new_path)

def autocrop_image(image, threshold=0.85):
    if image.ndim == 3:
        mask = np.all(image > threshold, axis=-1)
    else:
        mask = image > threshold
    coords = np.argwhere(~mask)
    if coords.size == 0:
        return image
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1
    cropped = image[x0:x1, y0:y1, ...]
    return cropped

def autofliprot_image(image):
    model = get_fliprot_model()
    if model is None:
        logging.warning("No cephalometric fliprot model loaded; skipping auto rotation/flipping")
        return (False, image)

    prediction_image = prepare_fliprot_prediction_image(image)
    pred_class, _, _ = model['model'].predict(prediction_image)
    label = str(pred_class)
    try:
        return apply_fliprot_label(image, label)
    except ValueError:
        logging.warning("Unexpected fliprot prediction label: %s", label)
        return (False, image)

def main():
    
    def On_CheckBox_Crop():
        if(file := listBox_Files.value):
            generate_previews(file)

    def On_Slider_AC_Cutoff():
        compute_autocontrast(original_img, combo_Colormap.value, slider_AC_Cutoff.value)
        plot_histogram(picture_AC.image)

    def On_MouseEnter_Preview(event_data):
        plot_histogram(event_data.widget.image)
    
    def On_DoubleClick_Preview(event):
        if (listBox_Files.items):
            img = io.imread(listBox_Files.value, as_gray=True)
            img = util.img_as_float64(img)
            if(checkBox_Crop.value):
                img = autocrop_image(img)
            if(checkBox_Fliprot.value):
                _, img = autofliprot_image(util.img_as_ubyte(img))
            name = event.widget.name
            if(name == "ac"):
                img = apply_enhancement(img, "Auto Contrast", slider_AC_Cutoff.value)
            elif(name == "he"):
                img = apply_enhancement(img, "Histogram Equalization (HE)")
            elif(name == "ahe"):
                img = apply_enhancement(img, "Adaptive HE")
            plt.imshow(img, cmap=combo_Colormap.value, vmin=0)
            plt.show()

    def On_ListBoxChanged(file):
        if (listBox_Files.items):
            generate_previews(file)

    def On_ComboColormapChanged():
        if (listBox_Files.items and listBox_Files.value):
            generate_previews(listBox_Files.value)

    def On_OpenFolder():
        selectedFolder = filedialog.askdirectory(title="Choose Source Folder", mustexist=True, initialdir=str(Path.home()))
        if (selectedFolder):
            logging.info(f"Source folder: {selectedFolder}")
            image_files = load_images(selectedFolder)
            update_listbox(image_files)
            update_progressbar(0)
        else:
            logging.info(f"Source folder selection canceled")
    
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
                fliprot = checkBox_Fliprot.value
                background_thread = threading.Thread(target=run_batch_compute, args=(selectedFolder, files, tot, compute, cmap, cutoff, autoCrop, fliprot))
                background_thread.start()
            else:
                logging.info(f"Target folder selection canceled")
    
    @stop_watch
    def run_batch_compute(selectedFolder, files, tot, compute, cmap, cutoff, autoCrop, fliprot):
        partial_BatchCompute = partial(compute_image, selectedFolder=selectedFolder, compute=compute, cmap=cmap, cutoff=cutoff, autoCrop=autoCrop, fliprot=fliprot)
        with ThreadPoolExecutor(2) as executor:
            futures = {executor.submit(partial_BatchCompute, file): file for file in files}
            for i, future in enumerate(as_completed(futures), 1):
                ok = True
                try:
                    res = future.result()
                except Exception as exc:
                    ok = False
                    res = f"{futures[future]}: {exc}"
                    logging.exception(f"Failed to compute {futures[future]}")
                app.after(0, update_batch_compute, [res, i, tot, ok])
    
    def update_batch_compute(res, i, tot, ok=True):
        msg = f"{i}/{tot} computed" if ok else f"{i}/{tot} failed"
        text_Status.value = msg
        logging.info(f"{i}/{tot} - {res}")
        percent = int((i/tot)*100)
        update_progressbar(percent)

    @stop_watch
    def load_images(folder_path):
        image_patterns = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tif', '*.tiff', '*.webp']
        image_files = []
        folder_path = Path(folder_path)

        if folder_path.exists() and folder_path.is_dir():
            for pattern in image_patterns:
                image_files.extend(folder_path.rglob(pattern))
        return image_files

    @stop_watch
    def update_listbox(image_files):
        listBox_Files.clear()
        listBox_Files.children[0].tk.insert("end", *image_files)
        msg = f"{len(image_files)} files loaded"
        text_Status.value = msg
        logging.info(msg)

    def update_progressbar(percent):
        prog_bar["value"] = percent

    def generate_previews(image_File):
        global original_img
        global prev_height
        img = util.img_as_float64(io.imread(image_File, as_gray=True))
        prev_height = preview_height_for(img)
        original_img = transform.resize(img, (prev_height, THUMBNAIL_SIZE), anti_aliasing=True)

        picture_Original.height = prev_height
        picture_Original.image = image_to_pil(original_img, combo_Colormap.value)

        if(checkBox_Crop.value):
            img = autocrop_image(img)
            prev_height = preview_height_for(img)
            original_img = transform.resize(img, (prev_height, THUMBNAIL_SIZE), anti_aliasing=True)
        
        if(checkBox_Fliprot.value):
            fliprotted, original_img = autofliprot_image(util.img_as_ubyte(original_img))
            if(fliprotted):
                prev_height = preview_height_for(original_img)

        if(not slider_AC_Cutoff.enabled):
            slider_AC_Cutoff.enable()
            slider_AC_Cutoff.value = 2

        app.after(1, compute_autocontrast, [original_img, combo_Colormap.value, slider_AC_Cutoff.value])
        app.after(2, compute_histogram_equalization, [original_img, combo_Colormap.value])
        app.after(3, compute_clahe, [original_img, combo_Colormap.value])
        app.after(4, plot_histogram, [picture_Original.image])

    def compute_autocontrast(image, cmap, cutoff=2):
        img_ac = image_to_pil(apply_enhancement(image, "Auto Contrast", cutoff), cmap)
        picture_AC.height = prev_height
        picture_AC.image = img_ac

    def compute_histogram_equalization(image, cmap):
        img_he = image_to_pil(apply_enhancement(image, "Histogram Equalization (HE)"), cmap)
        picture_HE.height = prev_height
        picture_HE.image = img_he

    def compute_clahe(image, cmap):
        img_ahe = image_to_pil(apply_enhancement(image, "Adaptive HE"), cmap)
        picture_AHE.height = prev_height
        picture_AHE.image = img_ahe
    
    def plot_histogram(image, image_size=(THUMBNAIL_SIZE, THUMBNAIL_SIZE), background_color='white'):
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


    app = App(title="RGWizard", layout="auto", width=1600, height=900)
    app.tk.config(padx=5, pady=5)

    box_Toolbar_1 = Box(app, align="top", width="fill")
    box_Toolbar_1.tk.config(padx=5, pady=5)
    button_OpenFolder = PushButton(box_Toolbar_1, text="Open Folder", command=On_OpenFolder, align="left")
    button_BatchCompute = PushButton(box_Toolbar_1, text="Batch Compute", align="left", command=On_BatchCompute)
    combo_Compute = Combo(box_Toolbar_1, align="left", options=["Adaptive HE","Histogram Equalization (HE)","Auto Contrast"])
    box_Toolbar_2 = Box(app, align="top", width="fill")
    box_Toolbar_2.tk.config(padx=5, pady=5)
    box_Colormap = Box(box_Toolbar_2, align='left')
    box_Colormap.tk.config(padx=5, pady=5)
    text_Colormap = Text(box_Colormap, "Colormap: ", size=10, align="left")
    # text_Colormap.visible = False
    combo_Colormap = Combo(box_Colormap, ['gray', 'viridis', 'plasma', 'inferno', 'magma', 'cividis'], align='left', command=On_ComboColormapChanged)
    # combo_Colormap.visible = False
    checkBox_Crop = CheckBox(box_Toolbar_2, text="Auto Crop", command=On_CheckBox_Crop, align="left")
    checkBox_Crop.value = 1
    checkBox_Fliprot = CheckBox(box_Toolbar_2, text="Auto Rotation/Flipping of Cephs", command=On_CheckBox_Crop, align="left")
    if get_fliprot_model() is None:
        checkBox_Fliprot.value = 0
        checkBox_Fliprot.disable()
    else:
        checkBox_Fliprot.value = 1

    box_Main = Box(app, border=0, width="fill", height="fill")
    box_Main.tk.config(padx=5, pady=5)
    

    titleBox_Files = TitleBox(box_Main, "Image Files", width="fill", height="fill")
    titleBox_Files.tk.config(padx=5, pady=5)
    listBox_Files = ListBox(titleBox_Files, scrollbar=True, width="fill", height="fill", command=On_ListBoxChanged)
    titleBox_Previews = TitleBox(box_Main, "Previewes", width="fill")
    titleBox_Previews.tk.config(padx=5, pady=5)


    box_original = Box(titleBox_Previews, align="left", height="fill")
    titleBox_Original = TitleBox(box_original, "Original", align="top")
    titleBox_Original.tk.config(padx=5, pady=5)
    picture_Original = Picture(titleBox_Original, align="top", image=dummy_image)
    picture_Original.name = "original"
    
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
    text_Status = Text(box_Status, "Welcome to RGWizard",
                       align="left", grid=[0, 0])
    prog_bar = ttk.Progressbar(box_Status.tk, mode="determinate", orient="horizontal")
    prog_bar.grid(row=1, column=0, sticky="ew")

    app.display()


if (__name__ == "__main__"):
    # No need for freeze_support if using threading only with pyinstaller
    # freeze_support()
    smoke_test = "--smoke-test" in sys.argv
    try:
        model_manager = ModelManager("models")
        logging.info(f"loaded {model_manager.get_loaded_models_qt()} model(s)")
        logging.info(model_manager.get_models_info())
        if smoke_test and get_fliprot_model() is None:
            raise RuntimeError("Cephalometric fliprot model was not bundled")
        if smoke_test:
            sys.exit(0)
        main()
    except Exception:
        logging.exception("RGWizard startup failed")
        raise
        
