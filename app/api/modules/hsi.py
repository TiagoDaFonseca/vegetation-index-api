import spectral.io.envi as envi
from spectral import get_rgb
from datetime import datetime
import os
import numpy as np
from tqdm import tqdm
import warnings
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger("hsi files module")
logging.basicConfig(level=logging.DEBUG)


# -- Files Management  -- #
def get_image_files(all_files: list) -> list:
    image_files = []
    for file in all_files:
        if '.' not in file:
            continue
        if file.split('.')[1] == 'img' or file.split('.')[1] == 'cue':
            image_files.append(file)
    return image_files


def load_image(path: str, filename: str):
    name = filename.split('.')[0]
    image = envi.open(file=os.path.join(path, f"{name}.hdr"), image=os.path.join(path, filename))
    metadata = image.metadata
    return image.load(), metadata


def set_mode(data: np.array, mode: str) -> np.array:
    image = None
    m = mode.upper()
    if m == 'RGB':
        image = get_bands(data, 76, 51, 31)
    elif m == 'CIR':
        image = get_bands(data, 128, 76, 51)
    elif m == 'INV':
        image = get_bands(data, 148, 11, 123)
    return image


def get_bands(data, a, b, c):
    R = np.array(data[:, :, a])
    G = np.array(data[:, :, b])
    B = np.array(data[:, :, c])
    rgb = np.concatenate((R, G, B), axis=2)
    return get_rgb(rgb)


def get_reference(ref_dir: str, datalist: list, average=False) -> np.array:
    data_list = []
    for file in datalist:
        name = file.split('.')[0]
        img = envi.open(os.path.join(ref_dir, file), image=os.path.join(ref_dir, f"{name}.cue"))
        data = img.load()
        data_list.append(data)
    if average:
        # averaging
        average = np.zeros_like(data_list[0])
        for band in range(average.shape[2]):
            for ref in range(len(data_list)):
                if ref == 0:
                    average[:, :, band] = data_list[ref][:, :, band]
                else:
                    to_average = np.concatenate((average[:, :, band], data_list[ref][:, :, band]), axis=2)
                    s = np.sum(to_average, axis=2).reshape(average.shape[0], average.shape[1], 1)
                    average[:, :, band] = s
            average[:, :, band] /= len(data_list)
        return average
    return data_list[0]


def get_datetime(data: str) -> datetime:
    items = data.split(',')
    # Get data
    date = items[0].split(' ')[1]
    time = items[1].split(' ')[1]
    # time = time[:len(time)-3]
    s = date + ' ' + time
    dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f%Z")
    return dt


def get_exposure(data: str) -> int:
    items = data.split(',')
    # Get data
    exposure = items[3].split('. ')[1].split(' ms')[0]
    return int(exposure)


def get_wavelengths(info: str) -> list:
    items = info.split(',')
    for item in items:
        if "wavelengths" in item:
            return list(item.split(': ')[1])


def convert_cue_to_img(some_dir: str):
    src_dir = os.path.join(some_dir, "ref")
    dst_dir = os.path.join(some_dir, "envi")
    main_files = os.listdir(src_dir)
    main_files.sort()
    main_files = [hdr for hdr in main_files if ".hdr" in hdr and "._" not in hdr]
    for filename in tqdm(main_files, desc="Converting files to BIP"):
        name = filename.split('.')[0]

        img = envi.open(os.path.join(src_dir, filename), image=os.path.join(src_dir, f"{name}.cue"))

        # Filter metadata
        meta = {
            "timestamp": get_datetime(img.metadata["description"]),
            "exposure": get_exposure(img.metadata["description"]),
            "wavelength": [int(item) for item in img.metadata["wavelength"]],
            "units": img.metadata["wavelength units"]
        }
        if os.path.isdir(dst_dir):
            try:
                envi.save_image(os.path.join(dst_dir, f"{name}.hdr"), img.load(), metadata=meta, force=True)
            except Exception as e:
                print(str(e))
        else:
            os.mkdir(dst_dir)
            try:
                envi.save_image(os.path.join(dst_dir, f"{name}.hdr"), img.load(), metadata=meta)
            except Exception as e:
                logger.debug(str(e))

    print("Conversion in complete.")


def create_reflectance_files(src_dir):
    main_dir = os.path.join(src_dir, "export")
    ref_dir = os.path.join(src_dir, "calib")
    dst_dir = os.path.join(src_dir, "ref")

    refs = os.listdir(ref_dir)
    refs.sort()
    refs = [ref for ref in refs if ".hdr" in ref]

    darks = [ref for ref in refs if "dark" in ref and "._dark" not in ref]
    whites = [ref for ref in refs if "white" in ref and "._white" not in ref]

    dark = get_reference(ref_dir, darks, average=False)
    print(f"{dark.shape=}")
    white = get_reference(ref_dir, whites, average=False)
    print(f"{white.shape=}")

    main_files = os.listdir(main_dir)
    main_files.sort()
    main_files = [hdr for hdr in main_files if ".hdr" in hdr and "._" not in hdr]
    # print(main_files)
    print(f"{len(main_files)} files found.")

    for filename in tqdm(main_files, desc="Converting files to BIP"):
        name = filename.split('.')[0]

        img = envi.open(os.path.join(main_dir, filename), image=os.path.join(main_dir, f"{name}.cue"))

        # Filter metadata
        meta = {
            "timestamp": get_datetime(img.metadata["description"]),
            "exposure": get_exposure(img.metadata["description"]),
            "wavelength": [int(item) for item in img.metadata["wavelength"]],
            "units": img.metadata["wavelength units"]
        }

        # Normalization
        numerator = img.load() - dark.load()
        denominator = white.load() - dark.load()
        with np.errstate(divide='ignore', invalid='ignore'):  # avoid singularities 1/0 and 0/0
            ref = np.true_divide(numerator, denominator + 0.000001)
            ref[ref == np.inf] = 0
            ref = np.nan_to_num(ref)
        ref = np.clip(ref, 0, 2.0)

        # Save in bip format
        new_name = name.split('RAW')[0]
        if os.path.isdir(dst_dir):
            try:
                envi.save_image(os.path.join(dst_dir, f"{new_name}_REF.hdr"), ref, metadata=meta, force=True)
            except Exception as e:
                logger.debug(str(e))
        else:
            os.mkdir(dst_dir)
            try:
                envi.save_image(os.path.join(dst_dir, f"{new_name}_REF.hdr"), ref, metadata=meta)
            except Exception as e:
                logger.debug(str(e))

    logger.debug("Conversion in complete.")
