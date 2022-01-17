import spectral.io.envi as envi
from datetime import datetime
import os
import numpy as np
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


def get_ref(refdir: str, datalist: list, average=False) -> np.array:
    data_list = []
    for file in datalist:
        name = file.split('.')[0]
        img = envi.open(os.path.join(refdir, file), image=os.path.join(refdir, f"{name}.cue"))
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


def get_exposure(data: str) -> str:
    items = data.split(',')
    # Get data
    exposure = items[3].split('. ')[1].split(' ms')[0]
    return int(exposure)


def get_wavelengths(info: str) -> list:
    items = info.split(',')
    for item in items:
        if "wavelengths" in item:
            return list(item.split(': ')[1])


def create_reflectance_files(src_dir):
    main_dir = os.path.join(src_dir, "export")
    ref_dir = os.path.join(src_dir, "calib")
    dst_dir = os.path.join(src_dir, "ref")

    refs = os.listdir(ref_dir)
    refs.sort()
    refs = [ref for ref in refs if ".hdr" in ref]

    darks = [ref for ref in refs if "dark" in ref and "._dark" not in ref]
    whites = [ref for ref in refs if "white" in ref and "._white" not in ref]

    dark = get_ref(ref_dir, darks, average=False)
    print(f"{dark.shape=}")
    white = get_ref(ref_dir, whites, average=False)
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
                print(str(e))
        else:
            os.mkdir(dst_dir)
            try:
                envi.save_image(os.path.join(dst_dir, f"{new_name}_REF.hdr"), ref, metadata=meta)
            except Exception as e:
                print(str(e))

    print("Conversion in complete.")


def convert_cue_to_img(dir):
    src_dir = os.path.join(dir, "ref")
    dst_dir = os.path.join(dir, "envi")
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
                print(str(e))

    print("Conversion in complete.")


if __name__ == "__main__":
    cwd = "/Volumes/Extreme SSD/Work/hsi_data/2021_11_12/40m"
    #convert_to_img(dir=cwd)
    create_reflectance_files(src_dir=cwd)
