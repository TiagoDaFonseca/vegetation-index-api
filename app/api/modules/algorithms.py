import numpy as np
from joblib import load
import logging
from time import time
from api.modules.hsi import *
import cv2
import psutil
from multiprocessing import Pool


logger = logging.getLogger("Algorithms module")
logging.basicConfig(level=logging.DEBUG)

NUM_CPUS = psutil.cpu_count(logical=False)

VEGETATION_INDEXES = ['VARI',
                      'GLI',
                      'NGRDI',
                      'NGBDI',
                      'NDVI',
                      'SR',
                      'EVI',
                      'SG',
                      'NDVI705',
                      'MSR705',
                      'MNDVI705',
                      'VOG1',
                      'VOG2',
                      'VOG3',
                      'REP',
                      'PRI',
                      'SIPI',
                      'RGR',
                      'PSRI',
                      'CRI1',
                      'CRI2',
                      'ARI1',
                      'ARI2',
                      'WBI',
                      'VVI']  # the last index is custom


class VIndexer:
    def __init__(self, image: np.array):
        self.center = {'VARI': -0.2,    # ~-0.4
                       'GLI': 0.5,      # ~
                       'NGRDI': 0.5,
                       'NGBDI': 0.4,    # ~0.6
                       'NDVI': 0.6,     # ~0.7
                       'SR': 3,         # >4
                       'EVI': 0.5,
                       'SG': 0.5,
                       'NDVI705': 0,
                       'MSR705': 5,
                       'MNDVI705': 0,
                       'VOG1': 5,
                       'VOG2': 5,
                       'VOG3': 5,
                       'REP': 720,      # 721 nm
                       'PRI': 0,
                       'SIPI': 1,
                       'RGR': 2,
                       'PSRI': 0,
                       'CRI1': 10,
                       'CRI2': 10,
                       'ARI1': 0.1,     # 0.001-0.2
                       'ARI2': 0.1,     # 0.001-0.2
                       'WBI': 2,        # 0.8-1.2
                       'VVI': 8}
        self.index_minmax = {'VARI': (-1, 1),  # ~-0.4
                             'GLI': (-1, 1),  # ~
                             'NGRDI': (-1, 1),
                             'NGBDI': (-1, 1),
                             'NDVI': (-1, 1),  # ~0.7
                             'SR': (0, 7),  # >4
                             'EVI': (-1, 1),
                             'SG': (0, 1),
                             'NDVI705': (-1, 1),
                             'MSR705': (0, 10),
                             'MNDVI705': (-1, 1),
                             'VOG1': (0, 10),
                             'VOG2': (0, 10),
                             'VOG3': (0, 10),
                             'REP': (715, 725),  # 721 nm
                             'PRI': (-1, 1),
                             'SIPI': (0, 2),
                             'RGR': (0, 5),
                             'PSRI': (-1, 1),
                             'CRI1': (0, 20),
                             'CRI2': (0, 20),
                             'ARI1': (0, 0.3),  # 0.001-0.2
                             'ARI2': (-0.3, 0.3),  # 0.001-0.2
                             'WBI': (0, 5),  # 0.8-1.2
                             'VVI': (0, 16)}  # >10
        logger.debug("Reading Image bands")
        initial = time()
        self.img = image
        self.Red = np.average(self.img[:, :, 67:100], axis=2).reshape(self.img.shape[0],
                                                                      self.img.shape[1],
                                                                      1)  # 620 - 750 nm
        self.Green = np.average(self.img[:, :, 36:55], axis=2).reshape(self.img.shape[0],
                                                                       self.img.shape[1],
                                                                       1)  # 495 - 570 nm
        self.Blue = np.average(self.img[:, :, 24:36], axis=2).reshape(self.img.shape[0],
                                                                      self.img.shape[1],
                                                                      1)  # 450 - 495 nm
        self.NIR = np.average(self.img[:, :, 99:], axis=2).reshape(self.img.shape[0],
                                                                   self.img.shape[1],
                                                                   1)  # 750 - 1000 nm
        self.b445 = np.average(self.img[:, :, 22:24], axis=2).reshape(self.img.shape[0], self.img.shape[1], 1)
        self.b500 = np.average(self.img[:, :, 36:38], axis=2).reshape(self.img.shape[0], self.img.shape[1], 1)
        self.b510 = self.img[:, :, 39]
        self.b550 = self.img[:, :, 49]
        self.b531 = np.average(self.img[:, :, 44:46], axis=2).reshape(self.img.shape[0], self.img.shape[1], 1)
        self.b570 = self.img[:, :, 54]
        self.b680 = np.average(self.img[:, :, 81:83], axis=2).reshape(self.img.shape[0], self.img.shape[1], 1)
        self.b700 = np.average(self.img[:, :, 86:88], axis=2).reshape(self.img.shape[0], self.img.shape[1], 1)
        self.b705 = self.img[:, :, 88]
        self.b750 = self.img[:, :, 99]
        self.b740 = np.average(self.img[:, :, 96:98], axis=2).reshape(self.img.shape[0], self.img.shape[1], 1)
        self.b720 = np.average(self.img[:, :, 91:93], axis=2).reshape(self.img.shape[0], self.img.shape[1], 1)
        self.b715 = self.img[:, :, 90]
        self.b726 = self.img[:, :, 93]
        self.b734 = self.img[:, :, 95]
        self.b747 = self.img[:, :, 98]
        self.b800 = np.average(self.img[:, :, 111:113], axis=2).reshape(self.img.shape[0], self.img.shape[1], 1)
        self.b857 = self.img[:, :, 126]
        self.b900 = np.average(self.img[:, :, 136:138], axis=2).reshape(self.img.shape[0], self.img.shape[1], 1)
        self.b970 = self.img[:, :, 154]
        elapsed = time() - initial
        logger.debug(f"VIndexer created. Bands loaded in {elapsed} seconds.")

    # -- Visible Atmospheric Resistant Index -- #
    def vari(self):
        vari = np.divide((self.Green - self.Red), (self.Green + self.Red - self.Blue + 0.00001))
        return np.clip(vari, -1, 1)

    # -- Green Leaf Index -- #
    def gli(self):
        gli = np.divide((2 * self.Green - self.Red - self.Blue), (2 * self.Green + self.Red + self.Blue + 0.00001))
        return np.clip(gli, -1, 1)

    # -- Normalized Green Red Difference Index -- #
    def ngrdi(self):
        ngrdi = np.divide((self.Green - self.Red), (self.Green + self.Red + 0.00001))
        return np.clip(ngrdi, -1, 1)

    # -- Normalized Green Blue Difference Index -- #
    def ngbdi(self):
        ngbdi = np.divide((self.Green - self.Blue), (self.Green + self.Blue + 0.00001))
        return np.clip(ngbdi, -1, 1)

    # -- Normalized Difference Vegetation Index -- #
    def ndvi(self):
        ndvi = np.divide((self.NIR - self.Red), (self.NIR + self.Red + 0.00001))
        return np.clip(ndvi, -1, 1)

    # -- Simple Ratio Index -- #
    def simple_ratio(self):
        sr = np.divide(self.NIR, self.Red + 0.00001)
        return np.clip(sr, 0, 15)

    # -- Enhanced Vegetation Index -- #
    def evi(self):
        evi = 2.5 * np.divide((self.NIR - self.Red), (self.NIR + (6 * self.Red) - (7.5 * self.Blue) + 1))
        return np.clip(evi, -1, 1)

    # -- Sum Green Index -- #
    def sum_green(self):
        sg = self.Green
        return np.clip(sg, 0, 1)

    # -- Red Edge Normalized Difference Vegetation Index -- #
    def ndvi705(self):
        ndvi = np.divide((self.b750 - self.b705), (self.b750 + self.b705 + 0.00001))
        return np.clip(ndvi, -1, 1)

    # -- Modified Red Edge Simple Ratio Index -- #
    def msr705(self):
        mSR = np.divide((self.b750 - self.b445), (self.b705 - self.b445 + 0.00001))
        return np.clip(mSR, 0, 15)

    # -- Modified Red Edge Normalized Difference Vegetation Index -- #
    def mndvi705(self):
        mndvi = np.divide((self.b750 - self.b705), (self.b750 + self.b705 - (2 * self.b445) + 0.00001))
        return np.clip(mndvi, 0, 15)

    # -- Vogelmann Red Edge Index 1 -- #
    def vog1(self):
        vog = np.divide(self.b740, self.b720 + 0.00001)
        return np.clip(vog, 0, 10)

    # -- Vogelmann Red Edge Index 2 -- #
    def vog2(self):
        vog = np.divide((self.b734 - self.b747), (self.b715 + self.b726 + 0.00001))
        return np.clip(vog, -5, 10)

    # -- Vogelmann Red Edge Index 3 -- #
    def vog3(self):
        vog = np.divide((self.b734 - self.b747), (self.b715 + self.b720 + 0.00001))
        return np.clip(vog, -5, 10)

    # -- Red Edge Position Index -- #
    def rep(self):
        A = self.img[:, :, 80]
        D = np.average(self.img[:, :, 107:109], axis=2)
        D = D.reshape(D.shape[0], D.shape[1], 1)
        C = np.average(self.img[:, :, 97:99], axis=2)
        C = C.reshape(C.shape[0], C.shape[1], 1)
        B = np.average(self.img[:, :, 87:89], axis=2)
        B = B.reshape(B.shape[0], B.shape[1], 1)
        RedEdge = (A + D) / 2.0
        rep = 40 * (RedEdge - B) / (C - B) + 700
        return np.clip(rep, 400, 1000)

    # -- Photochemical Reflectance Index -- #
    def pri(self):
        pri = np.divide((self.b531 - self.b570), (self.b531 + self.b570 + 0.00001))
        return np.clip(pri, -1, 1)

    # -- Structure Insensitive Pigment Index -- #
    def sipi(self):
        sipi = np.divide((self.b800 - self.b445), (self.b800 - self.b680 + 0.00001))
        return np.clip(sipi, 0, 2)

    # -- Red Green Ratio Index -- #
    def rgr(self):
        rgr = np.divide(self.Red, self.Green)
        return np.clip(rgr, 0, 5)

    # -- Plant Senescence Reflectance Index -- #
    def psri(self):
        psri = np.divide((self.b680 - self.b500), self.b750 + 0.00001)
        return np.clip(psri, -1, 1)

    # -- Carotenoid Reflectance Index 1 -- #
    def cri1(self):
        cri = np.reciprocal(self.b510) - np.reciprocal(self.b550)
        return np.clip(cri, 0, 30)

    # -- Carotenoid Reflectance Index 2 -- #
    def cri2(self):
        cri = np.reciprocal(self.b510) - np.reciprocal(self.b700)
        return np.clip(cri, 0, 30)

    # -- Anthocyanin Reflectance Index 1-- #
    def ari1(self):
        ari = np.reciprocal(self.b550) - np.reciprocal(self.b700)
        return np.clip(ari, -0.3, 0.3)

    # -- Anthocyanin Reflectance Index 2-- #
    def ari2(self):
        ari = self.b800 * (np.reciprocal(self.b550) - np.reciprocal(self.b700))
        return np.clip(ari, -0.3, 0.3)

    # -- Water Band Index -- #
    def wbi(self):
        wbi = np.divide(self.b900, self.b970 + 0.00001)
        return np.clip(wbi, 0, 5)

    # -- Modified Normalized Difference Water Index -- #
    def mndwi(self):
        mndwi = np.divide((self.b857 - self.b970), (self.b857 + self.b970 + 0.00001))
        return np.clip(mndwi, -1, 1)

    # -- Vegetative vigor Index -- # Relaciona a reflexao no NIR com a absorcao no vermelho
    def vvi(self):
        A = np.average(self.img[:, :, 90:99], axis=2)  # 710-740 nm
        A = A.reshape(A.shape[0], A.shape[1], 1)
        B = np.average(self.img[:, :, 80:86], axis=2)  # 660-690 nm
        B = B.reshape(B.shape[0], B.shape[1], 1)
        vvi = np.divide(A, B + 0.00001)
        return np.clip(vvi, 0, 30)

    # Method to get index output
    def get_index(self, index_name: str) -> np.array:
        idx = index_name.upper()
        if idx == 'VARI':
            return self.vari()
        elif idx == 'GLI':
            return self.gli()
        elif idx == 'NGRDI':
            return self.ngrdi()
        elif idx == 'NGBDI':
            return self.ngbdi()
        elif idx == 'NDVI':
            return self.ndvi()
        elif idx == 'SR':
            return self.simple_ratio()
        elif idx == 'EVI':
            return self.evi()
        elif idx == 'SG':
            return self.sum_green()
        elif idx == 'NDVI705':
            return self.ndvi705()
        elif idx == 'MSR705':
            return self.msr705()
        elif idx == 'MNDVI705':
            return self.mndvi705()
        elif idx == 'VOG1':
            return self.vog1()
        elif idx == 'VOG2':
            return self.vog2()
        elif idx == 'VOG3':
            return self.vog3()
        elif idx == 'REP':
            return self.rep()
        elif idx == 'PRI':
            return self.pri()
        elif idx == 'SIPI':
            return self.sipi()
        elif idx == 'RGR':
            return self.rgr()
        elif idx == 'PSRI':
            return self.psri()
        elif idx == 'CRI1':
            return self.cri1()
        elif idx == 'CRI2':
            return self.cri2()
        elif idx == 'ARI1':
            return self.ari1()
        elif idx == 'ARI2':
            return self.ari2()
        elif idx == 'WBI':
            return self.wbi()
        elif idx == 'VVI':
            return self.vvi()
        else:
            return np.array([])


class Classifier:
    def __init__(self, filename: str):
        self.clf = load(filename)
        self.prediction = None

    def classify(self, data):
        # 1. Input
        ref = data.reshape(-1, data.shape[2])

        # 2. Pre-processing
        corr_ref = snv(ref)

        # 3. Predict
        # pred = []
        # with Pool(processes=4) as pool:
        #     pred = pool.map(self.clf.predict, [corr_ref[:42025, :],
        #                                        corr_ref[42025:84050, :],
        #                                        corr_ref[84050:126075, :],
        #                                        corr_ref[126075:, :],
        #                                        ])
        y_pred = self.clf.predict(corr_ref)
        # y_pred = np.concatenate((pred[0], pred[1], pred[2], pred[3]))
        # y_pred = y_pred.reshape(y_pred.shape[0], 1)


        # 4. Image output
        return y_pred.reshape(data.shape[0], data.shape[1])


#  -- Pre-processing -- #

def snv(input_data):
    # Define a new array and populate it with the corrected data
    output_data = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        # Apply correction
        output_data[i, :] = (input_data[i, :] - np.mean(input_data[i, :])) / np.std(input_data[i, :])
    return output_data


# -- Estimations -- #

def estimate_average(data: np.array, mask=None) -> np.array:
    if mask:
        min_val = np.min(data)
        max_val = np.max(data)
        out = (data - min_val) / (max_val - min_val)
        out *= 255
        out = out.astype(np.uint8)
        masked = cv2.bitwise_and(out, out, mask=mask)
        pixels = masked[np.where(masked != 0)]
        # Convert to index quantity
        pixels /= 255
        result = (pixels * (max_val - min_val)) + min_val
        return np.average(result), np.std(result)
    return np.average(data), np.std(data)


def estimate_severity_level(data: np.array) -> np.array:
    '''
    Levels:
    l>=10 - None (0)
    7.5<l<10 - Moderate (1)
    5.0<l<7.5 - High (2)
    l<=5 - Severe (3)
    '''
    vvi = VIndexer(data).get_index('vvi')
    res = np.zeros_like(vvi).astype('uint8')
    res[vvi > 10] = 1
    res[np.logical_and(7.5 < vvi, vvi <= 10)] = 2
    res[np.logical_and(5 < vvi, vvi <= 7.5)] = 3
    res[np.logical_and(3 < vvi, vvi <= 5)] = 4
    return res

