
# Vegetation API
### Hyperspectral indexes calculation

**API that apply vegetation indexes onto hyperspectral images**

The ability to measure the intensity of electromagnetic radiation in different wavelength ranges, after its interaction with the material, is the cornerstone of remote monitoring and image spectroscopy. These interactions, when evaluated either through the analysis of the reflection, absorption or transmission of radiation, make it possible to determine what is known as the spectral signature of the material in question. Vegetation also interacts in a particular way with solar radiation. It absorbs a lot in blue and red, and reflects intensely at near-infrared (NIR) wavelengths. Measuring these variations and the way they are related can contribute to a better assessment of the phytosanitary status of crops. These relationships are often described as vegetation indices (VIs). 

24 indices were selected, firstly, taking into account the detection range of a VNIR camera (400nm-1000nm), and secondly, for their robustness, scientific basis and applicability. Additionally, A custom vegetation index was created, called VVI, to increase sensitivity in the various stages of plant development and reach a high level of accuracy in the detection of anomalies.

## How-to

Clone the repo, open a terminal and go to directory. Just run:
>`$ docker compose up -d`

depending on docker compose version you are using, you may need to write `docker-compose` instead. The command `up` will build both containers and put them up and running, the `-d` runs the command in detached mode.
You may write now in the same terminal, the command `docker compose ps` to check the active services.

A dashboard has been added to enable you to explore the api and you start exploring your awesome data! 
The dashboard was made using streamlit and it is exposed in port 8501. Just open a browser and write:
> `localhost:8501`

Have a fun reasearch!
