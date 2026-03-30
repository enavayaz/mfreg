## Setup local repository

* init LFS
```bash
git lfs install
```
* get submodules
```bash
git submodule init
git submodule update --recursive
```

## Reuirements
1. For fdasrsf (sqrv/elastic approach):
certifi, cycler, Cython, cffi, numba, joblib, kiwisolver, patsy, pyparsing, python-dateutil, scipy, six, GPy, tornado
2. For mpl_toolkits.basemap:
GEOS, PROJ4 or just (recommended): conda install -c anaconda basemap

## TODOs
1) Geographiclib can be used for spheroid earth: 
from geographiclib.geodesic import Geodesic
import math
geod = Geodesic.WGS84  # define the WGS84 ellipsoid (it is a spheroid)

2) Problem: Time-Interval normalization; Seqs have different lengths!
Is a subsequence of a cyclone also a cyclone? Does it typically happens that 
we start observing a cyclone later then its birth? 
Best fitting model-curve: smoothing, interpolatory or regression? 
Divide long seq into smaller or upsmaple smaller with MaxWinds = 0 (pos = last pos)
at non existing times, meaning it is dead (trivial extension)!!!
Similarily for positions: at endpositions it remains constant.
Do we have to make all seqs same length???
# Geodesic Analysis via Sasaki Metric



## Riemannian metric on the tangent bundle
    A prominent natural metric on the tangent bundle TM of a Riemannian manifold M is the Sasaki metric.
    Its characterization: Canonical projection of TM becomes a Riemannian submersion, parallel vector fields
    along curves are orthogonal to their fibres, and restriction to any tangent space is Euclidean.
## Longitudinal analysis and mean geodesic

## Something on Implementation: Little bug in geomstats/pre_shape.py, thus as standalone in this project??? 

## Literatur
    For Details and computational aspects, see:
-  [A Hierarchical Geodesic Model for Longitudinal Analysis on Manifolds](https://doi.org/10.1007/s10851-022-01079-x)
-  [Sasaki Metrics for Analysis of Longitudinal Data on Manifolds](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4270017)


## Add your files

- [ ] [A Hierarchical Geodesic Model for Longitudinal Analysis on Manifolds](https://doi.org/10.1007/s10851-022-01079-x) and [Add files using the command line](https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://git.zib.de/bzfnavay/elas_sas.git
git branch -M main
git push -uf origin main
```

## Hurricanes
The revised Atlantic hurricane database (HURDAT2) - Chris Landsea – April 2022
The National Hurricane Center (NHC) conducts a post-storm analysis of each tropical cyclone in its area of responsibility to determine the official assessment of the cyclone's history. This analysis makes use of all available observations, including those that may not have been available in real time. In addition, NHC conducts ongoing reviews of any retrospective tropical cyclone analyses brought to its attention, and on a regular basis updates the historical record to reflect changes introduced via the Best Track Change Committee (Landsea et al. 2004a, 2004b, 2008, 2012, Hagen et al. 2012, Kieper et al. 2016, and Delgado et al. 2018). NHC has traditionally disseminated the tropical cyclone historical database in a format known as HURDAT (short for HURricane DATabase – Jarvinen et al. 1984). This report updates the original HURDAT documentation to reflect significant changes since 2012 to both the format and content for the tropical cyclones and subtropical cyclones of the Atlantic basin (i.e., North Atlantic Ocean, Gulf of Mexico, and Caribbean Sea). (Note for April 2022: Radius of Maximum Wind added into HURDAT2 for the first time beginning with the 2021 hurricane season.)
The original HURDAT format substantially limited the type of best track information that could be conveyed. The format of this new version - HURDAT2 (HURricane DATa 2nd generation) - is based upon the “best tracks” available from the b-decks in the Automated Tropical Cyclone Forecast (ATCF – Sampson and Schrader 2000) system database and is described below. Reasons for the revised version include: 1) inclusion of non-synoptic (other than 00, 06, 12, and 18Z) best track times (mainly to indicate landfalls and intensity maxima); 2) inclusion of non-developing tropical depressions; and 3) inclusion of best track wind radii.
An example of the new HURDAT2 format for Hurricane Ida from 2021 follows:
AL092021, IDA, 40,
20210826, 1200, , TD, 16.5N, 78.9W, 30, 1006, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 60
20210826, 1800, , TS, 17.4N, 79.5W, 35, 1006, 60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 50
20210827, 0000, , TS, 18.3N, 80.2W, 40, 1004, 60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 50
20210827, 0600, , TS, 19.4N, 80.9W, 45, 1002, 70, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 40
20210827, 1200, , TS, 20.4N, 81.7W, 55, 996, 80, 60, 0, 60, 30, 0, 0, 0, 0, 0, 0, 0, 30
20210827, 1800, L, HU, 21.5N, 82.6W, 70, 987, 80, 60, 40, 60, 40, 30, 0, 20, 20, 0, 0, 0, 20
20210827, 2325, L, HU, 22.4N, 83.2W, 70, 988, 80, 60, 40, 60, 40, 30, 0, 20, 20, 0, 0, 0, 20
20210828, 0000, , HU, 22.6N, 83.5W, 70, 989, 100, 60, 40, 70, 50, 30, 0, 30, 20, 0, 0, 0, 20
20210828, 0600, , HU, 23.5N, 84.7W, 70, 987, 100, 60, 40, 70, 50, 30, 0, 30, 20, 0, 0, 0, 20
20210828, 1200, , HU, 24.4N, 85.7W, 70, 986, 110, 80, 60, 100, 50, 40, 20, 30, 25, 20, 0, 0, 20
20210828, 1800, , HU, 25.6N, 86.6W, 80, 976, 110, 100, 70, 100, 50, 40, 20, 40, 25, 20, 10, 20, 20
20210829, 0000, , HU, 26.7N, 87.6W, 90, 967, 120, 100, 80, 110, 70, 60, 40, 60, 35, 30, 20, 30, 20
20210829, 0600, , HU, 27.6N, 88.7W, 115, 950, 120, 100, 80, 110, 70, 60, 40, 60, 35, 30, 20, 30, 15
20210829, 1200, , HU, 28.5N, 89.6W, 130, 929, 130, 110, 80, 110, 70, 60, 40, 60, 45, 35, 20, 30, 10
20210829, 1655, L, HU, 29.1N, 90.2W, 130, 931, 130, 110, 80, 110, 70, 60, 40, 60, 45, 35, 20, 30, 10
20210829, 1800, , HU, 29.2N, 90.4W, 125, 932, 130, 120, 80, 80, 70, 60, 40, 40, 45, 35, 20, 25, 10
20210830, 0000, , HU, 29.9N, 90.6W, 105, 944, 80, 120, 80, 70, 50, 60, 40, 40, 30, 30, 20, 20, 10
20210830, 0600, , HU, 30.6N, 90.8W, 65, 978, 80, 130, 80, 40, 50, 50, 0, 0, 30, 30, 0, 0, 30
20210830, 1200, , TS, 31.5N, 90.9W, 40, 992, 50, 160, 60, 30, 0, 0, 0, 0, 0, 0, 0, 0, 40
20210830, 1800, , TS, 32.2N, 90.5W, 35, 996, 0, 160, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 80
20210831, 0000, , TD, 33.0N, 90.0W, 30, 996, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 250
20210831, 0600, , TD, 33.8N, 89.4W, 25, 996, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 210
20210831, 1200, , TD, 34.4N, 88.4W, 25, 996, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 250

There are two types of lines of data in the new format: the header line and the data lines. The format is comma delimited to maximize its ease in use. The header line has the following format:
AL092021, IDA, 40,
1234567890123456789012345768901234567
AL (Spaces 1 and 2) – Basin – Atlantic
09 (Spaces 3 and 4) – ATCF cyclone number for that year
2021 (Spaces 5-8, before first comma) – Year
IDA (Spaces 19-28, before second comma) – Name, if available, or else “UNNAMED”
40 (Spaces 34-36) – Number of best track entries – rows – to follow
Notes:
1) Cyclone number: In HURDAT2, the order cyclones appear in the file is determined by the date/time of the first tropical or subtropical cyclone record in the best track. This sequence may or may not correspond to the ATCF cyclone number. For example, the 2011 unnamed tropical storm AL20 which formed on 1 September, is sequenced here between AL12 (Katia – formed on 29 Aug) and AL13 (Lee – formed on 2 September). This mismatch between ATCF cyclone number and the HURDAT2 sequencing can occur if post-storm analysis alters the relative genesis times between two cyclones. In addition, in 2011 it became practice to assign operationally unnamed cyclones ATCF numbers from the end of the list, rather than insert them in sequence and alter the ATCF numbers of cyclones previously assigned.
2) Name: Tropical cyclones were not formally named before 1950 and are thus referred to as “UNNAMED” in the database. Systems that were added into the database after the season (such as AL20 in 2011) also are considered “UNNAMED”. Non-developing tropical depressions formally
3) 