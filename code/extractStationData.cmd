@echo off
title Extract daily maxima from 1-minute observations
CALL conda.bat activate tsed

python %CD%\extractStationData.py -c %CD%\extract_allevents.ini