# Automatic Machining Parameter Detection
A CNC controller that learns material characteristics and optimizes its own feeds and speeds.

# Project Status
Nearing completion, collecting test data

## Current Goals
The current goal of this project is to make a system that faces a material using an endmill while simultaneously performing regression on sensor data to complete its model. This model is used to optimize subsequent passes (feedrate and WOC) by means of an objective function that weighs MMR against the chance of failure (deflection, breakage, spindle overload).

## Modeling
Models for forces experienced during the cutting process and models for tool / machine failure are in [software/models.py](software/models.py).

The linear model converges well when given test data sweeps. 

![](assets/6061-sweep-wide_forces.png)

## Hardware
The hardware setup is finished. The machine is a Taig Micro Mill (kindly donated by Ted Hall).

![](assets/machine.jpg)

The spindle motor is an MDX servomotor from Applied Motion products.

![](assets/spindle.jpg)

A 1D tool-force dynamometer was constructed using a Schneeberger frictionless slide and a disc-type preloaded load cell.

![](assets/tfd_1.jpg)

The machine electronics are enclosed for safety.

![](assets/cabinet.jpg)

## Software
The machine controller is in the [software](software/) folder.


