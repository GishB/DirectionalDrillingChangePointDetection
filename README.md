# Time-Series Change Point Detection models for Oil and Gas Industries.

#### version: 0.0.1
- Syth data generator as well as real data available at data dir.
- Refactored models based on KalmanFilter and Singular Value Decomposition technique avalable at model dir.
- Custom fast optimization based on heuristics methods.
- Streamlit example app at examples dir.

#### Notes:
 - Most of the implemented idea/code based on my master thesis. SingularSequenceTransformation and WindowSizeSearch 
optimization 
classes has been refactored based on implementations from ***Fedot.Industrial*** legacy.
 - To score change point detection models functions from ***TSAD*** lib has been used\adopted.

#### Tasks:

## To set up local project dependencies:
```commandline
    python3 setup.py build
```

TO DO:

1. Docker images for example files at examples dir.
2. Hybrid model
3. Evolution optimization for model hyperparameters.
4. default notebook examples
5. FastAPI service for end-to-end use in container.
6. Advanced change point detection models.
7. More tests.

[//]: # (Here you find notebooks with Change Point Detection methods in Petroleum Data. Mainly I focus to experiment with Fedot.Industrial library.)

[//]: # ()
[//]: # (My point is to create an offline CPD algorithm:)

[//]: # ()
[//]: # (  1. Without a priori knowledge of CPs numbers in data.)

[//]: # (  )
[//]: # (  2. With auto-selected parameters.)

[//]: # (  )
[//]: # (  3. Minimum FPs detection.)

[//]: # (  )
[//]: # (  4. Minimum time delta detection.)

[//]: # (  )
[//]: # (  5. Maximum CPs detection.)

[//]: # (  )
[//]: # (  )
[//]: # (in progress...)

[//]: # ()
[//]: # (TO DO list:)

[//]: # ()
[//]: # (1. Refactoring && Restructure dir)

[//]: # (2. Update SST and Kalman Filter models)

[//]: # (3. Update streamlit app)

[//]: # (4. Create Docker Image && CI)

[//]: # (5. Add some tests for models)
