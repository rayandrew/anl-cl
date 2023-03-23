Avalanche framework
- has set of tasks
- you can specify the tasks sequence or just pure random
- when we do optimization, we put take sequence as an inputs
- old version of Avalanche can work
    - if you want use the newest version of the Avalanche, then we can do it
    - we need to run and compile new version of Avalanche in Argonne framework
    - Argonne Polaris is similar to Argonne Aurora
- let us create github repo and push the code
    - main thing we will need: couple the optimization framework such as DeepHyper
    - task variability that presents in the algorithm
    - very restricted on the subset of tasks
    - all documented algorithms performance are suffered from task sequence
    - how do we learn st we equip the network to reduce the variance of prediction


- This is a timeseries problems
- Will we able to learn CL in this setup?
  - This is a not sequence problems

- Core problems: 
  - temporal correlation
  - concept drift detection
  - CL for sequential data
    - how do you input the data into the model

- In the data for CORI / Argonne
  - they are actually prediction times for separate jobs
  - it is as not as correlated as Alibaba

- What would help me better predicting that?
- Q: given scheduler information
- Predicting
  - CPU util of system -> global
  - Need indicators to be global
    - Need to look at mkpi, net_in, etc.
  - Scheduler information
    - Tell us local information about particular app
    - Can have some effects on global utilization
    - We do not have information
    - We can get:
      - \# of jobs at given timestamp -> this is global number
      - LIst of running jobs

- Read EVOGWP
  - understand what does it mean by workload

- What we want: take global params and learn from that
  - There are two cases of temporal information

- Are we gonna leverage sequential information?