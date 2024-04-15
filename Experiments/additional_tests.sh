#!/bin/bash

# PLOTTING THE NS SNAPSHOTS AND POINT TIME EVOLUTION:

#python Test.py -bench 1 -rollout 4 -model 15 -use_dts 1,2,4,8 -samples 1 -ready -ylim 0,5 -seed 0 -point
#python Test.py -bench 1 -rollout 4 -model 15 -use_dts 1,2,4,8 -samples 1 -ready -ylim 0,5 -seed 1 -point

#python Test.py -bench 2 -rollout 4 -model 15 -use_dts 1,2,4 -samples 1 -ready -ylim 0,20 -seed 0 -point
#python Test.py -bench 2 -rollout 4 -model 15 -use_dts 1,2,4 -samples 1 -ready -ylim 0,20 -seed 1 -point

#python Test.py -bench 3 -rollout 4 -model 15 -use_dts 1,2 -samples 1 -ready -ylim 0,20 -seed 0 -point
#python Test.py -bench 3 -rollout 4 -model 15 -use_dts 1,2 -samples 1 -ready -ylim 0,20 -seed 1 -point

#=================================================================================================================

# PLOTTING THE KF SNAPSHOTS AND POINT TIME EVOLUTION:

#python Test.py -bench 7 -rollout 8 -model 1 -dts 1 -use_dts 1 -samples 1 -curves -final -ylim 0,5 -start_end 0,40 -seed 1 -show_error
#python Test.py -bench 7 -rollout 8 -model 1 -dts 1 -use_dts 1 -samples 1 -curves -final -ylim 0,5 -start_end 50,90 -seed 1 -show_error
#python Test.py -bench 7 -rollout 8 -model 1 -dts 1 -use_dts 1 -samples 1 -curves -final -ylim 0,5 -start_end 0,40 -seed 2 -show_error
#python Test.py -bench 7 -rollout 8 -model 1 -dts 1 -use_dts 1 -samples 1 -curves -final -ylim 0,5 -start_end 50,90 -seed 2 -show_error

#=================================================================================================================

# PLOTTING THE OUTPUT OVER A VERY LONG TIME:

#python Test.py -bench 1 -rollout 4 -model 15 -use_dts 1,2,4,8 -samples 1 -ready -ylim 0,5 -seed 0 -overtime -start_end 9,110 -cols 10
#python Test.py -bench 1 -rollout 4 -model 15 -use_dts 1,2,4,8 -samples 1 -ready -ylim 0,5 -seed 1 -overtime -start_end 9,110 -cols 10

#python Test.py -bench 2 -rollout 4 -model 15 -use_dts 1,2,4 -samples 1 -ready -ylim 0,20 -seed 0 -overtime -start_end 9,60 -cols 10
#python Test.py -bench 2 -rollout 4 -model 15 -use_dts 1,2,4 -samples 1 -ready -ylim 0,20 -seed 1 -overtime -start_end 9,60 -cols 10

#python Test.py -bench 3 -rollout 4 -model 15 -use_dts 1,2 -samples 1 -ready -ylim 0,20 -seed 0 -overtime -start_end 9,35 -cols 10
#python Test.py -bench 3 -rollout 4 -model 15 -use_dts 1,2 -samples 1 -ready -ylim 0,20 -seed 1 -overtime -start_end 9,35 -cols 10


#python Test.py -bench 7 -rollout 8 -model 1 -dts 1 -use_dts 1 -samples 1 -curves -final -ylim 0,5 -start_end 0,300 -seed 1 -overtime -cols 10
#python Test.py -bench 7 -rollout 8 -model 1 -dts 1 -use_dts 1 -samples 1 -curves -final -ylim 0,5 -start_end 0,300 -seed 2 -overtime -cols 10


#=================================================================================================================

# ENERGY SPECTRUM:
# python Test.py -bench 7 -rollout 8 -model 1 -dts 1 -use_dts 1 -samples 40 -start_end 0,50 -seed 1 -ES

# python Test.py -bench 7 -rollout 8 -model 1 -dts 1 -use_dts 1 -samples 40 -start_end 50,100 -seed 1 -ES

# python Test.py -bench 7 -rollout 8 -model 1 -dts 1 -use_dts 1 -samples 40 -start_end 0,100 -seed 1 -ES