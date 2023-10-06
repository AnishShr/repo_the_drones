# repo_the_drones
Repository containing all necessary files for the course: Perception and Control of Quadrotor drones

copy the **pid_drones** directory inside the src folder, or clone the entire repo inside the catkin workspace, build the package and source the workspace. You should be good to go.

Inside the src folder of the catkin ws, run the following command:   
`git clone https://github.com/AnishShr/repo_the_drones.git .`   

Once the workspace is built and sourced:   
`rosrun pid_drones drone_evaluation.py`   

Inside the file `/pid_drones/src/drone_evaluation.py`, you can set if you are trying to run it in simulation or hardware, and even give the path of reference trajectories and the starting position of the drone.

**N.B. You should have all other necesary packages as instructed in moodle:** https://moodle.ut.ee/pluginfile.php/2228759/mod_folder/content/0/instruction_nav_stack_real_drone_part1.pdf?forcedownload=1