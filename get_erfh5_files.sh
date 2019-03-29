#!/bin/bash 

start_dir=//137.250.170.56/share/data/RTM/Lautern/2_auto_solver_inputs/
target_dir=/home/lodes/Sim_Results

clear
echo "Please enter your name!"
read name 
echo "Hello, $name, i am Karen, a super annoying AI trying to help you."
echo "The curent start directory is $start_dir. Is this correct? [no/yes]" 
read answer 

while [ $answer = no ] 
do 
	echo "Alright, $name, please enter a new directory" 
	read start_dir
	echo "Alright, $name, you entered $start_dir. Is this correct? [no/yes]"

	read answer 
done 

echo "Good, $name. The current target directory is $target_dir. Is this correct? [no/yes]" 
read answer 

while [ $answer = no ] 
do 
	echo "Alright, $name, please enter a new target directory" 
	read target_dir
	echo "Alright, $name, you entered $target_dir. Is this correct? [no/yes]"

	read answer 
done 

echo "Good, $name, lets do what has to be done" 

#find $start_dir -name "*.erfh5" -exec cp {} $target_dir \;

find . -name "*.erfh5"