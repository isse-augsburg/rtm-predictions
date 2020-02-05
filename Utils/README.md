# How to use Dry Spot detection

Enter a path wich shall be scanned in the code.
Run it and keep the stdout.
Use the following snippets to get out the runs with atypical behaviour:
```
cat dryspots.txt | grep '\[\]' | grep -x '.\{220,600\}' > dry_spot_out_filtered.txt
 
cat dry_spot_out_filtered_*.txt | cut -c 30-59 > blacklist.txt
```
 
 
The numbers for the cut command may need some adjusting.

# Eval Utils

The SINGULARITY_DOCKER_PASSWORD is inserted automatically now.
Make sure to insert the current docker container: Current state of the art: pytorch_19.10