# How to use Dry Spot detection

Enter a path wich shall be scanned in the code.
Run it and keep the stdout.
Use the following snippets to get out the runs with atypical behaviour:
 cat *_dry_spot_out.txt | grep '\[\]' | grep -x '.\{220,600\}' > dry_spot_out_filtered_*.txt
 
 cat dry_spot_out_filtered_*.txt | cut -c 30-59 > blacklist.txt
 
The numbers for the cut command may need some adjusting.