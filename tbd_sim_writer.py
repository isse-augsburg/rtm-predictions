# Does not work
# def run_slurm(self):
#     for e in self.slurm_scripts:
#         call = f'ssh swt-clustermanager sbatch ~/slurm_scripts/{e}'
#         c = shlex.split(call)
#         p = subprocess.Popen(c, shell=True, stdout=subprocess.PIPE)
#         std, err = p.communicate()
#         print(std)
#         print(err)


# Does not work
# def solve_simulations(self, selfpath):
#     print('Unsave function: solve_simulations()\n Use slurm.')
#     exit(-1)
#     unfs = []
#     finished = set()
#     for root, dirs, files in os.walk(path, topdown=True):
#         for fn in files:
#             if fn[-5:] == "g.unf":
#                 unfs.append(os.path.join(root, fn))
#             if fn[-6:] == '.erfh5':
#                 finished.add(root)
#     solver = r'C:\Program Files\ESI Group\PAM-COMPOSITES\2019.0\RTMSolver\bin\pamcmxdmp.bat'
#     for e in unfs:
#         if os.path.split(e)[:-1][0] in finished:
#             print('Already finished/Started:', os.path.split(e)[:-1][0])
#             continue
#         print('Starting:', os.path.split(e)[:-1][0])
#         call_solve = rf'''"{solver}" -np 12 "{e}"'''
#         args = shlex.split(call_solve)
#         process = subprocess.Popen(args, shell=True, stdout=subprocess.PIPE)
#         output, error = process.communicate()
#         print(output)