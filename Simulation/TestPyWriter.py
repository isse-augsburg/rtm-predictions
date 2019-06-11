import unittest
from pathlib import Path

import h5py
import pandas

from Simulation.SimCreator import SimCreator, OutputFrequencyType


class TestPyWriter(unittest.TestCase):
    pywriter_reference = r'''#!/usr/bin/env python
null=''
from VgPoint3 import *
from VgPoint2 import *
from VgMatrix import *
import VScn
import VGuiUtl
import VBrowserManager
import VExpMngr
import VCmdGui
import VCmd
import VCmdFramework
import VMaterial
import VMeshMdlr
import VToolKit
import VistaDb
import VHostManager
import VE
NULL=VistaDb.PythonCNULL()
import VistaDb
var1=VCmd.Activate( 1, r"VHostManagerPlugin.VhmInterface", r"VhmCommand" )
var2=VCmd.Activate( 1, r"VSessionManager.Command", r"SessionCommand" )
ret=VExpMngr.LoadFile( r"Y:\data\RTM\Lautern\sources\flawless_one_layer.vdb", 0 )
ret=VHostManager.ChangeContext( r"Visual-RTM" )
var3=VCmd.Activate( 1, r"VRTMUtilities.VRTMInterface", r"SimulationParameters" )
VCmd.SetStringValue( var3, r"MaterialDB", r"Model" )
VCmd.SetStringValue( var3, r"InjectedResin", r"RapeSeedOil" )
VCmd.SetDoubleValue( var3, r"MaxInjectionTime", 800000.  )

VCmd.SetStringValue( var3, r"OutputFrequencyType", r"Time" )
VCmd.SetDoubleValue( var3, r"OutputFrequency", 0.5  )
VCmd.Accept( var3 )
VCmd.Quit( var3 )
var4=VCmd.Activate( 1, r"VRTMUtilities.VRTMInterface", r"ImportExportLocalProperties" )
VCmd.SetGuStringValue( var4, r"ImportFileName", r"X:\s\t\stiebesi\code\tests\solver_input_folder\0\2000-01-01_00-00-00_0.lperm" )
ret=VCmd.ExecuteCommand( var4, r"ImportLocalResults" )
VExpMngr.ExportFile( r"X:\s\t\stiebesi\code\tests\solver_input_folder\0\2000-01-01_00-00-00_0.vdb", 0 )
var5=VCmd.Activate( 1, r"VRTMUtilities.VRTMInterface", r"ImportExportLocalProperties" )
VCmd.SetGuStringValue( var5, r"ImportFileName", r"X:\s\t\stiebesi\code\tests\solver_input_folder\1\2000-01-01_00-00-00_1.lperm" )
ret=VCmd.ExecuteCommand( var5, r"ImportLocalResults" )
VExpMngr.ExportFile( r"X:\s\t\stiebesi\code\tests\solver_input_folder\1\2000-01-01_00-00-00_1.vdb", 0 )
var6=VCmd.Activate( 1, r"VRTMUtilities.VRTMInterface", r"ImportExportLocalProperties" )
VCmd.SetGuStringValue( var6, r"ImportFileName", r"X:\s\t\stiebesi\code\tests\solver_input_folder\2\2000-01-01_00-00-00_2.lperm" )
ret=VCmd.ExecuteCommand( var6, r"ImportLocalResults" )
VExpMngr.ExportFile( r"X:\s\t\stiebesi\code\tests\solver_input_folder\2\2000-01-01_00-00-00_2.vdb", 0 )
var7=VCmd.Activate( 1, r"VRTMUtilities.VRTMInterface", r"ImportExportLocalProperties" )
VCmd.SetGuStringValue( var7, r"ImportFileName", r"X:\s\t\stiebesi\code\tests\solver_input_folder\3\2000-01-01_00-00-00_3.lperm" )
ret=VCmd.ExecuteCommand( var7, r"ImportLocalResults" )
VExpMngr.ExportFile( r"X:\s\t\stiebesi\code\tests\solver_input_folder\3\2000-01-01_00-00-00_3.vdb", 0 )
var8=VCmd.Activate( 1, r"VRTMUtilities.VRTMInterface", r"ImportExportLocalProperties" )
VCmd.SetGuStringValue( var8, r"ImportFileName", r"X:\s\t\stiebesi\code\tests\solver_input_folder\4\2000-01-01_00-00-00_4.lperm" )
ret=VCmd.ExecuteCommand( var8, r"ImportLocalResults" )
VExpMngr.ExportFile( r"X:\s\t\stiebesi\code\tests\solver_input_folder\4\2000-01-01_00-00-00_4.vdb", 0 )
var9=VCmd.Activate( 1, r"VRTMUtilities.VRTMInterface", r"ImportExportLocalProperties" )
VCmd.SetGuStringValue( var9, r"ImportFileName", r"X:\s\t\stiebesi\code\tests\solver_input_folder\5\2000-01-01_00-00-00_5.lperm" )
ret=VCmd.ExecuteCommand( var9, r"ImportLocalResults" )
VExpMngr.ExportFile( r"X:\s\t\stiebesi\code\tests\solver_input_folder\5\2000-01-01_00-00-00_5.vdb", 0 )
var10=VCmd.Activate( 1, r"VRTMUtilities.VRTMInterface", r"ImportExportLocalProperties" )
VCmd.SetGuStringValue( var10, r"ImportFileName", r"X:\s\t\stiebesi\code\tests\solver_input_folder\6\2000-01-01_00-00-00_6.lperm" )
ret=VCmd.ExecuteCommand( var10, r"ImportLocalResults" )
VExpMngr.ExportFile( r"X:\s\t\stiebesi\code\tests\solver_input_folder\6\2000-01-01_00-00-00_6.vdb", 0 )
var11=VCmd.Activate( 1, r"VRTMUtilities.VRTMInterface", r"ImportExportLocalProperties" )
VCmd.SetGuStringValue( var11, r"ImportFileName", r"X:\s\t\stiebesi\code\tests\solver_input_folder\7\2000-01-01_00-00-00_7.lperm" )
ret=VCmd.ExecuteCommand( var11, r"ImportLocalResults" )
VExpMngr.ExportFile( r"X:\s\t\stiebesi\code\tests\solver_input_folder\7\2000-01-01_00-00-00_7.vdb", 0 )
var12=VCmd.Activate( 1, r"VRTMUtilities.VRTMInterface", r"ImportExportLocalProperties" )
VCmd.SetGuStringValue( var12, r"ImportFileName", r"X:\s\t\stiebesi\code\tests\solver_input_folder\8\2000-01-01_00-00-00_8.lperm" )
ret=VCmd.ExecuteCommand( var12, r"ImportLocalResults" )
VExpMngr.ExportFile( r"X:\s\t\stiebesi\code\tests\solver_input_folder\8\2000-01-01_00-00-00_8.vdb", 0 )
var13=VCmd.Activate( 1, r"VRTMUtilities.VRTMInterface", r"ImportExportLocalProperties" )
VCmd.SetGuStringValue( var13, r"ImportFileName", r"X:\s\t\stiebesi\code\tests\solver_input_folder\9\2000-01-01_00-00-00_9.lperm" )
ret=VCmd.ExecuteCommand( var13, r"ImportLocalResults" )
VExpMngr.ExportFile( r"X:\s\t\stiebesi\code\tests\solver_input_folder\9\2000-01-01_00-00-00_9.vdb", 0 )
'''

    def setUp(self):
        # print('setup')
        perturbation_factors = \
        {
        "General_Sigma": .001,
        "Shapes":
            {
                "Rectangles":
                    {
                        "Num": 1,
                        "Fiber_Content":
                            [.7, .8]
                    },
                "Circles":
                    {
                        "Num": 1,
                        "Fiber_Content": [.7, .8]
                    },
                "Runners":
                    {
                        "Num": 0,
                        "Fiber_Content": [-.7, -.8]
                    }
            }
        }
        self.sc = SimCreator(perturbation_factors=perturbation_factors)
        self.sc.initial_timestamp = '2000-01-01_00-00-00'
        self.sc.slurm_scripts_folder = Path(r'X:\s\t\stiebesi\code\tests\slurm_writer')
        self.sc.solver_input_folder = Path(r'X:\s\t\stiebesi\code\tests\solver_input_folder')

    def test_write_solver_input(self):
        # print('write')
        self.sc.create_folder_structure_and_perturbate_kN()
        self.sc.write_solver_input()
        with open(self.sc.solver_input_folder / 'vdb_writerpy2.py') as f:
            out_lines = f.readlines()
            reference_lines = self.pywriter_reference.splitlines(keepends=True)
            for a, b in zip(out_lines, reference_lines):
                self.assertEqual(a, b)

    def test_create_folder_structure_and_perturbate_kN(self):
        # print('create')
        desired_num_elements = 137992
        self.sc.create_folder_structure_and_perturbate_kN()
        lperms = self.sc.solver_input_folder.glob('**/*.lperm')
        dfs = [pandas.read_csv(x, sep='\t') for x in lperms]
        [self.assertEqual(len(x), desired_num_elements) for x in dfs]
        metas = self.sc.solver_input_folder.glob('**/*_meta_data.hdf5')
        for meta in metas:
            f = h5py.File(meta, 'r')
            self.assertEqual(len(f.keys()), 3)
            f.close()

    def tearDown(self):
        all_files = self.sc.solver_input_folder.glob('**/*')
        [x.unlink() for x in all_files if x.is_file()]
        all_files = self.sc.solver_input_folder.glob('**/*')
        [x.rmdir() for x in all_files if x.is_dir()]

if __name__ == '__main__':
    unittest.main()
